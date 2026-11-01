"""Local-first project similarity checker.

Run from any directory with ``python backend/server.py``. The saved corpus is
searched in full. Semantic ranking is optional; keyword ranking works with
Python's standard library when optional model packages are unavailable.
"""

from __future__ import annotations

import json
import math
import os
import re
import webbrowser
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
FRONTEND_FILE = ROOT / "frontend" / "index.html"
MAX_IDEA_CHARS = 5_000
MAX_REQUEST_BYTES = 64_000
RESULT_LIMIT = 10
MODEL_NAME = "BAAI/bge-small-en-v1.5"
TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9+#._-]*", re.IGNORECASE)
STOP_WORDS = {
    "a", "about", "an", "and", "are", "as", "at", "be", "by", "for",
    "from", "in", "into", "is", "it", "of", "on", "or", "our", "the",
    "their", "this", "to", "using", "with", "your", "you", "we", "will",
}


def tokens(text: str) -> list[str]:
    return [word.lower() for word in TOKEN_RE.findall(text) if word.lower() not in STOP_WORDS]


def project_terms(project: dict[str, Any]) -> Counter[str]:
    terms: Counter[str] = Counter()
    terms.update({word: 4 for word in tokens(str(project.get("title") or ""))})
    terms.update({word: 2 for word in tokens(str(project.get("description") or ""))})
    terms.update({word: 2 for word in tokens(" ".join(project.get("topics") or []))})
    terms.update(tokens(str(project.get("readme") or "")[:4_000]))
    return terms


def clean_project(item: Any) -> dict[str, Any] | None:
    if not isinstance(item, dict):
        return None
    name = str(item.get("full_name") or item.get("name") or item.get("title") or "").strip()
    if not name:
        return None
    raw_url = str(item.get("url") or "").strip()
    expected_url = f"https://github.com/{name}"
    url = raw_url if raw_url.startswith("https://github.com/") else expected_url
    topics = item.get("topics") or []
    if not isinstance(topics, list):
        topics = []
    return {
        "name": name,
        "url": url,
        "description": str(item.get("description") or "").strip(),
        "topics": [str(topic) for topic in topics if topic],
        "language": str(item.get("language") or "").strip(),
        "stars": item.get("stars", item.get("stargazers_count")),
        "readme": str(item.get("readme") or ""),
    }


def read_corpus() -> list[dict[str, Any]]:
    for path in (DATA_DIR / "index_projects.json", DATA_DIR / "projects.json"):
        if not path.is_file():
            continue
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Could not read project data at {path}: {exc}") from exc
        if not isinstance(raw, list):
            raise RuntimeError(f"Project data at {path} must contain a JSON list.")
        projects = [project for item in raw if (project := clean_project(item))]
        if projects:
            return projects
    raise RuntimeError(f"No project corpus found in {DATA_DIR}. Add index_projects.json or projects.json.")


def build_lexical_index(projects: list[dict[str, Any]]) -> tuple[list[Counter[str]], dict[str, float], list[float]]:
    documents = [project_terms(project) for project in projects]
    doc_frequency: Counter[str] = Counter()
    for document in documents:
        doc_frequency.update(document.keys())
    count = max(1, len(documents))
    idf = {term: math.log((count + 1) / (frequency + 1)) + 1 for term, frequency in doc_frequency.items()}
    vectors: list[Counter[str]] = []
    norms: list[float] = []
    for document in documents:
        vector = Counter({term: (1 + math.log(freq)) * idf[term] for term, freq in document.items()})
        vectors.append(vector)
        norms.append(math.sqrt(sum(value * value for value in vector.values())))
    return vectors, idf, norms


class ProjectIndex:
    def __init__(self) -> None:
        self.projects = read_corpus()
        self.lexical_vectors, self.idf, self.lexical_norms = build_lexical_index(self.projects)
        self.np = None
        self.embedding_vectors = None
        self.embedding_norms = None
        self.embedder = None
        self.mode = "keyword similarity"
        self._try_load_semantic_index()

    def _try_load_semantic_index(self) -> None:
        if os.getenv("PROJECT_CHECKER_SEMANTIC", "").strip().lower() not in {"1", "true", "yes", "on"}:
            return
        try:
            import numpy as np
            from fastembed import TextEmbedding

            vectors = np.load(DATA_DIR / "vectors.npy")
            if vectors.ndim != 2 or vectors.shape[0] != len(self.projects):
                raise ValueError("vectors.npy does not line up with the project corpus")
            norms = np.linalg.norm(vectors, axis=1)
            if not np.isfinite(vectors).all() or (norms == 0).any():
                raise ValueError("vectors.npy contains invalid values")
            self.np = np
            self.embedding_vectors = vectors
            self.embedding_norms = norms
            self.embedder = TextEmbedding(MODEL_NAME)
            self.mode = "semantic embeddings"
        except Exception as exc:
            # A missing optional model must never keep the checker from starting.
            print(f"Semantic search unavailable; using keyword similarity ({exc})")

    def _semantic_scores(self, idea: str) -> list[float] | None:
        if self.embedder is None or self.np is None:
            return None
        try:
            query = self.np.asarray(list(self.embedder.embed([idea]))[0], dtype=float)
            query_norm = float(self.np.linalg.norm(query))
            if query_norm == 0:
                return None
            values = self.embedding_vectors @ query / (self.embedding_norms * query_norm)
            return [max(0.0, min(1.0, float(value))) for value in values]
        except Exception as exc:
            print(f"Semantic query failed; using keyword similarity ({exc})")
            return None

    def _keyword_scores(self, idea: str) -> list[float]:
        query_terms: Counter[str] = Counter(tokens(idea))
        if not query_terms:
            return [0.0] * len(self.projects)
        query_vector = {term: (1 + math.log(freq)) * self.idf.get(term, 1.0) for term, freq in query_terms.items()}
        query_norm = math.sqrt(sum(value * value for value in query_vector.values()))
        scores = []
        for document, doc_norm in zip(self.lexical_vectors, self.lexical_norms):
            if not doc_norm or not query_norm:
                scores.append(0.0)
                continue
            dot = sum(weight * document.get(term, 0.0) for term, weight in query_vector.items())
            scores.append(max(0.0, min(1.0, dot / (query_norm * doc_norm))))
        return scores

    def _analysis(self, idea: str, matches: list[dict[str, Any]]) -> dict[str, Any]:
        query_terms = list(dict.fromkeys(tokens(idea)))
        comparisons = []
        for match in matches[:5]:
            project = self.projects[match["_index"]]
            indexed_terms = set(tokens(" ".join([
                project["name"], project["description"], " ".join(project["topics"]), project["readme"][:4_000]
            ])))
            shared = [term for term in query_terms if term in indexed_terms][:6]
            idea_only = [term for term in query_terms if term not in indexed_terms][:6]
            comparisons.append({
                "project": match["name"],
                "same": shared or ["No clear shared terms in the saved project text."],
                "different": idea_only or ["No obvious idea-specific terms distinguish this description."],
            })

        top_score = matches[0]["score"] if matches else 0
        if self.mode == "semantic embeddings":
            duplicate_cutoff, overlap_cutoff = 82, 55
        else:
            # Cosine scores from sparse keyword vectors are much lower than
            # embedding scores; these separate a strong text match from a
            # merely related description.
            duplicate_cutoff, overlap_cutoff = 40, 12
        if top_score >= duplicate_cutoff:
            verdict = "duplicate"
            lead = "A close description match was found"
        elif top_score >= overlap_cutoff:
            verdict = "overlapping"
            lead = "Related project descriptions were found"
        else:
            verdict = "distinct"
            lead = "No close description match was found"
        top_name = matches[0]["name"] if matches else "the indexed projects"
        top_detail = f" The closest result is {top_name} at {top_score}% text similarity." if matches else ""
        return {
            "verdict": verdict,
            "summary": f"{lead} in the local collection of {len(self.projects)} projects.{top_detail} This is a discovery signal, not proof that two projects are identical.",
            "comparisons": comparisons,
            "suggestions": [
                "Choose a specific user group or setting that the closest result does not describe.",
                "Add a testable capability that is not mentioned in the closest result's indexed text.",
                "Open the matching repositories and compare their actual features before deciding your idea is already built.",
            ],
            "note": "Differences are based on saved descriptions, topics, and README text; a missing mention does not prove a feature is absent.",
        }

    def search(self, idea: str) -> dict[str, Any]:
        scores = self._semantic_scores(idea)
        if scores is None:
            self.mode = "keyword similarity"
            scores = self._keyword_scores(idea)
        else:
            self.mode = "semantic embeddings"

        ranked = sorted(range(len(scores)), key=scores.__getitem__, reverse=True)
        matches = []
        for index in ranked:
            score = scores[index]
            if score <= 0:
                continue
            project = self.projects[index]
            matches.append({
                "name": project["name"],
                "url": project["url"],
                "score": round(score * 100),
                "description": project["description"] or "No description saved.",
                "topics": project["topics"][:8],
                "language": project["language"],
                "stars": project["stars"],
                "_index": index,
            })
            if len(matches) == RESULT_LIMIT:
                break
        analysis = self._analysis(idea, matches)
        for match in matches:
            match.pop("_index", None)
        return {
            "matches": matches,
            "analysis": analysis,
            "projects_scanned": len(self.projects),
            "results_returned": len(matches),
            "search_mode": self.mode,
            "corpus_scope": "Local project descriptions, topics, and saved README text",
        }


try:
    INDEX = ProjectIndex()
except Exception as exc:
    INDEX = None
    STARTUP_ERROR = str(exc)
else:
    STARTUP_ERROR = ""


class RequestHandler(BaseHTTPRequestHandler):
    server_version = "ProjectChecker/1.0"

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, status: int, data: dict[str, Any]) -> None:
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self._send(status, body, "application/json; charset=utf-8")

    def do_GET(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
        path = urlparse(self.path).path
        if path in {"/health", "/api/health"}:
            self._send_json(200 if INDEX else 503, {
                "ok": INDEX is not None,
                "projects_indexed": len(INDEX.projects) if INDEX else 0,
                "results_per_check": RESULT_LIMIT,
                "search_mode": INDEX.mode if INDEX else "unavailable",
                "error": STARTUP_ERROR or None,
            })
            return
        if path not in {"/", "/index.html"}:
            self._send_json(404, {"error": "Not found"})
            return
        try:
            self._send(200, FRONTEND_FILE.read_bytes(), "text/html; charset=utf-8")
        except OSError as exc:
            self._send_json(500, {"error": f"Could not load the interface: {exc}"})

    def do_POST(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
        if urlparse(self.path).path not in {"/check", "/api/check"}:
            self._send_json(404, {"error": "Not found"})
            return
        if INDEX is None:
            self._send_json(503, {"error": STARTUP_ERROR or "Project data could not be loaded."})
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._send_json(400, {"error": "Invalid request size."})
            return
        if size <= 0 or size > MAX_REQUEST_BYTES:
            self._send_json(413, {"error": f"Send an idea under {MAX_IDEA_CHARS} characters."})
            return
        try:
            payload = json.loads(self.rfile.read(size).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._send_json(400, {"error": "Request must contain valid JSON."})
            return
        idea = payload.get("text", "") if isinstance(payload, dict) else ""
        if not isinstance(idea, str):
            self._send_json(400, {"error": "Idea must be text."})
            return
        idea = idea.strip()
        if len(idea) < 15:
            self._send_json(400, {"error": "Describe your idea in at least 15 characters."})
            return
        if len(idea) > MAX_IDEA_CHARS:
            self._send_json(413, {"error": f"Keep your idea under {MAX_IDEA_CHARS} characters."})
            return
        self._send_json(200, INDEX.search(idea))

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[{self.log_date_time_string()}] {fmt % args}")


def main() -> None:
    if INDEX is None:
        raise SystemExit(f"Project checker could not start: {STARTUP_ERROR}")
    host = os.getenv("HOST", "127.0.0.1")
    try:
        port = int(os.getenv("PORT", "8000"))
    except ValueError as exc:
        raise SystemExit("PORT must be a number.") from exc
    # Prefer the familiar port but recover gracefully if another app is using it.
    candidates = [port] if os.getenv("PORT") else range(port, min(port + 21, 65536))
    server = None
    last_error = None
    for candidate in candidates:
        try:
            server = ThreadingHTTPServer((host, candidate), RequestHandler)
            break
        except OSError as exc:
            last_error = exc
    if server is None:
        raise SystemExit(f"Could not start Project Checker on {host}:{port}: {last_error}")
    actual_port = server.server_address[1]
    url_host = "127.0.0.1" if host in {"0.0.0.0", ""} else host
    url = f"http://{url_host}:{actual_port}"
    print(f"Project Checker is ready at {url}")
    print(f"Loaded {len(INDEX.projects)} projects; using {INDEX.mode}.")
    print("Press Ctrl+C to stop.")
    if os.getenv("PROJECT_CHECKER_OPEN_BROWSER", "1").strip().lower() not in {"0", "false", "no", "off"}:
        try:
            webbrowser.open(url)
        except Exception as exc:
            print(f"Could not open a browser automatically; visit {url} ({exc})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping Project Checker.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

