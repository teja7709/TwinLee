import os, json, numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
from fastembed import TextEmbedding
from google import genai
from google.genai import types

load_dotenv()
client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY"),
    http_options=types.HttpOptions(timeout=45000),
)
MODEL = "gemini-flash-lite-latest"

vectors = np.load("../data/vectors.npy")
norms = np.linalg.norm(vectors, axis=1)
with open("../data/index_projects.json", encoding="utf-8") as f:
    projects = json.load(f)
embedder = TextEmbedding("BAAI/bge-small-en-v1.5")

app = FastAPI()

class Idea(BaseModel):
    text: str

PROMPT = """You compare a NEW project idea with 5 EXISTING projects.
Use only the text given. Return ONLY JSON:
{{"verdict": "duplicate" | "overlapping" | "distinct",
  "summary": "2 sentences",
  "comparisons": [{{"project": "name", "same": ["..."], "different": ["..."]}}],
  "suggestions": ["3 specific ways to make the idea more unique"]}}

NEW IDEA: {idea}

EXISTING PROJECTS:
{items}
"""

def explain(idea, matches):
    items = "\n".join(
        f'- {m["name"]}: {m["description"]} | topics: {", ".join(m["topics"])} | {m["readme"]}'
        for m in matches
    )
    try:
        r = client.models.generate_content(
            model=MODEL,
            contents=PROMPT.format(idea=idea, items=items),
            config=types.GenerateContentConfig(
                response_mime_type="application/json", temperature=0),
        )
        return json.loads(r.text)
    except Exception as e:
        print("Gemini failed:", str(e)[:100])
        return None

@app.post("/check")
def check(idea: Idea):
    q = np.array(list(embedder.embed([idea.text])))[0]
    scores = vectors @ q / (norms * np.linalg.norm(q))
    matches = []
    for i in scores.argsort()[::-1][:5]:
        p = projects[i]
        matches.append({
            "name": p["full_name"], "url": p["url"],
            "score": max(0, min(100, round((float(scores[i]) - 0.6) / 0.25 * 100))),
            "description": p["description"], "topics": p["topics"],
            "readme": p["readme"][:500],
        })
    top = matches[0]["score"]
    if top < 40:
        return {"matches": matches, "analysis": {
            "verdict": "distinct",
            "summary": "No close match found in the database. Your idea looks new compared to the projects we have.",
            "comparisons": [], "suggestions": []}}
    analysis = explain(idea.text, matches)
    if analysis:
        analysis["verdict"] = "duplicate" if top >= 70 else "overlapping"
    return {"matches": matches, "analysis": analysis}

@app.get("/")
def home():
    return FileResponse("../frontend/index.html")