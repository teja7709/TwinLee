import os, json, time
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
MODEL = "gemini-flash-latest"
MAX_PROJECTS = 300
IN_FILE = "../data/projects.json"
OUT_FILE = "../data/projects_clean.json"

PROMPT = """Read this project text. Return ONLY JSON with these keys:
"problem": one sentence,
"target_users": short text,
"features": list of 3 to 7 short items,
"tech_stack": list of technologies.
Use only what the text says. If something is missing, use "" or [].

TITLE: {title}
DESCRIPTION: {description}
README: {readme}
"""

def extract(p):
    resp = client.models.generate_content(
        model=MODEL,
        contents=PROMPT.format(
            title=p["title"],
            description=p["description"],
            readme=p["readme"][:2500],
        ),
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0,
        ),
    )
    return json.loads(resp.text)

def main():
    with open(IN_FILE, encoding="utf-8") as f:
        projects = json.load(f)

    projects = [p for p in projects if p["description"] or p["readme"]]
    projects = projects[:MAX_PROJECTS]

    done = {}
    if os.path.exists(OUT_FILE):
        with open(OUT_FILE, encoding="utf-8") as f:
            done = {p["full_name"]: p for p in json.load(f)}

    for i, p in enumerate(projects, 1):
        if p["full_name"] in done:
            continue
        try:
            p["clean"] = extract(p)
            done[p["full_name"]] = p
            print(i, "OK", p["full_name"])
        except Exception as e:
            print(i, "FAILED", p["full_name"], str(e)[:80])
            time.sleep(10)
            continue
        time.sleep(4)
        if i % 10 == 0:
            with open(OUT_FILE, "w", encoding="utf-8") as f:
                json.dump(list(done.values()), f, ensure_ascii=False, indent=2)

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(list(done.values()), f, ensure_ascii=False, indent=2)
    print("Saved", len(done), "clean projects")

if __name__ == "__main__":
    main()