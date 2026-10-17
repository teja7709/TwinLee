import os, json, time, requests
from dotenv import load_dotenv

load_dotenv()
HEADERS = {
    "Authorization": f"Bearer {os.getenv('GITHUB_TOKEN')}",
    "Accept": "application/vnd.github+json",
}
TOPICS = ["hackathon", "chatbot", "healthcare",
          "education", "agritech", "fintech", "ai-assistant"]

def search_repos(topic, pages=3):
    repos = []
    for page in range(1, pages + 1):
        r = requests.get(
            "https://api.github.com/search/repositories",
            headers=HEADERS,
            params={"q": f"topic:{topic} stars:>3",
                    "sort": "stars", "per_page": 50, "page": page},
        )
        if r.status_code != 200:
            print("Error:", r.status_code, r.text[:100])
            break
        repos += r.json()["items"]
        time.sleep(2)
    return repos

def get_readme(full_name):
    r = requests.get(
        f"https://api.github.com/repos/{full_name}/readme",
        headers={**HEADERS, "Accept": "application/vnd.github.raw+json"},
    )
    return r.text[:3000] if r.status_code == 200 else ""

def main():
    seen, projects = set(), []
    for topic in TOPICS:
        print("Topic:", topic)
        for repo in search_repos(topic):
            name = repo["full_name"]
            if name in seen:
                continue
            seen.add(name)
            projects.append({
                "title": repo["name"],
                "full_name": name,
                "description": repo["description"] or "",
                "topics": repo.get("topics", []),
                "language": repo["language"],
                "stars": repo["stargazers_count"],
                "url": repo["html_url"],
                "readme": get_readme(name),
            })
            print(len(projects), name)
    with open("../data/projects.json", "w", encoding="utf-8") as f:
        json.dump(projects, f, ensure_ascii=False, indent=2)
    print("Saved", len(projects), "projects")

if __name__ == "__main__":
    main()