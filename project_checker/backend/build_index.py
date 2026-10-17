import json, numpy as np
from fastembed import TextEmbedding

with open("../data/projects.json", encoding="utf-8") as f:
    projects = json.load(f)
projects = [p for p in projects if p["description"] or p["readme"]]

def text_of(p):
    return (f'{p["title"]}. {p["description"]}. '
            f'Topics: {", ".join(p["topics"])}. {p["readme"][:800]}')

model = TextEmbedding("BAAI/bge-small-en-v1.5")
vectors = np.array(list(model.embed([text_of(p) for p in projects])))

np.save("../data/vectors.npy", vectors)
with open("../data/index_projects.json", "w", encoding="utf-8") as f:
    json.dump(projects, f, ensure_ascii=False)
print("Done", len(projects), "projects indexed")