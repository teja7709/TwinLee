import json, numpy as np
from fastembed import TextEmbedding

vectors = np.load("../data/vectors.npy")
with open("../data/index_projects.json", encoding="utf-8") as f:
    projects = json.load(f)
model = TextEmbedding("BAAI/bge-small-en-v1.5")

idea = input("Your idea: ")
q = np.array(list(model.embed([idea])))[0]
scores = vectors @ q / (np.linalg.norm(vectors, axis=1) * np.linalg.norm(q))
for i in scores.argsort()[::-1][:5]:
    print(round(float(scores[i]) * 100), projects[i]["full_name"], projects[i]["url"])