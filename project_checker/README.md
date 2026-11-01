# TwinLee Project Checker

TwinLee compares a project idea with the GitHub projects saved in `data/index_projects.json`. The checked-in collection currently contains **940 projects**. Every check ranks the full saved collection and shows the ten closest descriptions.

## Run on your computer

Install Python 3.10 or newer. The default app uses only Python's standard library, so no package installation or API key is needed.

On Windows, double-click `run.bat`. On macOS or Linux, open a terminal in this folder and run:

```sh
./run.sh
```

The app opens in your browser. If it does not, visit the local address printed in the terminal. Close the terminal or press `Ctrl+C` to stop it. If port 8000 is already in use, the app tries the next available ports automatically.

You can also start it from VS Code's terminal with `python backend/server.py` (Windows may use `py -3 backend/server.py`).

## Host it for other people

The local launch scripts are private to the computer running them. To let other people access a shared instance, deploy the included `Dockerfile` to a container host that supports Docker. The server reads the host's `PORT` setting and listens on the network interface required by the host. For a quick local Docker check:

```sh
docker build -t twinlee-project-checker .
docker run --rm -p 8000:8000 twinlee-project-checker
```

Then open <http://127.0.0.1:8000>. Anyone with access to a hosted URL can submit idea text to that server, so deploy it only where you intend to make it available. The app does not store submitted ideas or send them to an external AI service by default.

The default search uses local keyword similarity. It does not send the idea to an external AI service.

## Optional semantic search

The repository includes precomputed vectors for the 940 saved projects. To use them, install the optional packages and enable the local embedding model:

```powershell
python -m pip install -r backend/requirements.txt
$env:PROJECT_CHECKER_SEMANTIC = "1"
python backend/server.py
```

FastEmbed may download the embedding model the first time it starts. If the optional packages or model are unavailable, the app falls back to keyword similarity and still runs.

## Collection scope

The checker searches every row in the local index and reports the number scanned. It does **not** scan every repository on GitHub. The saved data is a topic-based snapshot with repository descriptions, topics, and README excerpts; the check is not a source-code comparison. A high score is a lead to review, not proof of a duplicate or a guarantee that an idea is original.

The optional `backend/collect_data.py` script fetches repositories for the topic list in that file. GitHub's API limits results and rate limits requests, so that snapshot is not an exhaustive crawl of GitHub.

## Data maintenance tools

`backend/build_index.py` rebuilds the vector index from `data/projects.json` and requires the optional packages in `backend/requirements.txt`.

`backend/extract_features.py` enriches saved projects through the Gemini API. It is not used by the web app. Each processed project sends its saved description and README excerpt to Gemini and uses the configured API key; the default limit is 300. Set `MAX_PROJECTS=0` only if you intend to process every project and have reviewed the API cost and data transfer.

`backend/search_test.py` is a command-line check for the semantic vectors.

All data paths are resolved relative to the project folder, so the scripts can be launched from a different current directory.

