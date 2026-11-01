FROM python:3.12-slim
WORKDIR /app
COPY backend/server.py backend/server.py
COPY frontend frontend
COPY data/index_projects.json data/index_projects.json
ENV HOST=0.0.0.0
ENV PROJECT_CHECKER_OPEN_BROWSER=0
EXPOSE 8000
CMD ["python", "backend/server.py"]

