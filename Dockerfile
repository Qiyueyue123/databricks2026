FROM python:3.12-slim-bookworm

WORKDIR /app
COPY starter-duckdb/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY starter-duckdb/pipeline/ ./pipeline/
COPY starter-duckdb/run.sh ./run.sh

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
USER 1000:1000
CMD ["/app/run.sh"]
