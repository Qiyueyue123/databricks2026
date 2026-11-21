FROM python:3.12-slim-bookworm AS csv-build
RUN apt-get update && apt-get install -y --no-install-recommends gcc libc6-dev
COPY starter-duckdb/pipeline/input_csv.c /input_csv.c
RUN gcc -O3 -std=c11 -fPIC -shared /input_csv.c -o /_input_csv.so

FROM python:3.12-slim-bookworm

WORKDIR /app
COPY starter-duckdb/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY starter-duckdb/pipeline/ ./pipeline/
COPY --from=csv-build /_input_csv.so ./pipeline/
COPY starter-duckdb/run.sh ./run.sh

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
USER 1000:1000
CMD ["/app/run.sh"]
