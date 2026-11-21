# Shopping Sessions: native C++20 engine.
# The scorers are linux/amd64 (Intel Xeon Cascade Lake+ with AVX-512).
FROM debian:bookworm-slim AS builder
RUN apt-get update && apt-get install -y --no-install-recommends g++ \
    && rm -rf /var/lib/apt/lists/*
COPY pipeline/engine.cpp pipeline/csv_fast.hpp pipeline/parquet_writer.hpp /src/
RUN ARCH=$(uname -m) && \
    if [ "$ARCH" = "x86_64" ]; then \
        g++ -O3 -std=c++20 -march=x86-64-v3 -mtune=cascadelake -flto -fno-exceptions -fno-rtti -pthread /src/engine.cpp -o /shopping-sessions; \
    else \
        g++ -O3 -std=c++20 -flto -fno-exceptions -fno-rtti -pthread /src/engine.cpp -o /shopping-sessions; \
    fi

FROM debian:bookworm-slim
RUN useradd --uid 1000 --no-create-home --home-dir /scratch app
WORKDIR /app
COPY --from=builder /shopping-sessions /app/shopping-sessions
COPY run.sh /app/run.sh
RUN chmod +x /app/shopping-sessions /app/run.sh
USER 1000:1000
CMD ["/app/shopping-sessions"]
