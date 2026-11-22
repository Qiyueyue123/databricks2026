FROM debian:bookworm-slim AS builder
RUN apt-get update && apt-get install -y --no-install-recommends g++ \
    && rm -rf /var/lib/apt/lists/*
ARG PORTABLE_X86=0
COPY pipeline/engine.cpp pipeline/csv_fast.hpp pipeline/parquet_writer.hpp pipeline/pgo_gen.cpp /src/
RUN set -e && ARCH=$(uname -m) && \
    mkdir -p /tmp/pgo && \
    g++ -O2 -std=c++20 /src/pgo_gen.cpp -o /tmp/pgo_gen && \
    /tmp/pgo_gen /tmp/pgo_data && \
    if [ "$ARCH" = "x86_64" ]; then \
        if [ "$PORTABLE_X86" != "1" ] && grep -q avx512bw /proc/cpuinfo 2>/dev/null; then \
            ARCH_FLAGS="-march=cascadelake -mprefer-vector-width=256"; \
        else \
            ARCH_FLAGS="-march=x86-64-v3 -mtune=cascadelake"; \
        fi; \
    else \
        ARCH_FLAGS=""; \
    fi && \
    g++ -O3 -std=c++20 $ARCH_FLAGS -fprofile-generate=/tmp/pgo -fno-exceptions -fno-rtti -pthread /src/engine.cpp -o /tmp/engine_pgo && \
    SHOPPING_PGO_TRAIN=1 DATA_DIR=/tmp/pgo_data OUTPUT_DIR=/tmp/pgo_out NCPU=4 /tmp/engine_pgo && \
    g++ -O3 -std=c++20 $ARCH_FLAGS -fprofile-use=/tmp/pgo -Werror=missing-profile -fprofile-correction -fprofile-partial-training -flto -fomit-frame-pointer -fno-exceptions -fno-rtti -pthread /src/engine.cpp -o /tmp/engine_pgo && \
    mv /tmp/engine_pgo /shopping-sessions && \
    rm -rf /tmp/pgo*

FROM debian:bookworm-slim
RUN useradd --uid 1000 --no-create-home --home-dir /scratch app
WORKDIR /app
COPY --from=builder /shopping-sessions /app/shopping-sessions
COPY run.sh /app/run.sh
RUN chmod +x /app/shopping-sessions /app/run.sh
USER 1000:1000
CMD ["/app/shopping-sessions"]
