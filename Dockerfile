FROM gcc:14-bookworm AS builder
ARG PORTABLE_X86=0
COPY starter-duckdb/pipeline/engine.cpp starter-duckdb/pipeline/csv_fast.hpp starter-duckdb/pipeline/parquet_writer.hpp starter-duckdb/pipeline/pgo_gen.cpp /src/
RUN set -e && ARCH=$(uname -m) && \
    mkdir -p /tmp/pgo && \
    g++ -O2 -std=c++20 /src/pgo_gen.cpp -o /tmp/pgo_gen && \
    /tmp/pgo_gen /tmp/pgo_data && \
    if [ "$ARCH" = "x86_64" ]; then \
        if [ "$PORTABLE_X86" = "1" ]; then \
            TRAIN_FLAGS="-march=x86-64-v3 -mtune=cascadelake"; \
            FINAL_FLAGS="-march=x86-64-v3 -mtune=cascadelake"; \
        elif grep -q avx512bw /proc/cpuinfo 2>/dev/null; then \
            TRAIN_FLAGS="-march=cascadelake -mprefer-vector-width=256"; \
            FINAL_FLAGS="-march=cascadelake -mprefer-vector-width=256"; \
        else \
            TRAIN_FLAGS="-march=x86-64-v3 -mtune=cascadelake"; \
            FINAL_FLAGS="-march=cascadelake -mprefer-vector-width=256"; \
        fi; \
    else \
        TRAIN_FLAGS=""; \
        FINAL_FLAGS=""; \
    fi && \
    g++ -O3 -std=c++20 $TRAIN_FLAGS -DPGO_TRAIN -fprofile-generate=/tmp/pgo -fno-exceptions -fno-rtti -pthread /src/engine.cpp -o /tmp/engine_pgo && \
    DATA_DIR=/tmp/pgo_data OUTPUT_DIR=/tmp/pgo_out NCPU=8 /tmp/engine_pgo && \
    test -n "$(find /tmp/pgo -name '*.gcda')" && \
    g++ -O3 -std=c++20 $FINAL_FLAGS -fprofile-use=/tmp/pgo -fprofile-correction -flto -fno-plt -fomit-frame-pointer -fno-exceptions -fno-rtti -static -pthread /src/engine.cpp -o /shopping-sessions && \
    strip -s /shopping-sessions && \
    rm -rf /tmp/pgo* /tmp/engine_pgo

FROM scratch
COPY --from=builder /shopping-sessions /shopping-sessions
USER 1000:1000
CMD ["/shopping-sessions"]
