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
            FINAL_FLAGS="-march=x86-64-v3 -mtune=cascadelake -Wa,-mbranches-within-32B-boundaries"; \
        elif grep -q avx512bw /proc/cpuinfo 2>/dev/null; then \
            TRAIN_FLAGS="-march=cascadelake -mprefer-vector-width=256"; \
            FINAL_FLAGS="-march=cascadelake -mprefer-vector-width=256 -Wa,-mbranches-within-32B-boundaries"; \
        else \
            TRAIN_FLAGS="-march=x86-64-v3 -mtune=cascadelake"; \
            FINAL_FLAGS="-march=cascadelake -mprefer-vector-width=256 -Wa,-mbranches-within-32B-boundaries"; \
        fi; \
    else \
        TRAIN_FLAGS=""; \
        FINAL_FLAGS=""; \
    fi && \
    g++ -O3 -std=c++20 $TRAIN_FLAGS -fprofile-generate=/tmp/pgo -fno-exceptions -fno-rtti -pthread /src/engine.cpp -o /shopping-sessions && \
    DATA_DIR=/tmp/pgo_data OUTPUT_DIR=/tmp/pgo_out NCPU=8 BUCKETS=1024 PARTS=512 PGO_CLEAN_EXIT=1 /shopping-sessions && \
    test -n "$(find /tmp/pgo -name '*.gcda')" && \
    g++ -O3 -std=c++20 $FINAL_FLAGS -fprofile-use=/tmp/pgo -fprofile-correction -Werror=missing-profile -Werror=coverage-mismatch -flto -fno-plt -fomit-frame-pointer -fno-exceptions -fno-rtti -static -pthread /src/engine.cpp -o /shopping-sessions && \
    strip -s /shopping-sessions && \
    rm -rf /tmp/pgo*

FROM scratch
COPY --from=builder /shopping-sessions /shopping-sessions
USER 1000:1000
CMD ["/shopping-sessions"]
