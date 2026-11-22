// Lightweight synthetic dataset generator for PGO training during Docker build.
// Exercises the common ID lengths, quoted fields, timestamp layouts, purchases,
// repeated event IDs, and linked users.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <sys/stat.h>

static uint64_t rng_state = 0x9e3779b97f4a7c15ULL;
static inline uint64_t next_rng() {
    uint64_t x = rng_state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    rng_state = x;
    return x * 0x2545F4914F6CDD1DULL;
}

static void fmt_b36(char* out, char prefix, uint64_t v) {
    static const char B36[] = "0123456789abcdefghijklmnopqrstuvwxyz";
    // An odd multiplier preserves uniqueness modulo 2^40 and mixes the ID lengths.
    v = (v * 0x9e3779b97f4bULL) & ((1ULL << 40) - 1);
    char digits[8];
    size_t start = sizeof(digits);
    do {
        digits[--start] = B36[v % 36];
        v /= 36;
    } while (v != 0);
    const size_t length = sizeof(digits) - start;
    out[0] = prefix;
    std::memcpy(out + 1, digits + start, length);
    out[length + 1] = '\0';
}

template <size_t N>
static void write_csv_row(FILE* file, const char* const (&fields)[N]) {
    uint64_t quotes = next_rng();
    for (size_t i = 0; i < N; ++i) {
        const bool quoted = quotes % 5 == 0;
        quotes /= 5;
        if (quoted) std::fputc('"', file);
        std::fputs(fields[i], file);
        if (quoted) std::fputc('"', file);
        std::fputc(i + 1 == N ? '\n' : ',', file);
    }
}

static void fmt_iso(char* out, int day_off, int sec_of_day) {
    int d = 7 + day_off;
    int h = sec_of_day / 3600;
    int m = (sec_of_day % 3600) / 60;
    int s = sec_of_day % 60;
    std::snprintf(out, 32, "2026-09-%02dT%02d:%02d:%02dZ", d, h, m, s);
}

int main(int argc, char** argv) {
    std::string dir = (argc > 1) ? argv[1] : "/tmp/pgo_data";
    mkdir(dir.c_str(), 0777);
    mkdir((dir + "/events").c_str(), 0777);
    mkdir((dir + "/clicks").c_str(), 0777);
    mkdir((dir + "/links").c_str(), 0777);

    // manifest.json: 14-day window 2026-09-07 .. 2026-09-21
    {
        FILE* f = std::fopen((dir + "/manifest.json").c_str(), "wb");
        std::fprintf(f, "{\"window_start\":\"2026-09-07T00:00:00Z\",\"window_end\":\"2026-09-21T00:00:00Z\"}\n");
        std::fclose(f);
    }

    // products.csv (50,000 products, 20 categories)
    {
        static const char* CATS[20] = {
            "apparel","auto","baby","beauty","books","electronics","food","garden","health","home",
            "jewelry","music","office","outdoors","pets","shoes","sports","tools","toys","travel"
        };
        FILE* f = std::fopen((dir + "/products.csv").c_str(), "wb");
        std::fprintf(f, "product_id,category\n");
        for (int i = 1; i <= 50000; ++i) {
            std::fprintf(f, "p%d,%s\n", i, CATS[i % 20]);
        }
        std::fclose(f);
    }

    // campaigns.csv (2,000 campaigns, 5 channels)
    {
        static const char* CHS[5] = {"search","social","display","email","affiliate"};
        FILE* f = std::fopen((dir + "/campaigns.csv").c_str(), "wb");
        std::fprintf(f, "campaign_id,channel\n");
        for (int i = 1; i <= 2000; ++i) {
            std::fprintf(f, "c_%d,%s\n", i, CHS[i % 5]);
        }
        std::fclose(f);
    }

    constexpr int NUM_USERS = 16000;
    constexpr int NUM_EVENTS = 400000;
    constexpr int NUM_CLICKS = 100000;
    constexpr int64_t WS_MS = 1788739200000LL; // 2026-09-07T00:00:00Z

    // links/part-0.csv
    {
        FILE* f = std::fopen((dir + "/links/part-0.csv").c_str(), "wb");
        std::fprintf(f, "user_id_a,user_id_b\n");
        char u1[16], u2[16];
        for (int u = 0; u < NUM_USERS; ++u) {
            if ((u % 10) < 6) {
                fmt_b36(u1, 'u', 100000000ULL + static_cast<uint64_t>(u) * 97ULL);
                fmt_b36(u2, 'u', 500000000ULL + static_cast<uint64_t>(u) * 131ULL);
                const char* fields[] = {u1, u2};
                write_csv_row(f, fields);
            }
        }
        std::fclose(f);
    }

    // events/part-0..3.csv
    {
        FILE* fs[4];
        for (int k = 0; k < 4; ++k) {
            fs[k] = std::fopen((dir + "/events/part-" + std::to_string(k) + ".csv").c_str(), "wb");
            std::fprintf(fs[k], "event_id,user_id,ts,ingest_ts,event_type,product_id,price_cents\n");
        }
        char eid[16], uid[16], ts_buf[40], ing_buf[40];
        for (int i = 0; i < NUM_EVENTS; ++i) {
            uint64_t r = next_rng();
            int u = (i % 100 < 5) ? static_cast<int>(r % 5) : static_cast<int>(r % NUM_USERS);
            uint64_t uid_num = ((r >> 16) & 1)
                                   ? (100000000ULL + static_cast<uint64_t>(u) * 97ULL)
                                   : (500000000ULL + static_cast<uint64_t>(u) * 131ULL);
            fmt_b36(uid, 'u', uid_num);
            uint64_t ev_num = (r % 33 == 0 && i > 100) ? static_cast<uint64_t>(i - 50) : static_cast<uint64_t>(i);
            fmt_b36(eid, 'e', 200000000ULL + ev_num * 37ULL);

            int day_off = static_cast<int>((r >> 20) % 14);
            int sod = static_cast<int>((r >> 28) % 86000);
            int ing_sod = sod + 60 + static_cast<int>((r >> 44) % 300);
            int ing_day = day_off;
            if (ing_sod >= 86400) { ing_sod -= 86400; ing_day++; }
            fmt_iso(ing_buf, ing_day, ing_sod);

            int tfmt = static_cast<int>((r >> 4) % 100);
            if (tfmt < 58) {
                fmt_iso(ts_buf, day_off, sod);
            } else if (tfmt < 90) {
                int64_t ms = WS_MS + static_cast<int64_t>(day_off) * 86400000LL + static_cast<int64_t>(sod) * 1000LL + static_cast<int64_t>(r % 1000);
                std::snprintf(ts_buf, sizeof(ts_buf), "%lld", static_cast<long long>(ms));
            } else {
                int d = 7 + day_off;
                int h = sod / 3600, m = (sod % 3600) / 60, s = sod % 60;
                std::snprintf(ts_buf, sizeof(ts_buf), "2026-09-%02dT%02d:%02d:%02d+00:00", d, h, m, s);
            }

            int et = static_cast<int>((r >> 10) % 100);
            const char* etype = (et < 87) ? "view" : ((et < 97) ? "cart" : "purchase");
            int prod = 1 + static_cast<int>((r >> 36) % 50000);
            int price = (et >= 97) ? (100 + static_cast<int>((r >> 50) % 50000)) : 0;

            char product[16], cents[16] = {};
            std::snprintf(product, sizeof(product), "p%d", prod);
            if (price) std::snprintf(cents, sizeof(cents), "%d", price);
            const char* fields[] = {eid, uid, ts_buf, ing_buf, etype, product, cents};
            write_csv_row(fs[i & 3], fields);
        }
        for (int k = 0; k < 4; ++k) std::fclose(fs[k]);
    }

    // clicks/part-0..1.csv
    {
        FILE* fs[2];
        for (int k = 0; k < 2; ++k) {
            fs[k] = std::fopen((dir + "/clicks/part-" + std::to_string(k) + ".csv").c_str(), "wb");
            std::fprintf(fs[k], "click_id,user_id,campaign_id,ts\n");
        }
        char cid[16], uid[16], ts_buf[40];
        for (int i = 0; i < NUM_CLICKS; ++i) {
            uint64_t r = next_rng();
            int u = (i % 100 < 5) ? static_cast<int>(r % 5) : static_cast<int>(r % NUM_USERS);
            uint64_t uid_num = ((r >> 16) & 1)
                                   ? (100000000ULL + static_cast<uint64_t>(u) * 97ULL)
                                   : (500000000ULL + static_cast<uint64_t>(u) * 131ULL);
            fmt_b36(uid, 'u', uid_num);
            fmt_b36(cid, 'k', 300000000ULL + static_cast<uint64_t>(i) * 41ULL);
            int camp = 1 + static_cast<int>((r >> 8) % 2000);
            int day_off = static_cast<int>((r >> 20) % 14);
            int sod = static_cast<int>((r >> 28) % 86000);
            if (((r >> 4) % 100) < 60) {
                fmt_iso(ts_buf, day_off, sod);
            } else {
                int64_t ms = WS_MS + static_cast<int64_t>(day_off) * 86400000LL + static_cast<int64_t>(sod) * 1000LL;
                std::snprintf(ts_buf, sizeof(ts_buf), "%lld", static_cast<long long>(ms));
            }
            char campaign[16];
            std::snprintf(campaign, sizeof(campaign), "c_%d", camp);
            const char* fields[] = {cid, uid, campaign, ts_buf};
            write_csv_row(fs[i & 1], fields);
        }
        for (int k = 0; k < 2; ++k) std::fclose(fs[k]);
    }
    return 0;
}
