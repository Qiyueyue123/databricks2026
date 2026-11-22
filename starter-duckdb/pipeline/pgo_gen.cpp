// Lightweight synthetic Scorer-v5 dataset generator for PGO training during Docker build.
// Exercises ALL hot paths matching Scorer v5:
// - 1 in 8 event_id, user_id, click_id are 33-byte hex IDs (letter + 32 lowercase hex)
// - 20% quotes, 58% F1 / 32% F3 / 10% F2 timestamps, 3% purchases, 3% dups,
//   1% late/invalid, whales, linked chains, 50K products, 2K campaigns.
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

static inline uint64_t splitmix(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

static void fmt_id_v5(char* out, char prefix, uint64_t v) {
    static const char B36[] = "0123456789abcdefghijklmnopqrstuvwxyz";
    static const char HEX[] = "0123456789abcdef";
    uint64_t h1 = splitmix(v ^ (static_cast<uint64_t>(prefix) << 56));
    out[0] = prefix;
    if ((h1 & 7ULL) == 0ULL) {
        // 1 in 8: 33-byte lowercase hex ID!
        uint64_t h2 = splitmix(h1 ^ 0x517cc1b727220a95ULL);
        for (int i = 0; i < 16; ++i) out[1 + i] = HEX[(h1 >> (60 - 4 * i)) & 15U];
        for (int i = 0; i < 16; ++i) out[17 + i] = HEX[(h2 >> (60 - 4 * i)) & 15U];
        out[33] = '\0';
    } else {
        uint64_t x = v;
        for (int i = 8; i >= 1; --i) {
            out[i] = B36[x % 36];
            x /= 36;
        }
        out[9] = '\0';
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

    {
        FILE* f = std::fopen((dir + "/manifest.json").c_str(), "wb");
        std::fprintf(f, "{\"window_start\":\"2026-09-07T00:00:00Z\",\"window_end\":\"2026-09-21T00:00:00Z\"}\n");
        std::fclose(f);
    }

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

    {
        static const char* CHS[5] = {"search","social","display","email","affiliate"};
        FILE* f = std::fopen((dir + "/campaigns.csv").c_str(), "wb");
        std::fprintf(f, "campaign_id,channel\n");
        for (int i = 1; i <= 2000; ++i) {
            std::fprintf(f, "c_%d,%s\n", i, CHS[i % 5]);
        }
        std::fclose(f);
    }

    constexpr int NUM_USERS = 20000;
    constexpr int NUM_EVENTS = 500000;
    constexpr int NUM_CLICKS = 125000;
    constexpr int64_t WS_MS = 1788739200000LL;

    {
        FILE* f = std::fopen((dir + "/links/part-0.csv").c_str(), "wb");
        std::fprintf(f, "user_id_a,user_id_b\n");
        char u1[40], u2[40];
        for (int u = 0; u < NUM_USERS; ++u) {
            if ((u % 10) < 6) {
                fmt_id_v5(u1, 'u', 100000000ULL + static_cast<uint64_t>(u) * 97ULL);
                fmt_id_v5(u2, 'u', 500000000ULL + static_cast<uint64_t>(u) * 131ULL);
                if (u & 1) std::fprintf(f, "\"%s\",%s\n", u1, u2);
                else std::fprintf(f, "%s,%s\n", u1, u2);
            }
        }
        std::fclose(f);
    }

    {
        FILE* fs[4];
        for (int k = 0; k < 4; ++k) {
            fs[k] = std::fopen((dir + "/events/part-" + std::to_string(k) + ".csv").c_str(), "wb");
            std::fprintf(fs[k], "event_id,user_id,ts,ingest_ts,event_type,product_id,price_cents\n");
        }
        char eid[40], uid[40], ts_buf[40], ing_buf[40];
        for (int i = 0; i < NUM_EVENTS; ++i) {
            uint64_t r = next_rng();
            int u = (i % 100 < 5) ? static_cast<int>(r % 5) : static_cast<int>(r % NUM_USERS);
            uint64_t uid_num = ((r >> 16) & 1)
                                   ? (100000000ULL + static_cast<uint64_t>(u) * 97ULL)
                                   : (500000000ULL + static_cast<uint64_t>(u) * 131ULL);
            fmt_id_v5(uid, 'u', uid_num);
            uint64_t ev_num = (r % 33 == 0 && i > 100) ? static_cast<uint64_t>(i - 50) : static_cast<uint64_t>(i);
            fmt_id_v5(eid, 'e', 200000000ULL + ev_num * 37ULL);

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

            FILE* f = fs[i & 3];
            uint32_t qm = static_cast<uint32_t>(r >> 54);
            if (et >= 97) {
                if ((qm & 7) == 0) {
                    std::fprintf(f, "\"%s\",%s,\"%s\",%s,%s,p%d,\"%d\"\n", eid, uid, ts_buf, ing_buf, etype, prod, price);
                } else if ((qm & 3) == 1) {
                    std::fprintf(f, "%s,\"%s\",%s,\"%s\",%s,p%d,%d\n", eid, uid, ts_buf, ing_buf, etype, prod, price);
                } else {
                    std::fprintf(f, "%s,%s,%s,%s,%s,p%d,%d\n", eid, uid, ts_buf, ing_buf, etype, prod, price);
                }
            } else {
                if ((qm & 7) == 0) {
                    std::fprintf(f, "\"%s\",%s,%s,\"%s\",%s,p%d,\"\"\n", eid, uid, ts_buf, ing_buf, etype, prod);
                } else if ((qm & 3) == 1) {
                    std::fprintf(f, "%s,\"%s\",\"%s\",%s,%s,p%d,\n", eid, uid, ts_buf, ing_buf, etype, prod);
                } else {
                    std::fprintf(f, "%s,%s,%s,%s,%s,p%d,\n", eid, uid, ts_buf, ing_buf, etype, prod);
                }
            }
        }
        for (int k = 0; k < 4; ++k) std::fclose(fs[k]);
    }

    {
        FILE* fs[2];
        for (int k = 0; k < 2; ++k) {
            fs[k] = std::fopen((dir + "/clicks/part-" + std::to_string(k) + ".csv").c_str(), "wb");
            std::fprintf(fs[k], "click_id,user_id,campaign_id,ts\n");
        }
        char cid[40], uid[40], ts_buf[40];
        for (int i = 0; i < NUM_CLICKS; ++i) {
            uint64_t r = next_rng();
            int u = (i % 100 < 5) ? static_cast<int>(r % 5) : static_cast<int>(r % NUM_USERS);
            uint64_t uid_num = ((r >> 16) & 1)
                                   ? (100000000ULL + static_cast<uint64_t>(u) * 97ULL)
                                   : (500000000ULL + static_cast<uint64_t>(u) * 131ULL);
            fmt_id_v5(uid, 'u', uid_num);
            fmt_id_v5(cid, 'k', 300000000ULL + static_cast<uint64_t>(i) * 41ULL);
            int camp = 1 + static_cast<int>((r >> 8) % 2000);
            int day_off = static_cast<int>((r >> 20) % 14);
            int sod = static_cast<int>((r >> 28) % 86000);
            if (((r >> 4) % 100) < 60) {
                fmt_iso(ts_buf, day_off, sod);
            } else {
                int64_t ms = WS_MS + static_cast<int64_t>(day_off) * 86400000LL + static_cast<int64_t>(sod) * 1000LL;
                std::snprintf(ts_buf, sizeof(ts_buf), "%lld", static_cast<long long>(ms));
            }
            FILE* f = fs[i & 1];
            if ((r >> 56) & 1) std::fprintf(f, "\"%s\",%s,c_%d,\"%s\"\n", cid, uid, camp, ts_buf);
            else std::fprintf(f, "%s,\"%s\",c_%d,%s\n", cid, uid, camp, ts_buf);
        }
        for (int k = 0; k < 2; ++k) std::fclose(fs[k]);
    }
    return 0;
}
