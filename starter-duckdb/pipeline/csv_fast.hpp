#pragma once

#include <cstdint>
#include <cstring>
#include <string_view>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace fast {

// Fast, high-quality 64-bit string hash (wyhash-style).
static inline uint64_t wy_mum(uint64_t a, uint64_t b) {
    __uint128_t r = static_cast<__uint128_t>(a) * b;
    return static_cast<uint64_t>(r) ^ static_cast<uint64_t>(r >> 64);
}

static inline uint64_t hash_bytes(const char* p, size_t len) {
    constexpr uint64_t s0 = 0xa0761d6478bd642fULL;
    constexpr uint64_t s1 = 0xe7037ed1a0b428dbULL;
    constexpr uint64_t s2 = 0x8ebc6af09c88c6e3ULL;
    constexpr uint64_t s3 = 0x589965cc75374cc3ULL;
    uint64_t seed = s0 ^ wy_mum(s0, s1);
    uint64_t a = 0, b = 0;
    const uint8_t* u = reinterpret_cast<const uint8_t*>(p);
    if (__builtin_expect(len <= 16, 1)) {
        if (__builtin_expect(len >= 4, 1)) {
            uint32_t lo, hi, mid1, mid2;
            std::memcpy(&lo, u, 4);
            std::memcpy(&hi, u + len - 4, 4);
            size_t off = (len >> 3) << 2;
            std::memcpy(&mid1, u + off, 4);
            std::memcpy(&mid2, u + len - 4 - off, 4);
            a = (static_cast<uint64_t>(lo) << 32) | mid1;
            b = (static_cast<uint64_t>(hi) << 32) | mid2;
        } else if (len > 0) {
            a = (static_cast<uint64_t>(u[0]) << 16) |
                (static_cast<uint64_t>(u[len >> 1]) << 8) | u[len - 1];
            b = 0;
        }
    } else {
        size_t i = len;
        if (i > 48) {
            uint64_t see1 = seed, see2 = seed;
            do {
                uint64_t v0, v1, v2, v3, v4, v5;
                std::memcpy(&v0, u, 8);
                std::memcpy(&v1, u + 8, 8);
                std::memcpy(&v2, u + 16, 8);
                std::memcpy(&v3, u + 24, 8);
                std::memcpy(&v4, u + 32, 8);
                std::memcpy(&v5, u + 40, 8);
                seed = wy_mum(v0 ^ s1, v1 ^ seed);
                see1 = wy_mum(v2 ^ s2, v3 ^ see1);
                see2 = wy_mum(v4 ^ s3, v5 ^ see2);
                u += 48;
                i -= 48;
            } while (i > 48);
            seed ^= see1 ^ see2;
        }
        while (i > 16) {
            uint64_t v0, v1;
            std::memcpy(&v0, u, 8);
            std::memcpy(&v1, u + 8, 8);
            seed = wy_mum(v0 ^ s1, v1 ^ seed);
            u += 16;
            i -= 16;
        }
        std::memcpy(&a, u + i - 16, 8);
        std::memcpy(&b, u + i - 8, 8);
    }
    return wy_mum(s1 ^ len, wy_mum(a ^ s1, b ^ seed));
}

static inline uint64_t hash_sv(std::string_view sv) {
    return hash_bytes(sv.data(), sv.size());
}

// Byte-wise unsigned string comparison (<0, 0, >0).
static inline int cmp_bytes(const char* a, size_t alen, const char* b, size_t blen) {
    size_t m = alen < blen ? alen : blen;
    int c = m ? std::memcmp(a, b, m) : 0;
    if (c != 0) return c;
    return alen < blen ? -1 : (alen > blen ? 1 : 0);
}

static inline bool less_sv(std::string_view a, std::string_view b) {
    return cmp_bytes(a.data(), a.size(), b.data(), b.size()) < 0;
}

// Exact Howard Hinnant civil -> days since 1970-01-01.
static inline int64_t days_from_civil(int32_t y, uint32_t m, uint32_t d) {
    y -= (m <= 2);
    int32_t era = (y >= 0 ? y : y - 399) / 400;
    uint32_t yoe = static_cast<uint32_t>(y - era * 400);
    uint32_t doy = (153 * (m + (m > 2 ? -3 : 9)) + 2) / 5 + d - 1;
    uint32_t doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    return static_cast<int64_t>(era) * 146097 + static_cast<int64_t>(doe) - 719468;
}

static inline void civil_from_days(int64_t z, int32_t& y, uint32_t& m, uint32_t& d) {
    z += 719468;
    int64_t era = (z >= 0 ? z : z - 146096) / 146097;
    uint32_t doe = static_cast<uint32_t>(z - era * 146097);
    uint32_t yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    y = static_cast<int32_t>(yoe) + static_cast<int32_t>(era) * 400;
    uint32_t doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    uint32_t mp = (5 * doy + 2) / 153;
    d = doy - (153 * mp + 2) / 5 + 1;
    m = mp + (mp < 10 ? 3 : -9);
    y += (m <= 2);
}

static inline int32_t floor_div_day(int64_t ms) {
    constexpr int64_t DAY_MS = 86400000LL;
    if (__builtin_expect(ms >= 0, 1)) {
        return static_cast<int32_t>(ms / DAY_MS);
    }
    return static_cast<int32_t>(-((-ms + DAY_MS - 1) / DAY_MS));
}

static inline bool is_leap_year(int32_t y) {
    return (y % 4 == 0) && (y % 100 != 0 || y % 400 == 0);
}

static constexpr uint8_t DAYS_IN_MONTH[13] = {
    0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31
};

// Direct-mapped 32-entry table of precomputed dates (indexed by day-of-month hash).
// Gives ~99.9% O(1) hit rate on shuffled 14-day windows (and click/ingest horizons)!
struct DateTable {
    struct Entry {
        uint64_t ym_bytes = 0; // "YYYY-MM-" (8 bytes)
        uint16_t dd_bytes = 0; // "DD" (2 bytes)
        uint8_t valid = 0;
        int64_t day_ms = 0;
    };
    Entry entries[32];

    static inline uint32_t slot_for(const char* s) {
        // Maps "01".."31" to unique slots in 0..31:
        // ((s[8] & 3) * 10 + (s[9] & 15)) & 31 -> 1..31 map to distinct slots!
        return ((static_cast<uint32_t>(s[8] & 3) * 10U) + static_cast<uint32_t>(s[9] & 15)) & 31U;
    }

    void precompute(int64_t window_start_ms, int64_t window_end_ms) {
        std::memset(entries, 0, sizeof(entries));
        int64_t d_start = floor_div_day(window_start_ms) - 8;
        int64_t d_end = floor_div_day(window_end_ms) + 6;
        if (d_end - d_start > 31) d_end = d_start + 30;
        for (int64_t d = d_start; d <= d_end; ++d) {
            int32_t y;
            uint32_t m, day;
            civil_from_days(d, y, m, day);
            if (y < 0 || y > 9999) continue;
            char buf[10];
            buf[0] = static_cast<char>('0' + (y / 1000) % 10);
            buf[1] = static_cast<char>('0' + (y / 100) % 10);
            buf[2] = static_cast<char>('0' + (y / 10) % 10);
            buf[3] = static_cast<char>('0' + y % 10);
            buf[4] = '-';
            buf[5] = static_cast<char>('0' + m / 10);
            buf[6] = static_cast<char>('0' + m % 10);
            buf[7] = '-';
            buf[8] = static_cast<char>('0' + day / 10);
            buf[9] = static_cast<char>('0' + day % 10);
            uint32_t idx = slot_for(buf);
            std::memcpy(&entries[idx].ym_bytes, buf, 8);
            std::memcpy(&entries[idx].dd_bytes, buf + 8, 2);
            entries[idx].day_ms = d * 86400000LL;
            entries[idx].valid = 1;
        }
    }
};

// Borrow-free SWAR 8-ASCII-digit parser: mathematically exact for all 2^64 inputs.
static inline bool parse8_digits(const char* p, uint32_t& out) {
    uint64_t v;
    std::memcpy(&v, p, 8);
    uint64_t bad = (v & 0x8080808080808080ULL) |
                   ((v + 0x4646464646464646ULL) & 0x8080808080808080ULL) |
                   (~((v | 0x8080808080808080ULL) - 0x3030303030303030ULL) & 0x8080808080808080ULL);
    if (__builtin_expect(bad != 0, 0)) return false;
    uint64_t lo = v - 0x3030303030303030ULL;
    lo = (lo * 10 + (lo >> 8)) & 0x00FF00FF00FF00FFULL;
    lo = (lo * 100 + (lo >> 16)) & 0x0000FFFF0000FFFFULL;
    out = static_cast<uint32_t>((lo * 10000 + (lo >> 32)) & 0xFFFFFFFFULL);
    return true;
}

// Parse civil prefix "YYYY-MM-DDTHH:MM:SS" (19 bytes) with strict ASCII validation.
static inline bool parse_civil19(const char* s, const DateTable& dtab, int64_t& out_ms) {
    if (__builtin_expect(s[10] != 'T' || s[13] != ':' || s[16] != ':', 0)) {
        return false;
    }

    uint32_t h0 = static_cast<uint8_t>(s[11]) - '0';
    uint32_t h1 = static_cast<uint8_t>(s[12]) - '0';
    uint32_t m0 = static_cast<uint8_t>(s[14]) - '0';
    uint32_t m1 = static_cast<uint8_t>(s[15]) - '0';
    uint32_t s0 = static_cast<uint8_t>(s[17]) - '0';
    uint32_t s1 = static_cast<uint8_t>(s[18]) - '0';
    if (__builtin_expect((h0 >= 10) | (h1 >= 10) | (m0 >= 10) | (m1 >= 10) | (s0 >= 10) | (s1 >= 10), 0)) {
        return false;
    }
    uint32_t hour = h0 * 10 + h1;
    uint32_t minute = m0 * 10 + m1;
    uint32_t second = s0 * 10 + s1;
    if (__builtin_expect((hour > 23) | (minute > 59) | (second > 59), 0)) {
        return false;
    }

    uint64_t ym_bytes;
    uint16_t dd_bytes;
    std::memcpy(&ym_bytes, s, 8);
    std::memcpy(&dd_bytes, s + 8, 2);

    const auto& e = dtab.entries[DateTable::slot_for(s)];
    int64_t day_ms;
    if (__builtin_expect(e.valid && e.ym_bytes == ym_bytes && e.dd_bytes == dd_bytes, 1)) {
        day_ms = e.day_ms;
    } else {
        if (__builtin_expect(s[4] != '-' || s[7] != '-', 0)) return false;
        uint32_t y0 = static_cast<uint8_t>(s[0]) - '0';
        uint32_t y1 = static_cast<uint8_t>(s[1]) - '0';
        uint32_t y2 = static_cast<uint8_t>(s[2]) - '0';
        uint32_t y3 = static_cast<uint8_t>(s[3]) - '0';
        uint32_t mo0 = static_cast<uint8_t>(s[5]) - '0';
        uint32_t mo1 = static_cast<uint8_t>(s[6]) - '0';
        uint32_t d0 = static_cast<uint8_t>(s[8]) - '0';
        uint32_t d1 = static_cast<uint8_t>(s[9]) - '0';
        if (__builtin_expect(
                (y0 >= 10) | (y1 >= 10) | (y2 >= 10) | (y3 >= 10) |
                (mo0 >= 10) | (mo1 >= 10) | (d0 >= 10) | (d1 >= 10), 0)) {
            return false;
        }
        int32_t year = static_cast<int32_t>(y0 * 1000 + y1 * 100 + y2 * 10 + y3);
        uint32_t month = mo0 * 10 + mo1;
        uint32_t day = d0 * 10 + d1;
        if (__builtin_expect(month < 1 || month > 12, 0)) return false;
        uint32_t max_d = DAYS_IN_MONTH[month];
        if (__builtin_expect(month == 2 && is_leap_year(year), 0)) max_d = 29;
        if (__builtin_expect(day < 1 || day > max_d, 0)) return false;
        day_ms = days_from_civil(year, month, day) * 86400000LL;
    }

    out_ms = day_ms + static_cast<int64_t>(hour * 3600 + minute * 60 + second) * 1000LL;
    return true;
}

// Strict F1 parser (used for ingest_ts and manifest).
static inline bool parse_f1_ms(const char* s, size_t len, const DateTable& dtab, int64_t& out_ms) {
    if (__builtin_expect(len != 20 || s[19] != 'Z', 0)) return false;
    return parse_civil19(s, dtab, out_ms);
}

// Strict F1 / F2 / F3 timestamp parser per rules.md.
static inline bool parse_timestamp_ms(const char* s, size_t len, const DateTable& dtab, int64_t& out_ms) {
    if (__builtin_expect(len == 20 && s[19] == 'Z', 1)) {
        return parse_civil19(s, dtab, out_ms);
    }
    // Fast path for common 13-digit F3 epoch ms (year 2001..2286) using SWAR!
    if (__builtin_expect(len == 13, 1)) {
        uint32_t hi8, lo8;
        if (__builtin_expect(!parse8_digits(s, hi8) || !parse8_digits(s + 5, lo8), 0)) return false;
        // s[5..7] overlap; lo8 % 100000 gives last 5 digits cleanly!
        out_ms = static_cast<int64_t>(hi8) * 100000LL + static_cast<int64_t>(lo8 % 100000U);
        return true;
    }
    if (len == 25 && (s[19] == '+' || s[19] == '-') && s[22] == ':') {
        int64_t local_ms;
        if (!parse_civil19(s, dtab, local_ms)) return false;
        uint32_t oh0 = static_cast<uint8_t>(s[20]) - '0';
        uint32_t oh1 = static_cast<uint8_t>(s[21]) - '0';
        uint32_t om0 = static_cast<uint8_t>(s[23]) - '0';
        uint32_t om1 = static_cast<uint8_t>(s[24]) - '0';
        if ((oh0 >= 10) | (oh1 >= 10) | (om0 >= 10) | (om1 >= 10)) return false;
        uint32_t oh = oh0 * 10 + oh1;
        uint32_t om = om0 * 10 + om1;
        if (oh > 23 || om > 59) return false;
        int64_t off_ms = static_cast<int64_t>(oh * 60 + om) * 60000LL;
        out_ms = (s[19] == '+') ? (local_ms - off_ms) : (local_ms + off_ms);
        return true;
    }
    if (__builtin_expect(len == 0, 0)) return false;
    size_t i = 0;
    while (i < len && s[i] == '0') ++i;
    if (i == len) {
        out_ms = 0;
        return true;
    }
    if (len - i > 19) return false;
    uint64_t val = 0;
    for (; i < len; ++i) {
        uint32_t d = static_cast<uint8_t>(s[i]) - '0';
        if (d >= 10) return false;
        if (val > (9223372036854775807ULL - d) / 10ULL) return false;
        val = val * 10 + d;
    }
    out_ms = static_cast<int64_t>(val);
    return true;
}

// Strict ASCII purchase price parser: digits only, 1 <= value <= 10,000,000 (leading zeros allowed).
static inline bool parse_price_cents(const char* s, size_t len, uint32_t& out) {
    if (__builtin_expect(len == 0, 0)) return false;
    size_t i = 0;
    while (i < len && s[i] == '0') ++i;
    if (__builtin_expect(i == len, 0)) return false;
    if (__builtin_expect(len - i > 8, 0)) return false;
    uint32_t val = 0;
    for (; i < len; ++i) {
        uint32_t d = static_cast<uint8_t>(s[i]) - '0';
        if (__builtin_expect(d >= 10, 0)) return false;
        val = val * 10 + d;
    }
    if (__builtin_expect(val == 0 || val > 10000000U, 0)) return false;
    out = val;
    return true;
}

struct Field {
    const char* ptr;
    size_t len;
};

static inline bool unquote_field(const char* s, const char* e, Field& out) {
    size_t len = static_cast<size_t>(e - s);
    if (len >= 2 && s[0] == '"' && e[-1] == '"') {
        s++;
        e--;
        len -= 2;
    }
    if (len > 0 && std::memchr(s, '"', len) != nullptr) return false;
    out.ptr = s;
    out.len = len;
    return true;
}

template <size_t N>
static inline bool parse_csv_line_scalar(const char* s, const char* e, Field (&fields)[N]) {
    const char* cur = s;
    bool has_quote = false;
    for (size_t i = 0; i < N - 1; ++i) {
        const char* comma = static_cast<const char*>(std::memchr(cur, ',', static_cast<size_t>(e - cur)));
        if (__builtin_expect(!comma, 0)) return false;
        fields[i].ptr = cur;
        fields[i].len = static_cast<size_t>(comma - cur);
        if (fields[i].len > 0 && cur[0] == '"') has_quote = true;
        cur = comma + 1;
    }
    if (__builtin_expect(std::memchr(cur, ',', static_cast<size_t>(e - cur)) != nullptr, 0)) {
        return false;
    }
    fields[N - 1].ptr = cur;
    fields[N - 1].len = static_cast<size_t>(e - cur);
    if (fields[N - 1].len > 0 && cur[0] == '"') has_quote = true;

    if (!has_quote) {
        if (__builtin_expect(std::memchr(s, '"', static_cast<size_t>(e - s)) != nullptr, 0)) {
            return false;
        }
        return true;
    }
    for (size_t i = 0; i < N; ++i) {
        const char* fs = fields[i].ptr;
        const char* fe = fs + fields[i].len;
        if (!unquote_field(fs, fe, fields[i])) return false;
    }
    return true;
}

template <size_t N, typename Fn>
static inline void scan_csv_chunk_scalar(const char* data, size_t size, Fn&& on_row) {
    const char* p = data;
    const char* end = data + size;
    Field fields[N];
    while (p < end) {
        const char* nl = static_cast<const char*>(std::memchr(p, '\n', static_cast<size_t>(end - p)));
        const char* line_end = nl ? nl : end;
        const char* next = nl ? nl + 1 : end;
        if (line_end > p && parse_csv_line_scalar<N>(p, line_end, fields)) {
            on_row(fields);
        }
        p = next;
    }
}

template <size_t N, typename Fn>
static inline void emit_if_valid(const char* line_start, const char* line_end,
                                 const char* const* commas, size_t ncommas,
                                 uint32_t line_quotes, Fn&& on_row) {
    if (__builtin_expect(ncommas != N - 1, 0)) return;
    Field fields[N];
    const char* cur = line_start;
    if (line_quotes == 0) {
        for (size_t i = 0; i < N - 1; ++i) {
            fields[i].ptr = cur;
            fields[i].len = static_cast<size_t>(commas[i] - cur);
            cur = commas[i] + 1;
        }
        fields[N - 1].ptr = cur;
        fields[N - 1].len = static_cast<size_t>(line_end - cur);
        on_row(fields);
        return;
    }
    // Every valid quote on the line must be a matched outer pair around a field.
    // Therefore stripped_quotes == line_quotes iff zero internal/unmatched quotes exist!
    uint32_t stripped = 0;
    for (size_t i = 0; i < N - 1; ++i) {
        const char* ce = commas[i];
        size_t len = static_cast<size_t>(ce - cur);
        if (len >= 2 && cur[0] == '"' && ce[-1] == '"') {
            cur++;
            len -= 2;
            stripped += 2;
        }
        fields[i].ptr = cur;
        fields[i].len = len;
        cur = ce + 1;
    }
    {
        size_t len = static_cast<size_t>(line_end - cur);
        if (len >= 2 && cur[0] == '"' && line_end[-1] == '"') {
            cur++;
            len -= 2;
            stripped += 2;
        }
        fields[N - 1].ptr = cur;
        fields[N - 1].len = len;
    }
    if (__builtin_expect(stripped != line_quotes, 0)) return;
    on_row(fields);
}

#if defined(__x86_64__) || defined(_M_X64)

template <size_t N, typename Fn>
__attribute__((target("avx512f,avx512bw,avx512vl,bmi2,popcnt")))
static void scan_csv_chunk_avx512(const char* data, size_t size, Fn&& on_row) {
    const __m512i v_nl = _mm512_set1_epi8('\n');
    const __m512i v_cm = _mm512_set1_epi8(',');
    const __m512i v_qt = _mm512_set1_epi8('"');

    const char* p = data;
    const char* end = data + size;
    const char* line_start = p;
    const char* commas[N];
    size_t ncommas = 0;
    uint32_t line_quotes = 0;

    while (p + 64 <= end) {
        __m512i v = _mm512_loadu_si512(reinterpret_cast<const void*>(p));
        uint64_t nl_mask = _mm512_cmpeq_epi8_mask(v, v_nl);
        uint64_t cm_mask = _mm512_cmpeq_epi8_mask(v, v_cm);
        uint64_t qt_mask = _mm512_cmpeq_epi8_mask(v, v_qt);
        uint64_t special = nl_mask | cm_mask;

        if (!nl_mask) {
            line_quotes += static_cast<uint32_t>(__builtin_popcountll(qt_mask));
            while (cm_mask) {
                unsigned bit = static_cast<unsigned>(__builtin_ctzll(cm_mask));
                if (ncommas < N) commas[ncommas] = p + bit;
                ncommas++;
                cm_mask &= cm_mask - 1;
            }
            p += 64;
            continue;
        }

        uint64_t seg_start_mask = ~0ULL;
        while (special) {
            unsigned bit = static_cast<unsigned>(__builtin_ctzll(special));
            uint64_t bit_mask = 1ULL << bit;
            if (nl_mask & bit_mask) {
                uint64_t before_nl = (bit_mask - 1) & seg_start_mask;
                line_quotes += static_cast<uint32_t>(__builtin_popcountll(qt_mask & before_nl));
                const char* line_end = p + bit;
                if (line_end > line_start) {
                    emit_if_valid<N>(line_start, line_end, commas, ncommas, line_quotes, on_row);
                }
                line_start = line_end + 1;
                ncommas = 0;
                line_quotes = 0;
                seg_start_mask = (bit == 63) ? 0ULL : (~0ULL << (bit + 1));
            } else {
                if (ncommas < N) commas[ncommas] = p + bit;
                ncommas++;
            }
            special &= special - 1;
        }
        line_quotes += static_cast<uint32_t>(__builtin_popcountll(qt_mask & seg_start_mask));
        p += 64;
    }

    while (p < end) {
        char c = *p;
        if (c == '\n') {
            if (p > line_start) {
                emit_if_valid<N>(line_start, p, commas, ncommas, line_quotes, on_row);
            }
            line_start = p + 1;
            ncommas = 0;
            line_quotes = 0;
        } else if (c == ',') {
            if (ncommas < N) commas[ncommas] = p;
            ncommas++;
        } else if (c == '"') {
            line_quotes++;
        }
        ++p;
    }
    if (end > line_start) {
        emit_if_valid<N>(line_start, end, commas, ncommas, line_quotes, on_row);
    }
}

template <size_t N, typename Fn>
__attribute__((target("avx2,bmi2,popcnt")))
static void scan_csv_chunk_avx2(const char* data, size_t size, Fn&& on_row) {
    const __m256i v_nl = _mm256_set1_epi8('\n');
    const __m256i v_cm = _mm256_set1_epi8(',');
    const __m256i v_qt = _mm256_set1_epi8('"');

    const char* p = data;
    const char* end = data + size;
    const char* line_start = p;
    const char* commas[N];
    size_t ncommas = 0;
    uint32_t line_quotes = 0;

    while (p + 32 <= end) {
        __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
        uint32_t nl_mask = static_cast<uint32_t>(_mm256_movemask_epi8(_mm256_cmpeq_epi8(v, v_nl)));
        uint32_t cm_mask = static_cast<uint32_t>(_mm256_movemask_epi8(_mm256_cmpeq_epi8(v, v_cm)));
        uint32_t qt_mask = static_cast<uint32_t>(_mm256_movemask_epi8(_mm256_cmpeq_epi8(v, v_qt)));
        uint32_t special = nl_mask | cm_mask;

        if (!nl_mask) {
            line_quotes += static_cast<uint32_t>(__builtin_popcount(qt_mask));
            while (cm_mask) {
                unsigned bit = static_cast<unsigned>(__builtin_ctz(cm_mask));
                if (ncommas < N) commas[ncommas] = p + bit;
                ncommas++;
                cm_mask &= cm_mask - 1;
            }
            p += 32;
            continue;
        }

        uint32_t seg_start_mask = ~0U;
        while (special) {
            unsigned bit = static_cast<unsigned>(__builtin_ctz(special));
            uint32_t bit_mask = 1U << bit;
            if (nl_mask & bit_mask) {
                uint32_t before_nl = (bit_mask - 1) & seg_start_mask;
                line_quotes += static_cast<uint32_t>(__builtin_popcount(qt_mask & before_nl));
                const char* line_end = p + bit;
                if (line_end > line_start) {
                    emit_if_valid<N>(line_start, line_end, commas, ncommas, line_quotes, on_row);
                }
                line_start = line_end + 1;
                ncommas = 0;
                line_quotes = 0;
                seg_start_mask = (bit == 31) ? 0U : (~0U << (bit + 1));
            } else {
                if (ncommas < N) commas[ncommas] = p + bit;
                ncommas++;
            }
            special &= special - 1;
        }
        line_quotes += static_cast<uint32_t>(__builtin_popcount(qt_mask & seg_start_mask));
        p += 32;
    }

    while (p < end) {
        char c = *p;
        if (c == '\n') {
            if (p > line_start) {
                emit_if_valid<N>(line_start, p, commas, ncommas, line_quotes, on_row);
            }
            line_start = p + 1;
            ncommas = 0;
            line_quotes = 0;
        } else if (c == ',') {
            if (ncommas < N) commas[ncommas] = p;
            ncommas++;
        } else if (c == '"') {
            line_quotes++;
        }
        ++p;
    }
    if (end > line_start) {
        emit_if_valid<N>(line_start, end, commas, ncommas, line_quotes, on_row);
    }
}

#elif defined(__aarch64__)

static inline uint16_t neon_movemask(uint8x16_t mask) {
    static const uint8_t bit_arr[16] = {
        1, 2, 4, 8, 16, 32, 64, 128,
        1, 2, 4, 8, 16, 32, 64, 128
    };
    uint8x16_t bits = vld1q_u8(bit_arr);
    uint8x16_t masked = vandq_u8(mask, bits);
    uint8x8_t lo = vget_low_u8(masked);
    uint8x8_t hi = vget_high_u8(masked);
    uint8x8_t p1 = vpadd_u8(lo, hi);
    uint8x8_t p2 = vpadd_u8(p1, p1);
    uint8x8_t p3 = vpadd_u8(p2, p2);
    return vget_lane_u16(vreinterpret_u16_u8(p3), 0);
}

template <size_t N, typename Fn>
static void scan_csv_chunk_neon(const char* data, size_t size, Fn&& on_row) {
    const uint8x16_t v_nl = vdupq_n_u8('\n');
    const uint8x16_t v_cm = vdupq_n_u8(',');
    const uint8x16_t v_qt = vdupq_n_u8('"');

    const char* p = data;
    const char* end = data + size;
    const char* line_start = p;
    const char* commas[N];
    size_t ncommas = 0;
    uint32_t line_quotes = 0;

    while (p + 16 <= end) {
        uint8x16_t v = vld1q_u8(reinterpret_cast<const uint8_t*>(p));
        uint16_t nl_mask = neon_movemask(vceqq_u8(v, v_nl));
        uint16_t cm_mask = neon_movemask(vceqq_u8(v, v_cm));
        uint16_t qt_mask = neon_movemask(vceqq_u8(v, v_qt));
        uint16_t special = nl_mask | cm_mask;

        if (!nl_mask) {
            line_quotes += static_cast<uint32_t>(__builtin_popcount(qt_mask));
            while (cm_mask) {
                unsigned bit = static_cast<unsigned>(__builtin_ctz(cm_mask));
                if (ncommas < N) commas[ncommas] = p + bit;
                ncommas++;
                cm_mask &= static_cast<uint16_t>(cm_mask - 1);
            }
            p += 16;
            continue;
        }

        uint16_t seg_start_mask = 0xFFFFU;
        while (special) {
            unsigned bit = static_cast<unsigned>(__builtin_ctz(special));
            uint16_t bit_mask = static_cast<uint16_t>(1U << bit);
            if (nl_mask & bit_mask) {
                uint16_t before_nl = static_cast<uint16_t>((bit_mask - 1) & seg_start_mask);
                line_quotes += static_cast<uint32_t>(__builtin_popcount(qt_mask & before_nl));
                const char* line_end = p + bit;
                if (line_end > line_start) {
                    emit_if_valid<N>(line_start, line_end, commas, ncommas, line_quotes, on_row);
                }
                line_start = line_end + 1;
                ncommas = 0;
                line_quotes = 0;
                seg_start_mask = (bit == 15) ? 0U : static_cast<uint16_t>(0xFFFFU << (bit + 1));
            } else {
                if (ncommas < N) commas[ncommas] = p + bit;
                ncommas++;
            }
            special &= static_cast<uint16_t>(special - 1);
        }
        line_quotes += static_cast<uint32_t>(__builtin_popcount(qt_mask & seg_start_mask));
        p += 16;
    }

    while (p < end) {
        char c = *p;
        if (c == '\n') {
            if (p > line_start) {
                emit_if_valid<N>(line_start, p, commas, ncommas, line_quotes, on_row);
            }
            line_start = p + 1;
            ncommas = 0;
            line_quotes = 0;
        } else if (c == ',') {
            if (ncommas < N) commas[ncommas] = p;
            ncommas++;
        } else if (c == '"') {
            line_quotes++;
        }
        ++p;
    }
    if (end > line_start) {
        emit_if_valid<N>(line_start, end, commas, ncommas, line_quotes, on_row);
    }
}

#endif

template <size_t N, typename Fn>
static inline void scan_csv_chunk(const char* data, size_t size, Fn&& on_row) {
#if defined(__x86_64__) || defined(_M_X64)
    static const int simd_level = []() {
        __builtin_cpu_init();
        if (__builtin_cpu_supports("avx512bw") && __builtin_cpu_supports("avx512vl") &&
            __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("bmi2")) {
            return 2;
        }
        if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("bmi2")) {
            return 1;
        }
        return 0;
    }();
    if (simd_level == 2) {
        scan_csv_chunk_avx512<N>(data, size, static_cast<Fn&&>(on_row));
    } else if (simd_level == 1) {
        scan_csv_chunk_avx2<N>(data, size, static_cast<Fn&&>(on_row));
    } else {
        scan_csv_chunk_scalar<N>(data, size, static_cast<Fn&&>(on_row));
    }
#elif defined(__aarch64__)
    scan_csv_chunk_neon<N>(data, size, static_cast<Fn&&>(on_row));
#else
    scan_csv_chunk_scalar<N>(data, size, static_cast<Fn&&>(on_row));
#endif
}

} // namespace fast
