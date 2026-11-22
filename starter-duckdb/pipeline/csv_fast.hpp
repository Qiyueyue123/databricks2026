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

#define FAST_ALWAYS_INLINE __attribute__((always_inline)) inline

namespace fast {

// Ultra-fast 64-bit integer mixer (1 multiply + 2 xor-shifts).
static FAST_ALWAYS_INLINE uint64_t hash_u64(uint64_t x) {
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33;
    return x;
}

static FAST_ALWAYS_INLINE uint64_t hash_u128(uint64_t hi, uint64_t lo) {
    return hash_u64(hi ^ hash_u64(lo));
}

static FAST_ALWAYS_INLINE uint64_t wy_mum(uint64_t a, uint64_t b) {
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

static inline int cmp_bytes(const char* a, size_t alen, const char* b, size_t blen) {
    size_t m = alen < blen ? alen : blen;
    int c = m ? std::memcmp(a, b, m) : 0;
    if (c != 0) return c;
    return alen < blen ? -1 : (alen > blen ? 1 : 0);
}

static inline bool less_sv(std::string_view a, std::string_view b) {
    return cmp_bytes(a.data(), a.size(), b.data(), b.size()) < 0;
}

// ============================================================================
// Scorer v5 Ultra-Fast Exact 64-Bit ID Codec:
//   - Raw short key (low byte != 0):
//       ID of length 2..9 starting with column prefix ('e'/'u'/'k'), no NUL byte.
//       Stores bytes s[1..len-1] raw little-endian, zero-padded in high bytes.
//       ZERO pext/bswap during parse! Order: bswap64(a) < bswap64(b)!
//   - Long 33-byte hex key (low byte == 0 && bit 63 == 1):
//       ID of length 33: prefix + 32 lowercase hex digits [0-9a-f]{32}.
//       Stores (1ULL << 63) | ((hi64 >> 9) << 8) = top 55 bits in exact lex order!
//   - Exotic interned key (low byte == 0 && bit 63 == 0 && key != 0):
//       256-shard interned handle: ((shard << 24) | idx_p1) << 8.
// ============================================================================

static constexpr uint64_t HEX_KEY_TAG = 1ULL << 63;

static FAST_ALWAYS_INLINE bool is_raw_key(uint64_t k) {
    return (k & 0xFFULL) != 0;
}

static FAST_ALWAYS_INLINE bool is_hex_key(uint64_t k) {
    return (k & (HEX_KEY_TAG | 0xFFULL)) == HEX_KEY_TAG;
}

static constexpr uint64_t LEN_MASK64[9] = {
    0ULL,
    0x00000000000000FFULL,
    0x000000000000FFFFULL,
    0x0000000000FFFFFFULL,
    0x00000000FFFFFFFFULL,
    0x000000FFFFFFFFFFULL,
    0x0000FFFFFFFFFFFFULL,
    0x00FFFFFFFFFFFFFFULL,
    0xFFFFFFFFFFFFFFFFULL,
};

// Try packing a short ID (2..9 bytes starting with `pfx`, no NUL byte).
// Safe even at EOF: `end` is end of readable buffer (when `p + 9 <= end`, single 8B load!).
static FAST_ALWAYS_INLINE uint64_t try_pack_raw(const char* p, size_t len, char pfx, const char* buf_end) {
    if (__builtin_expect(len == 9 && p[0] == pfx, 1)) {
        uint64_t v;
        std::memcpy(&v, p + 1, 8);
        if (__builtin_expect(((v - 0x0101010101010101ULL) & ~v & 0x8080808080808080ULL) == 0, 1)) {
            return v;
        }
        return 0;
    }
    if (__builtin_expect((len - 2) <= 7 && p[0] == pfx, 1)) {
        size_t n = len - 1; // 1..8
        uint64_t v;
        if (__builtin_expect(p + 9 <= buf_end, 1)) {
            std::memcpy(&v, p + 1, 8);
            v &= LEN_MASK64[n];
        } else {
            uint8_t tmp[8] = {};
            std::memcpy(tmp, p + 1, n);
            std::memcpy(&v, tmp, 8);
        }
        uint64_t w = v | ~LEN_MASK64[n];
        if (__builtin_expect(((w - 0x0101010101010101ULL) & ~w & 0x8080808080808080ULL) == 0, 1)) {
            return v;
        }
    }
    return 0;
}

static FAST_ALWAYS_INLINE size_t unpack_raw(uint64_t k, char pfx, char* out) {
    out[0] = pfx;
    std::memcpy(out + 1, &k, 8);
    return 9U - (static_cast<size_t>(__builtin_clzll(k)) >> 3);
}

// SWAR parse of 8 lowercase hex digits [0-9a-f] -> 32-bit big-endian value.
static FAST_ALWAYS_INLINE bool parse_hex8(const char* s, uint32_t& out) {
    uint64_t x;
    std::memcpy(&x, s, 8);
    constexpr uint64_t O = 0x0101010101010101ULL;
    constexpr uint64_t H = 0x8080808080808080ULL;
    if (__builtin_expect((x & H) != 0, 0)) return false;
    uint64_t r09 = (x + O * (0x80U - '0')) & ~(x + O * (0x7FU - '9')) & H;
    uint64_t raf = (x + O * (0x80U - 'a')) & ~(x + O * (0x7FU - 'f')) & H;
    if (__builtin_expect((r09 | raf) != H, 0)) return false;
    uint64_t n = (x & 0x0F0F0F0F0F0F0F0FULL) + 9ULL * ((x >> 6) & O);
    uint64_t p = ((n << 4) | (n >> 8)) & 0x00FF00FF00FF00FFULL;
    p = ((p << 8) | (p >> 16)) & 0x0000FFFF0000FFFFULL;
    p = ((p << 16) | (p >> 32)) & 0xFFFFFFFFULL;
    out = static_cast<uint32_t>(p);
    return true;
}

// Parse 32 lowercase hex digits [0-9a-f]{32} into big-endian (hi64, lo64).
static FAST_ALWAYS_INLINE bool parse_hex128(const char* s, uint64_t& hi, uint64_t& lo) {
    uint32_t a, b, c, d;
    if (__builtin_expect(parse_hex8(s, a) && parse_hex8(s + 8, b) &&
                         parse_hex8(s + 16, c) && parse_hex8(s + 24, d), 1)) {
        hi = (static_cast<uint64_t>(a) << 32) | b;
        lo = (static_cast<uint64_t>(c) << 32) | d;
        return true;
    }
    return false;
}

static FAST_ALWAYS_INLINE uint64_t make_hex_key(uint64_t hi) {
    return HEX_KEY_TAG | ((hi >> 9) << 8);
}

// Format 128-bit hex value (pfx + 32 lowercase hex chars = 33 bytes) via 256-entry 2-byte LUT!
static inline void format_hex128(char pfx, uint64_t hi, uint64_t lo, char* out) {
    static constexpr char HEX2[513] =
        "000102030405060708090a0b0c0d0e0f"
        "101112131415161718191a1b1c1d1e1f"
        "202122232425262728292a2b2c2d2e2f"
        "303132333435363738393a3b3c3d3e3f"
        "404142434445464748494a4b4c4d4e4f"
        "505152535455565758595a5b5c5d5e5f"
        "606162636465666768696a6b6c6d6e6f"
        "707172737475767778797a7b7c7d7e7f"
        "808182838485868788898a8b8c8d8e8f"
        "909192939495969798999a9b9c9d9e9f"
        "a0a1a2a3a4a5a6a7a8a9aaabacadaeaf"
        "b0b1b2b3b4b5b6b7b8b9babbbcbdbebf"
        "c0c1c2c3c4c5c6c7c8c9cacbcccdcecf"
        "d0d1d2d3d4d5d6d7d8d9dadbdcdddedf"
        "e0e1e2e3e4e5e6e7e8e9eaebecedeeef"
        "f0f1f2f3f4f5f6f7f8f9fafbfcfdfeff";
    out[0] = pfx;
    for (int i = 0; i < 8; ++i) {
        unsigned b = static_cast<unsigned>((hi >> (56 - 8 * i)) & 0xFFU);
        std::memcpy(out + 1 + 2 * i, HEX2 + 2 * b, 2);
    }
    for (int i = 0; i < 8; ++i) {
        unsigned b = static_cast<unsigned>((lo >> (56 - 8 * i)) & 0xFFU);
        std::memcpy(out + 17 + 2 * i, HEX2 + 2 * b, 2);
    }
}

// Extract the known 14-byte prefix (pfx + top 13 hex digits) directly from a fast hex key!
static inline void format_hex_prefix14(char pfx, uint64_t k, char* out) {
    static constexpr char HEX[17] = "0123456789abcdef";
    uint64_t top55 = (k & ~HEX_KEY_TAG) >> 8;
    out[0] = pfx;
    for (int i = 0; i < 13; ++i) {
        out[1 + i] = HEX[(top55 >> (51 - 4 * i)) & 15U];
    }
}

// ============================================================================
// Calendar & Timestamp Parsers
// ============================================================================

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

static FAST_ALWAYS_INLINE int32_t floor_div_day(int64_t ms) {
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

struct DateTable {
    struct Entry {
        uint64_t ym_bytes = 0; // "YYYY-MM-" (0 = invalid)
        uint16_t dd_bytes = 0; // "DD"
        int64_t day_sec = 0;
    };
    Entry entries[64];

    static FAST_ALWAYS_INLINE uint32_t slot_for(const char* s) {
        return ((static_cast<uint32_t>(s[8] & 3) << 4) | static_cast<uint32_t>(s[9] & 15));
    }

    void precompute(int64_t window_start_ms, int64_t window_end_ms) {
        std::memset(entries, 0, sizeof(entries));
        int64_t d_start = floor_div_day(window_start_ms) - 8;
        int64_t d_end = floor_div_day(window_end_ms) + 6;
        if (d_end - d_start > 30) d_end = d_start + 30;
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
            entries[idx].day_sec = d * 86400LL;
        }
    }
};

// 3-multiply SWAR 8-ASCII-digit parser!
static FAST_ALWAYS_INLINE bool parse8_digits(uint64_t v, uint32_t& out) {
    uint64_t x = v ^ 0x3030303030303030ULL;
    if (__builtin_expect((((x + 0x7676767676767676ULL) | x) & 0x8080808080808080ULL) != 0, 0)) {
        return false;
    }
    x = (x * 2561ULL) >> 8;
    x = ((x & 0x00FF00FF00FF00FFULL) * 6553601ULL) >> 16;
    out = static_cast<uint32_t>(((x & 0x0000FFFF0000FFFFULL) * 42949672960001ULL) >> 32);
    return true;
}

// 1-add, 1-multiply SWAR "HH:MM:SS" validator + parser!
static FAST_ALWAYS_INLINE bool parse_hms_sec(const char* s, uint32_t& out_sec) {
    uint64_t v;
    std::memcpy(&v, s, 8);
    uint64_t w = v ^ 0x30303A30303A3030ULL;
    // Checks: digits in 0..9, colons == ':', tens of M/S <= 5, tens of H <= 2!
    if (__builtin_expect((((w + 0x060A0F060A0F060DULL) | w) & 0xF0F0F0F0F0F0F0F0ULL) != 0, 0)) {
        return false;
    }
    uint64_t t = w * 2561ULL;
    uint32_t hour = static_cast<uint32_t>((t >> 8) & 0xFFU);
    uint32_t minute = static_cast<uint32_t>((t >> 32) & 0xFFU);
    uint32_t second = static_cast<uint32_t>(t >> 56);
    if (__builtin_expect(hour > 23, 0)) return false;
    out_sec = hour * 3600U + minute * 60U + second;
    return true;
}

static inline bool parse_civil19_day_sec(const char* s, const DateTable& dtab, int64_t& out_day_sec) {
    uint64_t ym_bytes;
    uint16_t dd_bytes;
    std::memcpy(&ym_bytes, s, 8);
    std::memcpy(&dd_bytes, s + 8, 2);

    const auto& e = dtab.entries[DateTable::slot_for(s)];
    if (__builtin_expect(e.ym_bytes == ym_bytes && e.dd_bytes == dd_bytes && ym_bytes != 0, 1)) {
        out_day_sec = e.day_sec;
        return true;
    }
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
    out_day_sec = days_from_civil(year, month, day) * 86400LL;
    return true;
}

static inline bool parse_civil19_sec(const char* s, const DateTable& dtab, int64_t& out_sec) {
    if (__builtin_expect(s[10] != 'T', 0)) return false;
    uint32_t hms_sec;
    int64_t day_sec;
    if (__builtin_expect(!parse_hms_sec(s + 11, hms_sec), 0)) return false;
    if (__builtin_expect(!parse_civil19_day_sec(s, dtab, day_sec), 0)) return false;
    out_sec = day_sec + static_cast<int64_t>(hms_sec);
    return true;
}

// Fast F1 parser for ingest_ts (guaranteed valid F1 by rules.md!) — returns ym/dd in registers for F1 hint reuse!
static FAST_ALWAYS_INLINE bool parse_ingest_f1_sec(const char* s, size_t len, const DateTable& dtab,
                                                   int64_t& out_sec, int64_t& out_day_sec,
                                                   uint64_t& out_ym, uint16_t& out_dd) {
    if (__builtin_expect(len != 20, 0)) return false;
    std::memcpy(&out_ym, s, 8);
    std::memcpy(&out_dd, s + 8, 2);
    const auto& e = dtab.entries[DateTable::slot_for(s)];
    if (__builtin_expect(e.ym_bytes == out_ym && e.dd_bytes == out_dd && out_ym != 0, 1)) {
        out_day_sec = e.day_sec;
    } else if (__builtin_expect(!parse_civil19_day_sec(s, dtab, out_day_sec), 0)) {
        return false;
    }
    uint64_t v;
    std::memcpy(&v, s + 11, 8);
    uint64_t w = v ^ 0x30303A30303A3030ULL;
    uint64_t t = w * 2561ULL;
    uint32_t h = static_cast<uint32_t>((t >> 8) & 0xFFU);
    uint32_t m = static_cast<uint32_t>((t >> 32) & 0xFFU);
    uint32_t sc = static_cast<uint32_t>(t >> 56);
    out_sec = out_day_sec + static_cast<int64_t>(h * 3600U + m * 60U + sc);
    return true;
}

// Strict F1 / F2 / F3 timestamp parser returning BOTH ms and sec (ZERO division, register date-hint reuse!).
static FAST_ALWAYS_INLINE bool parse_timestamp_ms_sec(const char* s, size_t len, const DateTable& dtab,
                                                      int64_t& out_ms, int64_t& out_sec,
                                                      uint64_t hint_ym = 0, uint16_t hint_dd = 0, int64_t hint_day_sec = 0) {
    if (len == 20 && s[19] == 'Z') {
        if (__builtin_expect(s[10] != 'T', 0)) return false;
        uint32_t hms_sec;
        if (__builtin_expect(!parse_hms_sec(s + 11, hms_sec), 0)) return false;
        int64_t day_sec;
        if (hint_ym != 0) {
            uint64_t a0;
            uint16_t a1;
            std::memcpy(&a0, s, 8);
            std::memcpy(&a1, s + 8, 2);
            if (__builtin_expect(a0 == hint_ym && a1 == hint_dd, 1)) {
                out_sec = hint_day_sec + static_cast<int64_t>(hms_sec);
                out_ms = out_sec * 1000LL;
                return true;
            }
        }
        if (__builtin_expect(!parse_civil19_day_sec(s, dtab, day_sec), 0)) return false;
        out_sec = day_sec + static_cast<int64_t>(hms_sec);
        out_ms = out_sec * 1000LL;
        return true;
    }
    if (len == 13) {
        uint64_t v0, v1;
        std::memcpy(&v0, s, 8);
        std::memcpy(&v1, s + 5, 8);
        uint32_t hi8;
        if (__builtin_expect(!parse8_digits(v0, hi8), 0)) return false;
        uint64_t x1 = (v1 & 0xFFFFFFFFFF000000ULL) ^ 0x3030303030000000ULL;
        if (__builtin_expect((((x1 + 0x7676767676000000ULL) | x1) & 0x8080808080000000ULL) != 0, 0)) {
            return false;
        }
        uint32_t d89 = static_cast<uint32_t>(((x1 >> 24) & 0xFFU) * 10U + ((x1 >> 32) & 0xFFU));
        uint32_t d1012 = static_cast<uint32_t>(((x1 >> 40) & 0xFFU) * 100U + ((x1 >> 48) & 0xFFU) * 10U + ((x1 >> 56) & 0xFFU));
        int64_t sec = static_cast<int64_t>(hi8) * 100LL + static_cast<int64_t>(d89);
        out_sec = sec;
        out_ms = sec * 1000LL + static_cast<int64_t>(d1012);
        return true;
    }
    if (len == 25 && (s[19] == '+' || s[19] == '-') && s[22] == ':') {
        if (__builtin_expect(s[10] != 'T', 0)) return false;
        uint32_t hms_sec;
        if (__builtin_expect(!parse_hms_sec(s + 11, hms_sec), 0)) return false;
        uint32_t oh0 = static_cast<uint8_t>(s[20]) - '0';
        uint32_t oh1 = static_cast<uint8_t>(s[21]) - '0';
        uint32_t om0 = static_cast<uint8_t>(s[23]) - '0';
        uint32_t om1 = static_cast<uint8_t>(s[24]) - '0';
        if (__builtin_expect((oh0 >= 10) | (oh1 >= 10) | (om0 >= 10) | (om1 >= 10), 0)) return false;
        uint32_t oh = oh0 * 10 + oh1;
        uint32_t om = om0 * 10 + om1;
        if (__builtin_expect(oh > 23 || om > 59, 0)) return false;
        int64_t day_sec;
        bool got_day = false;
        if (hint_ym != 0) {
            uint64_t a0;
            uint16_t a1;
            std::memcpy(&a0, s, 8);
            std::memcpy(&a1, s + 8, 2);
            if (__builtin_expect(a0 == hint_ym && a1 == hint_dd, 1)) {
                day_sec = hint_day_sec;
                got_day = true;
            }
        }
        if (!got_day && !parse_civil19_day_sec(s, dtab, day_sec)) return false;
        int64_t local_sec = day_sec + static_cast<int64_t>(hms_sec);
        int64_t off_sec = static_cast<int64_t>(oh * 60 + om) * 60LL;
        out_sec = (s[19] == '+') ? (local_sec - off_sec) : (local_sec + off_sec);
        out_ms = out_sec * 1000LL;
        return true;
    }
    if (__builtin_expect(len == 0, 0)) return false;
    size_t i = 0;
    while (i < len && s[i] == '0') ++i;
    if (i == len) {
        out_ms = 0;
        out_sec = 0;
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
    out_sec = out_ms / 1000LL;
    return true;
}

static FAST_ALWAYS_INLINE bool parse_timestamp_ms(const char* s, size_t len, const DateTable& dtab,
                                                  int64_t& out_ms) {
    int64_t dummy_sec = 0;
    return parse_timestamp_ms_sec(s, len, dtab, out_ms, dummy_sec);
}

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

static FAST_ALWAYS_INLINE bool parse_pos_u32_no_lz(const char* s, size_t len, uint32_t max_val, uint32_t& out) {
    if (len == 0 || len > 7 || s[0] == '0') return false;
    uint32_t v = 0;
    for (size_t i = 0; i < len; ++i) {
        uint32_t d = static_cast<uint8_t>(s[i]) - '0';
        if (d >= 10) return false;
        v = v * 10 + d;
    }
    if (v == 0 || v > max_val) return false;
    out = v;
    return true;
}

// ============================================================================
// Chunked 64-Byte Block-Classified SIMD CSV Scanner (Scorer v5: up to 192B lines!)
// - Classifies each 64-byte block ONCE (ZERO overlapping SIMD work!)
// - 1-carry bitmask quote validation (ZERO memory reads for quote stripping!)
// - Covers 100% of Scorer v5 lines (including TWO 33-byte hex IDs = ~145B!)
// ============================================================================

struct Field {
    const char* ptr;
    size_t len;
};

template <size_t N>
static inline bool split_line_exact(const char* p, const char* end, Field (&out)[N]) {
    if (__builtin_expect(p >= end, 0)) return false;
    size_t nf = 0;
    const char* cur = p;
    while (true) {
        const char* cm = static_cast<const char*>(std::memchr(cur, ',', static_cast<size_t>(end - cur)));
        const char* stop = cm ? cm : end;
        if (nf == N) return false;
        const char* s = cur;
        const char* e = stop;
        if (std::memchr(cur, '"', static_cast<size_t>(stop - cur)) != nullptr) {
            if (stop - cur < 2 || cur[0] != '"' || stop[-1] != '"' ||
                std::memchr(cur + 1, '"', static_cast<size_t>(stop - cur - 2)) != nullptr) {
                return false;
            }
            s = cur + 1;
            e = stop - 1;
        }
        out[nf++] = Field{s, static_cast<size_t>(e - s)};
        if (!cm) break;
        cur = cm + 1;
    }
    return nf == N;
}

template <size_t N, typename Fn>
static inline void scan_csv_chunk_scalar(const char* data, size_t size, Fn&& on_row) {
    const char* p = data;
    const char* end = data + size;
    Field fields[N];
    while (p < end) {
        const char* nl = static_cast<const char*>(std::memchr(p, '\n', static_cast<size_t>(end - p)));
        const char* le = nl ? nl : end;
        if (split_line_exact<N>(p, le, fields)) {
            on_row(fields);
        }
        p = nl ? (nl + 1) : end;
    }
}

// Validate & extract N fields from a line of length L < 64 using 64-bit masks (1-carry quote check!).
template <size_t N>
static FAST_ALWAYS_INLINE bool extract_line64(const char* line, uint32_t L, uint64_t cm, uint64_t qm, Field (&f)[N]) {
    const uint64_t keep = (1ULL << L) - 1ULL;
    cm &= keep;
    if (__builtin_expect(static_cast<size_t>(__builtin_popcountll(cm)) != N - 1, 0)) return false;
    qm &= keep;
    const uint64_t sep = cm | (1ULL << L);
    const uint64_t body = ~sep & keep;
    const uint64_t starts = ((sep << 1) | 1ULL) & body;
    const uint64_t lasts = (sep >> 1) & body;
    const uint64_t qs = qm & starts;
    const uint64_t qe = qm & lasts;
    const uint64_t open = (body + qs) & sep;
    if (__builtin_expect((qm != (qs | qe)) | ((qs & qe) != 0) | (open != (qe << 1)), 0)) return false;
    uint32_t s = 0;
    if constexpr (N == 2) {
        uint32_t e0 = static_cast<uint32_t>(__builtin_ctzll(cm));
        uint32_t w0 = static_cast<uint32_t>(line[0] == '"');
        f[0] = Field{line + w0, e0 - (w0 << 1)};
        s = e0 + 1;
        uint32_t w1 = static_cast<uint32_t>(line[s] == '"');
        f[1] = Field{line + s + w1, (L - s) - (w1 << 1)};
    } else if constexpr (N == 4) {
        uint32_t e0 = static_cast<uint32_t>(__builtin_ctzll(cm)); cm &= cm - 1;
        uint32_t e1 = static_cast<uint32_t>(__builtin_ctzll(cm)); cm &= cm - 1;
        uint32_t e2 = static_cast<uint32_t>(__builtin_ctzll(cm));
        uint32_t w0 = static_cast<uint32_t>(line[0] == '"');
        uint32_t w1 = static_cast<uint32_t>(line[e0 + 1] == '"');
        uint32_t w2 = static_cast<uint32_t>(line[e1 + 1] == '"');
        uint32_t w3 = static_cast<uint32_t>(line[e2 + 1] == '"');
        f[0] = Field{line + w0, e0 - (w0 << 1)};
        f[1] = Field{line + e0 + 1 + w1, (e1 - e0 - 1) - (w1 << 1)};
        f[2] = Field{line + e1 + 1 + w2, (e2 - e1 - 1) - (w2 << 1)};
        f[3] = Field{line + e2 + 1 + w3, (L - e2 - 1) - (w3 << 1)};
    } else {
        for (size_t k = 0; k < N; ++k) {
            uint32_t e;
            if (k + 1 < N) {
                e = static_cast<uint32_t>(__builtin_ctzll(cm));
                cm &= cm - 1;
            } else {
                e = L;
            }
            uint32_t w = static_cast<uint32_t>(line[s] == '"');
            f[k] = Field{line + s + w, (e - s) - (w << 1)};
            s = e + 1;
        }
    }
    return true;
}

// Validate & extract N fields from a line of length 1 <= L < 128 using 128-bit masks (1-carry quote check!).
template <size_t N>
static FAST_ALWAYS_INLINE bool extract_line128(const char* line, uint32_t L,
                                               uint64_t c0, uint64_t c1, uint64_t q0, uint64_t q1,
                                               Field (&f)[N]) {
    const __uint128_t keep = (static_cast<__uint128_t>(1) << L) - 1;
    const __uint128_t cm = ((static_cast<__uint128_t>(c1) << 64) | c0) & keep;
    uint64_t lo = static_cast<uint64_t>(cm);
    uint64_t hi = static_cast<uint64_t>(cm >> 64);
    const uint32_t nlo = static_cast<uint32_t>(__builtin_popcountll(lo));
    const uint32_t nhi = static_cast<uint32_t>(__builtin_popcountll(hi));
    if (__builtin_expect(nlo + nhi != N - 1, 0)) return false;

    const __uint128_t qm = ((static_cast<__uint128_t>(q1) << 64) | q0) & keep;
    const __uint128_t sep = cm | (static_cast<__uint128_t>(1) << L);
    const __uint128_t body = ~sep & keep;
    const __uint128_t starts = ((sep << 1) | 1) & body;
    const __uint128_t lasts = (sep >> 1) & body;
    const __uint128_t qs = qm & starts;
    const __uint128_t qe = qm & lasts;
    const __uint128_t open = (body + qs) & sep;
    if (__builtin_expect((qm != (qs | qe)) | ((qs & qe) != 0) | (open != (qe << 1)), 0)) return false;

    uint32_t cs[N];
    if constexpr (N == 7) {
        if (__builtin_expect(nlo >= 3 && nhi >= 1, 1)) {
            cs[0] = static_cast<uint32_t>(__builtin_ctzll(lo)); lo &= lo - 1;
            cs[1] = static_cast<uint32_t>(__builtin_ctzll(lo)); lo &= lo - 1;
            cs[2] = static_cast<uint32_t>(__builtin_ctzll(lo)); lo &= lo - 1;
            unsigned b5 = 63U - static_cast<unsigned>(__builtin_clzll(hi));
            cs[5] = 64U + b5;
            hi ^= (1ULL << b5);
            for (int k = 3; k <= 4; ++k) {
                bool u = (lo != 0);
                uint64_t m = u ? lo : hi;
                cs[k] = static_cast<uint32_t>(__builtin_ctzll(m)) + (u ? 0U : 64U);
                lo = u ? (lo & (lo - 1)) : lo;
                hi = u ? hi : (hi & (hi - 1));
            }
        } else {
            for (size_t k = 0; k < 6; ++k) {
                bool u = (lo != 0);
                uint64_t m = u ? lo : hi;
                cs[k] = static_cast<uint32_t>(__builtin_ctzll(m)) + (u ? 0U : 64U);
                lo = u ? (lo & (lo - 1)) : lo;
                hi = u ? hi : (hi & (hi - 1));
            }
        }
    } else {
        for (size_t k = 0; k < N - 1; ++k) {
            bool u = (lo != 0);
            uint64_t m = u ? lo : hi;
            cs[k] = static_cast<uint32_t>(__builtin_ctzll(m)) + (u ? 0U : 64U);
            lo = u ? (lo & (lo - 1)) : lo;
            hi = u ? hi : (hi & (hi - 1));
        }
    }
    cs[N - 1] = L;

    uint32_t s = 0;
    for (size_t k = 0; k < N; ++k) {
        uint32_t e = cs[k];
        uint32_t w = static_cast<uint32_t>(line[s] == '"');
        f[k] = Field{line + s + w, (e - s) - (w << 1)};
        s = e + 1;
    }
    return true;
}

// Validate & extract 7 fields from a long Scorer v5 line (128 <= L < 192, e.g. two 33B hex IDs!) from 3 words.
static inline bool extract_line192_7(const char* line, uint32_t L,
                                     uint64_t c0, uint64_t c1, uint64_t c2,
                                     uint64_t q0, uint64_t q1, uint64_t q2,
                                     Field (&f)[7]) {
    uint64_t m2 = (1ULL << (L - 128)) - 1ULL;
    c2 &= m2;
    q2 &= m2;
    if (__builtin_expect(__builtin_popcountll(c0) + __builtin_popcountll(c1) + __builtin_popcountll(c2) != 6, 0)) {
        return false;
    }
    uint32_t cs[7];
    uint64_t ws[3] = {c0, c1, c2};
    size_t k = 0;
    for (int wi = 0; wi < 3; ++wi) {
        uint64_t m = ws[wi];
        while (m) {
            cs[k++] = static_cast<uint32_t>(wi * 64) + static_cast<uint32_t>(__builtin_ctzll(m));
            m &= m - 1;
        }
    }
    cs[6] = L;
    uint32_t total_q = static_cast<uint32_t>(__builtin_popcountll(q0) + __builtin_popcountll(q1) + __builtin_popcountll(q2));
    uint32_t stripped = 0;
    uint32_t s = 0;
    for (size_t i = 0; i < 7; ++i) {
        uint32_t e = cs[i];
        uint32_t len = e - s;
        uint32_t q = (len >= 2 && line[s] == '"' && line[e - 1] == '"') ? 1U : 0U;
        f[i] = Field{line + s + q, len - (q << 1)};
        stripped += (q << 1);
        s = e + 1;
    }
    return stripped == total_q;
}

// Chunked 4 KB block-classified scanner macro (inlines 100% into caller!).
#define FAST_SCAN_BLOCKS_LOOP(LOAD64_MACRO) \
    const char* cur = data; \
    const char* const end = data + size; \
    uint64_t nls[64], cms[67], qms[67]; \
    Field fields[N]; \
    while (cur + 64 <= end) { \
        const size_t avail = static_cast<size_t>(end - cur); \
        const size_t nblks = (avail >> 6) < 64 ? (avail >> 6) : 64; \
        for (size_t bi = 0; bi < nblks; ++bi) { \
            if ((bi & 3U) == 0) __builtin_prefetch(cur + (bi << 6) + 1536, 0, 1); \
            LOAD64_MACRO(cur + (bi << 6), nls[bi], cms[bi], qms[bi]); \
        } \
        cms[nblks] = 0; cms[nblks + 1] = 0; cms[nblks + 2] = 0; \
        qms[nblks] = 0; qms[nblks + 1] = 0; qms[nblks + 2] = 0; \
        size_t ls = 0; \
        for (size_t bi = 0; bi < nblks; ++bi) { \
            uint64_t nm = nls[bi]; \
            while (nm) { \
                const size_t nl_rel = (bi << 6) + static_cast<size_t>(__builtin_ctzll(nm)); \
                nm &= nm - 1; \
                const uint32_t L = static_cast<uint32_t>(nl_rel - ls); \
                const size_t sbi = ls >> 6; \
                const unsigned sh = static_cast<unsigned>(ls & 63U); \
                if constexpr (N == 7) { \
                    if (__builtin_expect(L > 0 && L < 128U, 1)) { \
                        uint64_t c0, c1, q0, q1; \
                        if (__builtin_expect(sh != 0, 1)) { \
                            const unsigned rsh = 64U - sh; \
                            const uint64_t nc = cms[sbi + 1]; \
                            const uint64_t nq = qms[sbi + 1]; \
                            c0 = (cms[sbi] >> sh) | (nc << rsh); \
                            c1 = (nc >> sh) | (cms[sbi + 2] << rsh); \
                            q0 = (qms[sbi] >> sh) | (nq << rsh); \
                            q1 = (nq >> sh) | (qms[sbi + 2] << rsh); \
                        } else { \
                            c0 = cms[sbi]; c1 = cms[sbi + 1]; \
                            q0 = qms[sbi]; q1 = qms[sbi + 1]; \
                        } \
                        if (__builtin_expect(extract_line128<7>(cur + ls, L, c0, c1, q0, q1, fields), 1)) { \
                            on_row(fields); \
                        } \
                    } else if (__builtin_expect(L >= 128U && L < 192U, 1)) { \
                        uint64_t c0, c1, c2, q0, q1, q2; \
                        if (sh != 0) { \
                            const unsigned rsh = 64U - sh; \
                            c0 = (cms[sbi] >> sh) | (cms[sbi + 1] << rsh); \
                            c1 = (cms[sbi + 1] >> sh) | (cms[sbi + 2] << rsh); \
                            c2 = (cms[sbi + 2] >> sh) | (cms[sbi + 3] << rsh); \
                            q0 = (qms[sbi] >> sh) | (qms[sbi + 1] << rsh); \
                            q1 = (qms[sbi + 1] >> sh) | (qms[sbi + 2] << rsh); \
                            q2 = (qms[sbi + 2] >> sh) | (qms[sbi + 3] << rsh); \
                        } else { \
                            c0 = cms[sbi]; c1 = cms[sbi + 1]; c2 = cms[sbi + 2]; \
                            q0 = qms[sbi]; q1 = qms[sbi + 1]; q2 = qms[sbi + 2]; \
                        } \
                        if (__builtin_expect(extract_line192_7(cur + ls, L, c0, c1, c2, q0, q1, q2, fields), 1)) { \
                            on_row(fields); \
                        } \
                    } else if (L >= 192U) { \
                        if (split_line_exact<7>(cur + ls, cur + nl_rel, fields)) on_row(fields); \
                    } \
                } else { \
                    if (__builtin_expect(L > 0 && L < 64U, 1)) { \
                        uint64_t c0, q0; \
                        if (__builtin_expect(sh != 0, 1)) { \
                            const unsigned rsh = 64U - sh; \
                            c0 = (cms[sbi] >> sh) | (cms[sbi + 1] << rsh); \
                            q0 = (qms[sbi] >> sh) | (qms[sbi + 1] << rsh); \
                        } else { \
                            c0 = cms[sbi]; q0 = qms[sbi]; \
                        } \
                        if (__builtin_expect(extract_line64<N>(cur + ls, L, c0, q0, fields), 1)) { \
                            on_row(fields); \
                        } \
                    } else if (__builtin_expect(L >= 64U && L < 128U, 1)) { \
                        uint64_t c0, c1, q0, q1; \
                        if (sh != 0) { \
                            const unsigned rsh = 64U - sh; \
                            const uint64_t nc = cms[sbi + 1]; \
                            const uint64_t nq = qms[sbi + 1]; \
                            c0 = (cms[sbi] >> sh) | (nc << rsh); \
                            c1 = (nc >> sh) | (cms[sbi + 2] << rsh); \
                            q0 = (qms[sbi] >> sh) | (nq << rsh); \
                            q1 = (nq >> sh) | (qms[sbi + 2] << rsh); \
                        } else { \
                            c0 = cms[sbi]; c1 = cms[sbi + 1]; \
                            q0 = qms[sbi]; q1 = qms[sbi + 1]; \
                        } \
                        if (__builtin_expect(extract_line128<N>(cur + ls, L, c0, c1, q0, q1, fields), 1)) { \
                            on_row(fields); \
                        } \
                    } else if (L >= 128U) { \
                        if (split_line_exact<N>(cur + ls, cur + nl_rel, fields)) on_row(fields); \
                    } \
                } \
                ls = nl_rel + 1; \
            } \
        } \
        if (__builtin_expect(ls == 0, 0)) { \
            const char* nl = static_cast<const char*>(std::memchr(cur + (nblks << 6), '\n', static_cast<size_t>(end - (cur + (nblks << 6))))); \
            const char* le = nl ? nl : end; \
            if (split_line_exact<N>(cur, le, fields)) on_row(fields); \
            cur = nl ? (nl + 1) : end; \
        } else { \
            cur += ls; \
        } \
    } \
    if (cur < end) { \
        scan_csv_chunk_scalar<N>(cur, static_cast<size_t>(end - cur), on_row); \
    }

#if defined(__x86_64__) || defined(_M_X64)

#define FAST_LOAD64_AVX512(PTR, NL, CM, QT) do { \
    __m512i _v = _mm512_loadu_si512(reinterpret_cast<const void*>(PTR)); \
    NL = _mm512_cmpeq_epi8_mask(_v, v_nl); \
    CM = _mm512_cmpeq_epi8_mask(_v, v_cm); \
    QT = _mm512_cmpeq_epi8_mask(_v, v_qt); \
} while (0)

#define FAST_LOAD64_AVX2(PTR, NL, CM, QT) do { \
    __m256i _v0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(PTR)); \
    __m256i _v1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>((PTR) + 32)); \
    NL = static_cast<uint32_t>(_mm256_movemask_epi8(_mm256_cmpeq_epi8(_v0, v_nl))) | \
         (static_cast<uint64_t>(static_cast<uint32_t>(_mm256_movemask_epi8(_mm256_cmpeq_epi8(_v1, v_nl)))) << 32); \
    CM = static_cast<uint32_t>(_mm256_movemask_epi8(_mm256_cmpeq_epi8(_v0, v_cm))) | \
         (static_cast<uint64_t>(static_cast<uint32_t>(_mm256_movemask_epi8(_mm256_cmpeq_epi8(_v1, v_cm)))) << 32); \
    QT = static_cast<uint32_t>(_mm256_movemask_epi8(_mm256_cmpeq_epi8(_v0, v_qt))) | \
         (static_cast<uint64_t>(static_cast<uint32_t>(_mm256_movemask_epi8(_mm256_cmpeq_epi8(_v1, v_qt)))) << 32); \
} while (0)

#if defined(__AVX512BW__) && defined(__AVX512VL__) && defined(__AVX512F__)
template <size_t N, typename Fn>
static FAST_ALWAYS_INLINE void scan_csv_chunk_avx512(const char* data, size_t size, Fn&& on_row) {
    const __m512i v_nl = _mm512_set1_epi8('\n');
    const __m512i v_cm = _mm512_set1_epi8(',');
    const __m512i v_qt = _mm512_set1_epi8('"');
    FAST_SCAN_BLOCKS_LOOP(FAST_LOAD64_AVX512);
}
#else
template <size_t N, typename Fn>
static FAST_ALWAYS_INLINE void scan_csv_chunk_avx2(const char* data, size_t size, Fn&& on_row) {
    const __m256i v_nl = _mm256_set1_epi8('\n');
    const __m256i v_cm = _mm256_set1_epi8(',');
    const __m256i v_qt = _mm256_set1_epi8('"');
    FAST_SCAN_BLOCKS_LOOP(FAST_LOAD64_AVX2);
}
#endif

#elif defined(__aarch64__)

static FAST_ALWAYS_INLINE uint64_t neon_movemask64(uint8x16_t m0, uint8x16_t m1, uint8x16_t m2, uint8x16_t m3) {
    static const uint8_t bit_arr[16] = {
        1, 2, 4, 8, 16, 32, 64, 128,
        1, 2, 4, 8, 16, 32, 64, 128
    };
    uint8x16_t bits = vld1q_u8(bit_arr);
    uint8x16_t t0 = vandq_u8(m0, bits);
    uint8x16_t t1 = vandq_u8(m1, bits);
    uint8x16_t t2 = vandq_u8(m2, bits);
    uint8x16_t t3 = vandq_u8(m3, bits);
    uint8x16_t sum0 = vpaddq_u8(t0, t1);
    uint8x16_t sum1 = vpaddq_u8(t2, t3);
    sum0 = vpaddq_u8(sum0, sum1);
    sum0 = vpaddq_u8(sum0, sum0);
    return vgetq_lane_u64(vreinterpretq_u64_u8(sum0), 0);
}

#define FAST_LOAD64_NEON(PTR, NL, CM, QT) do { \
    const uint8_t* _up = reinterpret_cast<const uint8_t*>(PTR); \
    uint8x16_t _a0 = vld1q_u8(_up); \
    uint8x16_t _a1 = vld1q_u8(_up + 16); \
    uint8x16_t _a2 = vld1q_u8(_up + 32); \
    uint8x16_t _a3 = vld1q_u8(_up + 48); \
    NL = neon_movemask64(vceqq_u8(_a0, v_nl), vceqq_u8(_a1, v_nl), vceqq_u8(_a2, v_nl), vceqq_u8(_a3, v_nl)); \
    CM = neon_movemask64(vceqq_u8(_a0, v_cm), vceqq_u8(_a1, v_cm), vceqq_u8(_a2, v_cm), vceqq_u8(_a3, v_cm)); \
    QT = neon_movemask64(vceqq_u8(_a0, v_qt), vceqq_u8(_a1, v_qt), vceqq_u8(_a2, v_qt), vceqq_u8(_a3, v_qt)); \
} while (0)

template <size_t N, typename Fn>
static FAST_ALWAYS_INLINE void scan_csv_chunk_neon(const char* data, size_t size, Fn&& on_row) {
    const uint8x16_t v_nl = vdupq_n_u8('\n');
    const uint8x16_t v_cm = vdupq_n_u8(',');
    const uint8x16_t v_qt = vdupq_n_u8('"');
    FAST_SCAN_BLOCKS_LOOP(FAST_LOAD64_NEON);
}

#endif

template <size_t N, typename Fn>
static FAST_ALWAYS_INLINE void scan_csv_chunk(const char* data, size_t size, Fn&& on_row) {
#if defined(__x86_64__) || defined(_M_X64)
#if defined(__AVX512BW__) && defined(__AVX512VL__) && defined(__AVX512F__)
    scan_csv_chunk_avx512<N>(data, size, static_cast<Fn&&>(on_row));
#elif defined(__AVX2__)
    scan_csv_chunk_avx2<N>(data, size, static_cast<Fn&&>(on_row));
#else
    scan_csv_chunk_scalar<N>(data, size, static_cast<Fn&&>(on_row));
#endif
#elif defined(__aarch64__)
    scan_csv_chunk_neon<N>(data, size, static_cast<Fn&&>(on_row));
#else
    scan_csv_chunk_scalar<N>(data, size, static_cast<Fn&&>(on_row));
#endif
}

} // namespace fast
