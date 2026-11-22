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

// Ultra-fast 64-bit integer mixer (1 multiply + 2 xor-shifts).
static inline uint64_t hash_u64(uint64_t x) {
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33;
    return x;
}

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
// Lossless, order-preserving 7-bit ASCII packing into 63 bits (len 1..9, bytes 1..127).
// Bit 63 == 0: inline packed key (s[0] in bits 56..62, ..., s[8] in bits 0..6, 0 = end).
// Bit 63 == 1: fallback interned string handle (len >= 10, or contains '\0' or byte >= 128).
// Integer `<` on inline keys is 100% identical to unsigned byte-wise memcmp!
// ============================================================================

static inline uint64_t pack8_ascii_be_to_56(uint64_t be) {
#if defined(__BMI2__)
    return _pext_u64(be, 0x7F7F7F7F7F7F7F7FULL);
#else
    uint64_t t = (be & 0x007F007F007F007FULL) | ((be & 0x7F007F007F007F00ULL) >> 1);
    t = (t & 0x00003FFF00003FFFULL) | ((t & 0x3FFF00003FFF0000ULL) >> 2);
    t = (t & 0x000000000FFFFFFFULL) | ((t & 0x0FFFFFFF00000000ULL) >> 4);
    return t;
#endif
}

static inline uint64_t unpack56_to_be(uint64_t p56) {
#if defined(__BMI2__)
    return _pdep_u64(p56, 0x7F7F7F7F7F7F7F7FULL);
#else
    uint64_t t = (p56 & 0x000000000FFFFFFFULL) | ((p56 & 0x00FFFFFFF0000000ULL) << 4);
    t = (t & 0x00003FFF00003FFFULL) | ((t & 0x0FFFC0000FFFC000ULL) << 2);
    t = (t & 0x007F007F007F007FULL) | ((t & 0x3F803F803F803F80ULL) << 1);
    return t;
#endif
}

static inline bool all_ascii_nonzero8(uint64_t w) {
    return ((w | (w - 0x0101010101010101ULL)) & 0x8080808080808080ULL) == 0;
}

// Merged branchless len 8 & 9 fast path (eliminates 20% len==8 vs len==9 mispredict!).
static inline uint64_t try_pack7(const char* p, size_t len) {
    if (__builtin_expect(len >= 8 && len <= 9, 1)) {
        uint64_t w;
        std::memcpy(&w, p, 8);
        bool is9 = (len == 9);
        const char* p8 = is9 ? (p + 8) : p;
        uint8_t raw8 = static_cast<uint8_t>(*p8);
        uint8_t b8 = is9 ? raw8 : 0U;
        bool bad8 = is9 & ((raw8 == 0) | (raw8 >= 128));
        if (__builtin_expect(!all_ascii_nonzero8(w) || bad8, 0)) return 0;
        return (pack8_ascii_be_to_56(__builtin_bswap64(w)) << 7) | static_cast<uint64_t>(b8);
    }
    if (__builtin_expect(len == 0 || len > 9, 0)) return 0;
    uint64_t key = 0;
    unsigned shift = 56;
    for (size_t i = 0; i < len; ++i) {
        uint8_t c = static_cast<uint8_t>(p[i]);
        if (__builtin_expect(c == 0 || c >= 128, 0)) return 0;
        key |= (static_cast<uint64_t>(c) << shift);
        shift -= 7;
    }
    return key;
}

// Unordered 7-bit ASCII pack for event_id (equality-only: skips bswap64!).
static inline uint64_t try_pack7_unordered(const char* p, size_t len) {
    if (__builtin_expect(len >= 8 && len <= 9, 1)) {
        uint64_t w;
        std::memcpy(&w, p, 8);
        bool is9 = (len == 9);
        const char* p8 = is9 ? (p + 8) : p;
        uint8_t raw8 = static_cast<uint8_t>(*p8);
        uint8_t b8 = is9 ? raw8 : 0U;
        bool bad8 = is9 & ((raw8 == 0) | (raw8 >= 128));
        if (__builtin_expect(!all_ascii_nonzero8(w) || bad8, 0)) return 0;
        return (pack8_ascii_be_to_56(w) << 7) | static_cast<uint64_t>(b8);
    }
    return try_pack7(p, len);
}

static inline size_t unpack7(uint64_t key, char* out) {
    uint64_t be = unpack56_to_be(key >> 7);
    uint64_t le = __builtin_bswap64(be);
    std::memcpy(out, &le, 8);
    uint8_t b8 = static_cast<uint8_t>(key & 0x7FU);
    if (__builtin_expect(b8 != 0, 1)) {
        out[8] = static_cast<char>(b8);
        return 9;
    }
    if (__builtin_expect(be == 0, 0)) return 0;
    return 8 - (static_cast<size_t>(__builtin_ctzll(be)) >> 3);
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

struct DateTable {
    struct Entry {
        uint64_t ym_bytes = 0; // "YYYY-MM-" (0 = invalid)
        uint16_t dd_bytes = 0; // "DD"
        int64_t day_sec = 0;
    };
    Entry entries[64];

    static inline uint32_t slot_for(const char* s) {
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

// 4-op borrow-free SWAR 8-ASCII-digit parser:
// Byte b in ['0'..'9'] iff (b & 0xF0) == 0x30 AND ((b + 0x06) & 0xF0) == 0x30!
static inline bool parse8_digits(uint64_t v, uint32_t& out) {
    uint64_t bad = ((v & 0xF0F0F0F0F0F0F0F0ULL) ^ 0x3030303030303030ULL) |
                   (((v + 0x0606060606060606ULL) & 0xF0F0F0F0F0F0F0F0ULL) ^ 0x3030303030303030ULL);
    if (__builtin_expect(bad != 0, 0)) return false;
    uint64_t lo = v & 0x0F0F0F0F0F0F0F0FULL;
    lo = (lo * 10 + (lo >> 8)) & 0x00FF00FF00FF00FFULL;
    lo = (lo * 100 + (lo >> 16)) & 0x0000FFFF0000FFFFULL;
    out = static_cast<uint32_t>((lo * 10000 + (lo >> 32)) & 0xFFFFFFFFULL);
    return true;
}

// 5-op SWAR "HH:MM:SS" parser (1 64-bit load, 1-multiply parallel pair extraction!).
static inline bool parse_hms_sec(const char* s, uint32_t& out_sec) {
    uint64_t v;
    std::memcpy(&v, s, 8);
    uint64_t bad = ((v & 0x0000FF0000FF0000ULL) ^ 0x00003A00003A0000ULL) |
                   ((v & 0xF0F000F0F000F0F0ULL) ^ 0x3030003030003030ULL) |
                   (((v + 0x0606000606000606ULL) & 0xF0F000F0F000F0F0ULL) ^ 0x3030003030003030ULL);
    if (__builtin_expect(bad != 0, 0)) return false;
    uint64_t lo = v & 0x0F0F000F0F000F0FULL;
    uint64_t pairs = lo * 10 + (lo >> 8);
    uint32_t hour = static_cast<uint32_t>(pairs & 0xFFU);
    uint32_t minute = static_cast<uint32_t>((pairs >> 24) & 0xFFU);
    uint32_t second = static_cast<uint32_t>((pairs >> 48) & 0xFFU);
    if (__builtin_expect((hour > 23) | (minute > 59) | (second > 59), 0)) return false;
    out_sec = hour * 3600 + minute * 60 + second;
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
static inline bool parse_ingest_f1_sec(const char* s, size_t len, const DateTable& dtab,
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
    uint64_t lo = v & 0x0F0F000F0F000F0FULL;
    uint64_t pairs = lo * 10 + (lo >> 8);
    uint32_t h = static_cast<uint32_t>(pairs & 0xFFU);
    uint32_t m = static_cast<uint32_t>((pairs >> 24) & 0xFFU);
    uint32_t sc = static_cast<uint32_t>((pairs >> 48) & 0xFFU);
    out_sec = out_day_sec + static_cast<int64_t>(h * 3600U + m * 60U + sc);
    return true;
}

// Strict F1 / F2 / F3 timestamp parser returning BOTH ms and sec (ZERO division, register date-hint reuse!).
static inline bool parse_timestamp_ms_sec(const char* s, size_t len, const DateTable& dtab,
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
        // Validate top 5 bytes of v1 (s[8..12]) in 4 SWAR ops!
        uint64_t v1m = (v1 & 0xFFFFFFFFFF000000ULL) | 0x0000000000303030ULL;
        uint64_t bad = ((v1m & 0xF0F0F0F0F0F0F0F0ULL) ^ 0x3030303030303030ULL) |
                       (((v1m + 0x0606060606060606ULL) & 0xF0F0F0F0F0F0F0F0ULL) ^ 0x3030303030303030ULL);
        if (__builtin_expect(bad != 0, 0)) return false;
        uint64_t lo = v1m & 0x0F0F0F0F0F0F0F0FULL;
        uint32_t d89 = static_cast<uint32_t>(((lo >> 24) & 0xFFU) * 10U + ((lo >> 32) & 0xFFU));
        uint32_t d1012 = static_cast<uint32_t>(((lo >> 40) & 0xFFU) * 100U + ((lo >> 48) & 0xFFU) * 10U + ((lo >> 56) & 0xFFU));
        int64_t sec = static_cast<int64_t>(hi8) * 100LL + static_cast<int64_t>(d89);
        out_sec = sec;
        out_ms = sec * 1000LL + static_cast<int64_t>(d1012);
        return true;
    }
    if (len == 25 && (s[19] == '+' || s[19] == '-') && s[22] == ':') {
        int64_t local_sec;
        if (!parse_civil19_sec(s, dtab, local_sec)) return false;
        uint32_t oh0 = static_cast<uint8_t>(s[20]) - '0';
        uint32_t oh1 = static_cast<uint8_t>(s[21]) - '0';
        uint32_t om0 = static_cast<uint8_t>(s[23]) - '0';
        uint32_t om1 = static_cast<uint8_t>(s[24]) - '0';
        if ((oh0 >= 10) | (oh1 >= 10) | (om0 >= 10) | (om1 >= 10)) return false;
        uint32_t oh = oh0 * 10 + oh1;
        uint32_t om = om0 * 10 + om1;
        if (oh > 23 || om > 59) return false;
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

static inline bool parse_timestamp_ms(const char* s, size_t len, const DateTable& dtab,
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

static inline bool parse_pos_u32_no_lz(const char* s, size_t len, uint32_t max_val, uint32_t& out) {
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
// Line-Aligned Loop-Free 128B / 64B SIMD CSV Scanners!
// Zero cm_this/cm_rem branch mispredicts, unrolled branchless comma extraction,
// zero-CMOV quote stripping for fields 1..N-1!
// ============================================================================

struct Field {
    const char* ptr;
    size_t len;
};

static inline void strip_field_branchless(const char* cur, const char* ce,
                                          Field& out, uint32_t& stripped) {
    static constexpr char DUMMY[1] = {'\0'};
    size_t len = static_cast<size_t>(ce - cur);
    bool ge2 = (len >= 2);
    const char* p0 = ge2 ? cur : DUMMY;
    const char* p1 = ge2 ? (ce - 1) : DUMMY;
    uint32_t q = static_cast<uint32_t>(*p0 == '"') & static_cast<uint32_t>(*p1 == '"');
    out.ptr = cur + q;
    out.len = len - (q + q);
    stripped += (q + q);
}

// Fast quote stripping for fields 1..N-1 where cur > line_start and ce <= line_end < end
// (so both cur[0] and ce[-1] are guaranteed in-bounds!).
static inline void strip_field_fast(const char* cur, const char* ce,
                                    Field& out, uint32_t& stripped) {
    size_t len = static_cast<size_t>(ce - cur);
    uint32_t q = static_cast<uint32_t>(len >= 2) &
                 static_cast<uint32_t>(cur[0] == '"') &
                 static_cast<uint32_t>(ce[-1] == '"');
    out.ptr = cur + q;
    out.len = len - (q + q);
    stripped += (q + q);
}

template <size_t N, typename Fn>
static inline void emit_line_from_commas(const char* line_start, const char* line_end,
                                         const char* const* commas, size_t ncommas,
                                         uint32_t line_quotes, Fn&& on_row) {
    if (__builtin_expect(ncommas != N - 1, 0)) return;
    Field fields[N];
    const char* cur = line_start;
    uint32_t stripped = 0;
    for (size_t i = 0; i < N - 1; ++i) {
        strip_field_branchless(cur, commas[i], fields[i], stripped);
        cur = commas[i] + 1;
    }
    strip_field_branchless(cur, line_end, fields[N - 1], stripped);
    if (__builtin_expect(stripped != line_quotes, 0)) return;
    on_row(fields);
}

template <size_t N, typename Fn>
static inline void scan_csv_chunk_scalar(const char* data, size_t size, Fn&& on_row) {
    const char* p = data;
    const char* end = data + size;
    const char* line_start = p;
    const char* commas[8];
    size_t ncommas = 0;
    uint32_t line_quotes = 0;
    while (p < end) {
        char c = *p;
        if (c == '\n') {
            if (p > line_start) {
                emit_line_from_commas<N>(line_start, p, commas, ncommas, line_quotes, on_row);
            }
            line_start = p + 1;
            ncommas = 0;
            line_quotes = 0;
        } else if (c == ',') {
            commas[ncommas & 7] = p;
            ncommas++;
        } else if (c == '"') {
            line_quotes++;
        }
        ++p;
    }
    if (end > line_start) {
        emit_line_from_commas<N>(line_start, end, commas, ncommas, line_quotes, on_row);
    }
}

// Shared loop-free 128B (N==7) and 64B (N==2,4) line-aligned scanner macros so GCC target("avx512bw")
// functions never call non-target-attributed template helpers (guaranteeing 100% inlining!).
#define FAST_SCAN_LINE128_7_LOOP(LOAD64_MACRO) \
    const char* p = data; \
    const char* end = data + size; \
    while (p + 128 <= end) { \
        uint64_t nl0, cm0, qt0, nl1, cm1, qt1; \
        LOAD64_MACRO(p, nl0, cm0, qt0); \
        LOAD64_MACRO(p + 64, nl1, cm1, qt1); \
        if (__builtin_expect((nl0 | nl1) == 0, 0)) { \
            const char* nl = static_cast<const char*>(std::memchr(p + 128, '\n', static_cast<size_t>(end - (p + 128)))); \
            const char* le = nl ? nl : end; \
            scan_csv_chunk_scalar<7>(p, static_cast<size_t>(le - p), on_row); \
            p = nl ? (nl + 1) : end; \
            continue; \
        } \
        unsigned nl_bit; \
        uint64_t lo_cm, hi_cm, lo_qt, hi_qt; \
        if (__builtin_expect(nl0 == 0, 1)) { \
            unsigned b1 = static_cast<unsigned>(__builtin_ctzll(nl1)); \
            nl_bit = 64 + b1; \
            uint64_t m1 = (1ULL << b1) - 1; \
            lo_cm = cm0; \
            hi_cm = cm1 & m1; \
            lo_qt = qt0; \
            hi_qt = qt1 & m1; \
        } else { \
            nl_bit = static_cast<unsigned>(__builtin_ctzll(nl0)); \
            uint64_t m0 = (1ULL << nl_bit) - 1; \
            lo_cm = cm0 & m0; \
            hi_cm = 0; \
            lo_qt = qt0 & m0; \
            hi_qt = 0; \
        } \
        uint32_t nlo = static_cast<uint32_t>(__builtin_popcountll(lo_cm)); \
        uint32_t nhi = static_cast<uint32_t>(__builtin_popcountll(hi_cm)); \
        if (__builtin_expect(nlo + nhi == 6, 1)) { \
            const char* commas[6]; \
            if (__builtin_expect(nlo >= 3 && nhi >= 1, 1)) { \
                commas[0] = p + static_cast<unsigned>(__builtin_ctzll(lo_cm)); lo_cm &= lo_cm - 1; \
                commas[1] = p + static_cast<unsigned>(__builtin_ctzll(lo_cm)); lo_cm &= lo_cm - 1; \
                commas[2] = p + static_cast<unsigned>(__builtin_ctzll(lo_cm)); lo_cm &= lo_cm - 1; \
                unsigned b5 = 63U - static_cast<unsigned>(__builtin_clzll(hi_cm)); \
                commas[5] = p + 64 + b5; \
                hi_cm ^= (1ULL << b5); \
                for (int ri = 3; ri <= 4; ++ri) { \
                    bool u = (lo_cm != 0); \
                    uint64_t m = u ? lo_cm : hi_cm; \
                    unsigned bit = static_cast<unsigned>(__builtin_ctzll(m)) + (u ? 0U : 64U); \
                    commas[ri] = p + bit; \
                    lo_cm = u ? (lo_cm & (lo_cm - 1)) : lo_cm; \
                    hi_cm = u ? hi_cm : (hi_cm & (hi_cm - 1)); \
                } \
            } else { \
                for (int ri = 0; ri < 6; ++ri) { \
                    bool u = (lo_cm != 0); \
                    uint64_t m = u ? lo_cm : hi_cm; \
                    unsigned bit = static_cast<unsigned>(__builtin_ctzll(m)) + (u ? 0U : 64U); \
                    commas[ri] = p + bit; \
                    lo_cm = u ? (lo_cm & (lo_cm - 1)) : lo_cm; \
                    hi_cm = u ? hi_cm : (hi_cm & (hi_cm - 1)); \
                } \
            } \
            uint32_t line_quotes = static_cast<uint32_t>(__builtin_popcountll(lo_qt) + __builtin_popcountll(hi_qt)); \
            const char* line_end = p + nl_bit; \
            Field fields[7]; \
            uint32_t stripped = 0; \
            strip_field_branchless(p, commas[0], fields[0], stripped); \
            strip_field_fast(commas[0] + 1, commas[1], fields[1], stripped); \
            strip_field_fast(commas[1] + 1, commas[2], fields[2], stripped); \
            strip_field_fast(commas[2] + 1, commas[3], fields[3], stripped); \
            strip_field_fast(commas[3] + 1, commas[4], fields[4], stripped); \
            strip_field_fast(commas[4] + 1, commas[5], fields[5], stripped); \
            strip_field_fast(commas[5] + 1, line_end, fields[6], stripped); \
            if (__builtin_expect(stripped == line_quotes, 1)) { \
                on_row(fields); \
            } \
        } \
        p += nl_bit + 1; \
    } \
    if (p < end) { \
        scan_csv_chunk_scalar<7>(p, static_cast<size_t>(end - p), on_row); \
    }

#define FAST_SCAN_LINE64_LOOP(LOAD64_MACRO) \
    static_assert(N == 2 || N == 4); \
    const char* p = data; \
    const char* end = data + size; \
    while (p + 64 <= end) { \
        uint64_t nl0, cm0, qt0; \
        LOAD64_MACRO(p, nl0, cm0, qt0); \
        if (__builtin_expect(nl0 == 0, 0)) { \
            if constexpr (N == 4) { \
                if (__builtin_expect(p + 128 <= end, 1)) { \
                    uint64_t nl1, cm1, qt1; \
                    LOAD64_MACRO(p + 64, nl1, cm1, qt1); \
                    if (__builtin_expect(nl1 != 0, 1)) { \
                        unsigned b1 = static_cast<unsigned>(__builtin_ctzll(nl1)); \
                        unsigned nl_bit = 64 + b1; \
                        uint64_t m1 = (1ULL << b1) - 1; \
                        uint64_t lo_cm = cm0, hi_cm = cm1 & m1; \
                        uint32_t ncm = static_cast<uint32_t>(__builtin_popcountll(lo_cm) + __builtin_popcountll(hi_cm)); \
                        if (__builtin_expect(ncm == 3, 1)) { \
                            const char* cs[3]; \
                            for (int ri = 0; ri < 3; ++ri) { \
                                bool u = (lo_cm != 0); \
                                uint64_t m = u ? lo_cm : hi_cm; \
                                unsigned bit = static_cast<unsigned>(__builtin_ctzll(m)) + (u ? 0U : 64U); \
                                cs[ri] = p + bit; \
                                lo_cm = u ? (lo_cm & (lo_cm - 1)) : lo_cm; \
                                hi_cm = u ? hi_cm : (hi_cm & (hi_cm - 1)); \
                            } \
                            uint32_t line_quotes = static_cast<uint32_t>(__builtin_popcountll(qt0) + __builtin_popcountll(qt1 & m1)); \
                            const char* line_end = p + nl_bit; \
                            Field fields[4]; \
                            uint32_t stripped = 0; \
                            strip_field_branchless(p, cs[0], fields[0], stripped); \
                            strip_field_fast(cs[0] + 1, cs[1], fields[1], stripped); \
                            strip_field_fast(cs[1] + 1, cs[2], fields[2], stripped); \
                            strip_field_fast(cs[2] + 1, line_end, fields[3], stripped); \
                            if (__builtin_expect(stripped == line_quotes, 1)) on_row(fields); \
                        } \
                        p += nl_bit + 1; \
                        continue; \
                    } \
                } \
            } \
            const char* nl = static_cast<const char*>(std::memchr(p + 64, '\n', static_cast<size_t>(end - (p + 64)))); \
            const char* le = nl ? nl : end; \
            scan_csv_chunk_scalar<N>(p, static_cast<size_t>(le - p), on_row); \
            p = nl ? (nl + 1) : end; \
            continue; \
        } \
        unsigned nl_bit = static_cast<unsigned>(__builtin_ctzll(nl0)); \
        uint64_t m0 = (1ULL << nl_bit) - 1; \
        uint64_t lcm = cm0 & m0; \
        if (__builtin_expect(static_cast<size_t>(__builtin_popcountll(lcm)) == N - 1, 1)) { \
            uint32_t line_quotes = static_cast<uint32_t>(__builtin_popcountll(qt0 & m0)); \
            const char* line_end = p + nl_bit; \
            Field fields[N]; \
            uint32_t stripped = 0; \
            if constexpr (N == 2) { \
                const char* c0 = p + static_cast<unsigned>(__builtin_ctzll(lcm)); \
                strip_field_branchless(p, c0, fields[0], stripped); \
                strip_field_fast(c0 + 1, line_end, fields[1], stripped); \
            } else if constexpr (N == 4) { \
                const char* c0 = p + static_cast<unsigned>(__builtin_ctzll(lcm)); lcm &= lcm - 1; \
                const char* c1 = p + static_cast<unsigned>(__builtin_ctzll(lcm)); lcm &= lcm - 1; \
                const char* c2 = p + static_cast<unsigned>(__builtin_ctzll(lcm)); \
                strip_field_branchless(p, c0, fields[0], stripped); \
                strip_field_fast(c0 + 1, c1, fields[1], stripped); \
                strip_field_fast(c1 + 1, c2, fields[2], stripped); \
                strip_field_fast(c2 + 1, line_end, fields[3], stripped); \
            } \
            if (__builtin_expect(stripped == line_quotes, 1)) { \
                on_row(fields); \
            } \
        } \
        p += nl_bit + 1; \
    } \
    if (p < end) { \
        scan_csv_chunk_scalar<N>(p, static_cast<size_t>(end - p), on_row); \
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
static inline void scan_csv_chunk_avx512(const char* data, size_t size, Fn&& on_row) {
    const __m512i v_nl = _mm512_set1_epi8('\n');
    const __m512i v_cm = _mm512_set1_epi8(',');
    const __m512i v_qt = _mm512_set1_epi8('"');
    if constexpr (N == 7) {
        FAST_SCAN_LINE128_7_LOOP(FAST_LOAD64_AVX512);
    } else {
        FAST_SCAN_LINE64_LOOP(FAST_LOAD64_AVX512);
    }
}
#else
template <size_t N, typename Fn>
__attribute__((target("avx512f,avx512bw,avx512vl,bmi2,popcnt")))
static void scan_csv_chunk_avx512(const char* data, size_t size, Fn&& on_row) {
    const __m512i v_nl = _mm512_set1_epi8('\n');
    const __m512i v_cm = _mm512_set1_epi8(',');
    const __m512i v_qt = _mm512_set1_epi8('"');
    if constexpr (N == 7) {
        FAST_SCAN_LINE128_7_LOOP(FAST_LOAD64_AVX512);
    } else {
        FAST_SCAN_LINE64_LOOP(FAST_LOAD64_AVX512);
    }
}

template <size_t N, typename Fn>
__attribute__((target("avx2,bmi2,popcnt")))
static void scan_csv_chunk_avx2(const char* data, size_t size, Fn&& on_row) {
    const __m256i v_nl = _mm256_set1_epi8('\n');
    const __m256i v_cm = _mm256_set1_epi8(',');
    const __m256i v_qt = _mm256_set1_epi8('"');
    if constexpr (N == 7) {
        FAST_SCAN_LINE128_7_LOOP(FAST_LOAD64_AVX2);
    } else {
        FAST_SCAN_LINE64_LOOP(FAST_LOAD64_AVX2);
    }
}
#endif

#elif defined(__aarch64__)

static inline uint64_t neon_movemask64(uint8x16_t m0, uint8x16_t m1, uint8x16_t m2, uint8x16_t m3) {
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
static void scan_csv_chunk_neon(const char* data, size_t size, Fn&& on_row) {
    const uint8x16_t v_nl = vdupq_n_u8('\n');
    const uint8x16_t v_cm = vdupq_n_u8(',');
    const uint8x16_t v_qt = vdupq_n_u8('"');
    if constexpr (N == 7) {
        FAST_SCAN_LINE128_7_LOOP(FAST_LOAD64_NEON);
    } else {
        FAST_SCAN_LINE64_LOOP(FAST_LOAD64_NEON);
    }
}

#endif

template <size_t N, typename Fn>
static inline void scan_csv_chunk(const char* data, size_t size, Fn&& on_row) {
#if defined(__x86_64__) || defined(_M_X64)
#if defined(__AVX512BW__) && defined(__AVX512VL__) && defined(__AVX512F__)
    scan_csv_chunk_avx512<N>(data, size, static_cast<Fn&&>(on_row));
#else
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
#endif
#elif defined(__aarch64__)
    scan_csv_chunk_neon<N>(data, size, static_cast<Fn&&>(on_row));
#else
    scan_csv_chunk_scalar<N>(data, size, static_cast<Fn&&>(on_row));
#endif
}

} // namespace fast
