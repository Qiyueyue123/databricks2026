#pragma once

#include <algorithm>
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
static FAST_ALWAYS_INLINE bool parse_hex128_scalar(const char* s, uint64_t& hi, uint64_t& lo) {
    uint32_t a, b, c, d;
    if (__builtin_expect(parse_hex8(s, a) && parse_hex8(s + 8, b) &&
                         parse_hex8(s + 16, c) && parse_hex8(s + 24, d), 1)) {
        hi = (static_cast<uint64_t>(a) << 32) | b;
        lo = (static_cast<uint64_t>(c) << 32) | d;
        return true;
    }
    return false;
}

static FAST_ALWAYS_INLINE bool parse_hex128(const char* s, uint64_t& hi, uint64_t& lo) {
#if defined(__AVX2__)
    __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(s));
    __m256i is_dig = _mm256_and_si256(
        _mm256_cmpgt_epi8(v, _mm256_set1_epi8('0' - 1)),
        _mm256_cmpgt_epi8(_mm256_set1_epi8('9' + 1), v));
    __m256i is_af = _mm256_and_si256(
        _mm256_cmpgt_epi8(v, _mm256_set1_epi8('a' - 1)),
        _mm256_cmpgt_epi8(_mm256_set1_epi8('f' + 1), v));
    __m256i ok = _mm256_or_si256(is_dig, is_af);
    if (__builtin_expect(_mm256_movemask_epi8(ok) != -1, 0)) return false;
    __m256i sub = _mm256_add_epi8(
        _mm256_set1_epi8('0'),
        _mm256_and_si256(is_af, _mm256_set1_epi8('a' - '0' - 10)));
    __m256i nib = _mm256_sub_epi8(v, sub);
    __m256i pairs16 = _mm256_maddubs_epi16(nib, _mm256_set1_epi16(0x0110));
    __m128i bytes16 = _mm_packus_epi16(_mm256_castsi256_si128(pairs16),
                                       _mm256_extracti128_si256(pairs16, 1));
    uint64_t h_le = static_cast<uint64_t>(_mm_cvtsi128_si64(bytes16));
    uint64_t l_le = static_cast<uint64_t>(_mm_extract_epi64(bytes16, 1));
    hi = __builtin_bswap64(h_le);
    lo = __builtin_bswap64(l_le);
    return true;
#else
    return parse_hex128_scalar(s, hi, lo);
#endif
}

static FAST_ALWAYS_INLINE uint64_t make_hex_key(uint64_t hi) {
    return HEX_KEY_TAG | ((hi >> 9) << 8);
}

// Format 128-bit hex value (pfx + 32 lowercase hex chars = 33 bytes).
static inline void format_hex128_scalar(char pfx, uint64_t hi, uint64_t lo, char* out) {
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

static inline void format_hex128(char pfx, uint64_t hi, uint64_t lo, char* out) {
#if defined(__SSSE3__)
    out[0] = pfx;
    uint64_t h_le = __builtin_bswap64(hi);
    uint64_t l_le = __builtin_bswap64(lo);
    __m128i b = _mm_set_epi64x(static_cast<long long>(l_le), static_cast<long long>(h_le));
    __m128i mask0f = _mm_set1_epi8(0x0F);
    __m128i hi_nib = _mm_and_si128(_mm_srli_epi16(b, 4), mask0f);
    __m128i lo_nib = _mm_and_si128(b, mask0f);
    __m128i out_lo = _mm_unpacklo_epi8(hi_nib, lo_nib);
    __m128i out_hi = _mm_unpackhi_epi8(hi_nib, lo_nib);
    __m128i lut = _mm_setr_epi8('0','1','2','3','4','5','6','7','8','9','a','b','c','d','e','f');
    _mm_storeu_si128(reinterpret_cast<__m128i*>(out + 1),  _mm_shuffle_epi8(lut, out_lo));
    _mm_storeu_si128(reinterpret_cast<__m128i*>(out + 17), _mm_shuffle_epi8(lut, out_hi));
#else
    format_hex128_scalar(pfx, hi, lo, out);
#endif
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

static constexpr int64_t days_from_civil(int32_t y, uint32_t m, uint32_t d) {
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

// Strict F1 / F2 / F3 scalar timestamp parser returning BOTH ms and sec (ZERO division, register date-hint reuse!).
static FAST_ALWAYS_INLINE bool parse_timestamp_ms_sec_scalar(const char* s, size_t len, const DateTable& dtab,
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

#if defined(__AVX2__)
// Compile-time branch-free AVX2 timestamp tables covering ALL 3 formats (F3=13, F1=20, F2=25) with ZERO branch mispredicts!
struct alignas(32) Avx2TsForm {
    uint8_t t8[32]{};
    uint8_t l8[32]{};
    int8_t wa[32]{};
    int8_t wb[32]{};
    uint16_t lo[16]{};
    uint16_t rg[16]{};
    int16_t m[16]{};
};

struct Avx2TsTables {
    static constexpr uint32_t Y0 = 1900;
    static constexpr uint32_t NY = 400;
    Avx2TsForm form[3]{};
    int32_t ystart[NY + 1]{};
    int16_t moff[2][16] = {
        {0, 0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 0, 0, 0},
        {0, 0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335, 0, 0, 0},
    };
    uint8_t mdays[2][16] = {
        {0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31, 0, 0, 0},
        {0, 31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31, 0, 0, 0},
    };
    uint8_t kind[26] = {3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 3, 3, 3, 3, 3, 3, 1, 3, 3, 3, 3, 2};

    constexpr Avx2TsTables() {
        for (uint32_t i = 0; i <= NY; ++i) {
            ystart[i] = static_cast<int32_t>(days_from_civil(static_cast<int32_t>(Y0 + i), 1, 1));
        }
        const char* pats[3] = {"DDDDDDDDDDDDD", "DDDD-DD-DDTDD:DD:DDZ", "DDDD-DD-DDTDD:DD:DD+DD:DD"};
        for (int k = 0; k < 3; ++k) {
            auto& f = form[k];
            for (int i = 0; i < 32; ++i) f.l8[i] = 0xFF;
            for (int i = 0; pats[k][i] != '\0'; ++i) {
                char c = pats[k][i];
                f.t8[i] = (c == 'D') ? static_cast<uint8_t>('0') : static_cast<uint8_t>(c);
                f.l8[i] = (c == 'D') ? 9 : ((c == '+') ? 2 : 0);
            }
            for (int i = 0; i < 16; ++i) f.rg[i] = 0xFFFF;
            if (k == 0) {
                for (int i = 0; i < 12; i += 2) { f.wa[i] = 10; f.wa[i + 1] = 1; }
                f.wa[12] = 1;
                for (int i = 0; i < 6; i += 2) { f.m[i] = 100; f.m[i + 1] = 1; }
                f.m[6] = 1;
            } else {
                const int fa[5] = {0, 1, 4, 7, 10};
                const int fb[4] = {2, 5, 8, 11};
                for (int j = 0; j < 5; ++j) { f.wa[2 * fa[j]] = 10; f.wa[2 * fa[j] + 1] = 1; }
                for (int j = 0; j < 4; ++j) { f.wb[2 * fb[j]] = 10; f.wb[2 * fb[j] + 1] = 1; }
                f.lo[2] = 1; f.rg[2] = 11;
                f.lo[4] = 1; f.rg[4] = 30;
                f.rg[5] = 23; f.rg[7] = 59; f.rg[8] = 59;
                f.m[0] = 100; f.m[1] = 1; f.m[2] = 1; f.m[4] = 24; f.m[5] = 1; f.m[7] = 60; f.m[8] = 1;
                if (k == 2) {
                    f.rg[10] = 23; f.rg[11] = 59;
                    f.m[10] = 60; f.m[11] = 1;
                }
            }
        }
    }
};

static constexpr Avx2TsTables AVX2_TS{};

// Branch-free AVX2 F1/F2/F3 timestamp validator + parser (reads s[0..32] safely via buf_end guard!).
static FAST_ALWAYS_INLINE bool parse_timestamp_ms_sec(const char* s, size_t len, const DateTable& dtab,
                                                      int64_t& out_ms, int64_t& out_sec,
                                                      const char* buf_end,
                                                      uint64_t hint_ym = 0, uint16_t hint_dd = 0, int64_t hint_day_sec = 0) {
    unsigned k = (len < 26) ? AVX2_TS.kind[len] : 3U;
    if (__builtin_expect(k == 3U || s + 33 > buf_end, 0)) {
        return parse_timestamp_ms_sec_scalar(s, len, dtab, out_ms, out_sec, hint_ym, hint_dd, hint_day_sec);
    }
    const Avx2TsForm& f = AVX2_TS.form[k];
    __m256i x = _mm256_sub_epi8(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(s)),
                                _mm256_load_si256(reinterpret_cast<const __m256i*>(f.t8)));
    __m256i y = _mm256_sub_epi8(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(s + 1)),
                                _mm256_set1_epi8('0'));
    __m256i l8 = _mm256_load_si256(reinterpret_cast<const __m256i*>(f.l8));
    __m256i u = _mm256_add_epi16(_mm256_maddubs_epi16(x, _mm256_load_si256(reinterpret_cast<const __m256i*>(f.wa))),
                                 _mm256_maddubs_epi16(y, _mm256_load_si256(reinterpret_cast<const __m256i*>(f.wb))));
    __m256i rg = _mm256_load_si256(reinterpret_cast<const __m256i*>(f.rg));
    __m256i ur = _mm256_sub_epi16(u, _mm256_load_si256(reinterpret_cast<const __m256i*>(f.lo)));
    __m256i good = _mm256_and_si256(_mm256_cmpeq_epi8(_mm256_max_epu8(x, l8), l8),
                                    _mm256_cmpeq_epi16(_mm256_max_epu16(ur, rg), rg));
    unsigned bad = static_cast<unsigned>(_mm256_movemask_epi8(good)) != 0xFFFFFFFFU;
    bad |= static_cast<unsigned>(k == 2U) & static_cast<unsigned>(s[19] == ',');

    __m256i g = _mm256_madd_epi16(u, _mm256_load_si256(reinterpret_cast<const __m256i*>(f.m)));
    __m128i g0 = _mm256_castsi256_si128(g);
    uint64_t a = static_cast<uint64_t>(_mm_cvtsi128_si64(g0));
    uint64_t b = static_cast<uint64_t>(_mm_extract_epi64(g0, 1));
    uint64_t c = static_cast<uint64_t>(_mm_cvtsi128_si64(_mm256_extracti128_si256(g, 1)));
    uint32_t g1 = static_cast<uint32_t>(a);
    uint32_t g2 = static_cast<uint32_t>(a >> 32);
    uint32_t g3 = static_cast<uint32_t>(b);
    uint32_t g4 = static_cast<uint32_t>(b >> 32);

    int64_t f3_ms = static_cast<int64_t>(g1) * 1000000000LL +
                    static_cast<int64_t>(g2) * 100000LL +
                    static_cast<int64_t>(g3) * 10LL +
                    static_cast<int64_t>(g4);
    int64_t f3_sec = static_cast<int64_t>(g1) * 1000000LL +
                     static_cast<int64_t>(g2) * 100LL +
                     static_cast<int64_t>(g3 / 100U);

    unsigned cm = -static_cast<unsigned>(k != 0U);
    unsigned yi = (g1 - Avx2TsTables::Y0) & cm;
    unsigned out_yr = static_cast<unsigned>(yi >= Avx2TsTables::NY);
    if (__builtin_expect((out_yr & !bad) != 0U, 0)) {
        return parse_timestamp_ms_sec_scalar(s, len, dtab, out_ms, out_sec, hint_ym, hint_dd, hint_day_sec);
    }
    yi &= (out_yr - 1U);
    unsigned mo = g2 & 15U;
    unsigned da = static_cast<unsigned>(_mm256_extract_epi16(u, 4));
    unsigned leap = static_cast<unsigned>(AVX2_TS.ystart[yi + 1] - AVX2_TS.ystart[yi]) - 365U;
    bad |= cm & static_cast<unsigned>(da > AVX2_TS.mdays[leap][mo]);

    int64_t hours = static_cast<int64_t>(AVX2_TS.ystart[yi] + AVX2_TS.moff[leap][mo] - 1) * 24LL +
                    static_cast<int64_t>(g3);
    int64_t off_sec = static_cast<int64_t>(static_cast<uint32_t>(c >> 32)) * 60LL *
                      static_cast<int64_t>(1 - ((static_cast<unsigned char>(s[19]) >> 1) & 2));
    int64_t cal_sec = hours * 3600LL + static_cast<int64_t>(g4) +
                      static_cast<int64_t>(static_cast<uint32_t>(c)) - off_sec;
    int64_t cal_ms = cal_sec * 1000LL;

    int64_t km = static_cast<int64_t>(static_cast<int32_t>(cm));
    out_ms = (cal_ms & km) | (f3_ms & ~km);
    out_sec = (cal_sec & km) | (f3_sec & ~km);
    return bad == 0U;
}
#else
static FAST_ALWAYS_INLINE bool parse_timestamp_ms_sec(const char* s, size_t len, const DateTable& dtab,
                                                      int64_t& out_ms, int64_t& out_sec,
                                                      const char* /*buf_end*/,
                                                      uint64_t hint_ym = 0, uint16_t hint_dd = 0, int64_t hint_day_sec = 0) {
    return parse_timestamp_ms_sec_scalar(s, len, dtab, out_ms, out_sec, hint_ym, hint_dd, hint_day_sec);
}
#endif

static FAST_ALWAYS_INLINE bool parse_timestamp_ms(const char* s, size_t len, const DateTable& dtab,
                                                  int64_t& out_ms, const char* buf_end = nullptr) {
    int64_t dummy_sec = 0;
    return parse_timestamp_ms_sec(s, len, dtab, out_ms, dummy_sec, buf_end);
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

// ============================================================================
// k6 Streaming-Mask 4KB Segment CSV Scanner:
// - Out-of-line stage1 gets ALL 16 x86-64 registers (ZERO spills!)
// - 5-op cross-block exact dialect quote rule (ZERO __uint128_t shifts!)
// - BMI2 pext/pdep flag-byte stream -> single u64/u32/u16 check per line!
// - ZERO byte loads for quote stripping, handles lines up to 4096B on fast path!
// - Leaves >= 64 readable bytes past every scanned block (safe 33B AVX2 loads!)
// ============================================================================

static constexpr size_t K6_NB = 64;
static constexpr size_t K6_SEG = K6_NB * 64; // 4096 bytes
static constexpr size_t K6_CAP = K6_SEG + 64;
static constexpr size_t K6_NSENT = 16;
static constexpr uint8_t K6_F_Q = 1;
static constexpr uint8_t K6_F_NL = 2;
static constexpr uint8_t K6_F_BAD = 4;
static constexpr uint64_t K6_L1 = 0x0101010101010101ULL;

struct alignas(64) K6Ring {
    uint32_t e[K6_CAP];
    uint8_t fl[K6_CAP];
};

__attribute__((cold, noinline))
static size_t k6_block_slow(uint64_t f, uint64_t qs, uint64_t n, uint64_t bad,
                            uint32_t base, uint32_t* ep, uint8_t* flp) {
    size_t i = 0;
    while (f) {
        uint32_t tz = static_cast<uint32_t>(__builtin_ctzll(f));
        ep[i] = base + tz;
        flp[i] = static_cast<uint8_t>(((qs >> tz) & 1ULL) |
                                      (((n >> tz) & 1ULL) << 1) |
                                      (((bad >> tz) & 1ULL) << 2));
        f &= f - 1ULL;
        ++i;
    }
    return i;
}

#if defined(__aarch64__)
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
#endif

__attribute__((noinline))
static size_t k6_stage1(const char* p, size_t nblks, K6Ring& ring) {
#if defined(__AVX512BW__) && defined(__AVX512VL__) && defined(__AVX512F__)
    const __m512i v_nl = _mm512_set1_epi8('\n');
    const __m512i v_cm = _mm512_set1_epi8(',');
    const __m512i v_qt = _mm512_set1_epi8('"');
#elif defined(__AVX2__)
    const __m256i v_nl = _mm256_set1_epi8('\n');
    const __m256i v_cm = _mm256_set1_epi8(',');
    const __m256i v_qt = _mm256_set1_epi8('"');
#elif defined(__aarch64__)
    const uint8x16_t v_nl = vdupq_n_u8('\n');
    const uint8x16_t v_cm = vdupq_n_u8(',');
    const uint8x16_t v_qt = vdupq_n_u8('"');
#endif

    uint32_t* e = ring.e;
    uint8_t* fl = ring.fl;
    e[0] = UINT32_MAX;
    fl[0] = K6_F_NL;

    uint64_t dp = 1ULL << 63;
    uint64_t qp = 0;
    uint8_t bin = 1;
    size_t kb = 1;
    uint32_t base = 0;

    for (size_t bi = 0; bi < nblks; ++bi, base += 64U) {
        const char* q = p + base;
        __builtin_prefetch(q + 1024, 0, 1);

        uint64_t n, c, qq;
#if defined(__AVX512BW__) && defined(__AVX512VL__) && defined(__AVX512F__)
        __m512i v = _mm512_loadu_si512(reinterpret_cast<const void*>(q));
        n = _mm512_cmpeq_epi8_mask(v, v_nl);
        c = _mm512_cmpeq_epi8_mask(v, v_cm);
        qq = _mm512_cmpeq_epi8_mask(v, v_qt);
#elif defined(__AVX2__)
        __m256i v0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q));
        __m256i v1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q + 32));
        n = static_cast<uint32_t>(_mm256_movemask_epi8(_mm256_cmpeq_epi8(v0, v_nl))) |
            (static_cast<uint64_t>(static_cast<uint32_t>(_mm256_movemask_epi8(_mm256_cmpeq_epi8(v1, v_nl)))) << 32);
        c = static_cast<uint32_t>(_mm256_movemask_epi8(_mm256_cmpeq_epi8(v0, v_cm))) |
            (static_cast<uint64_t>(static_cast<uint32_t>(_mm256_movemask_epi8(_mm256_cmpeq_epi8(v1, v_cm)))) << 32);
        qq = static_cast<uint32_t>(_mm256_movemask_epi8(_mm256_cmpeq_epi8(v0, v_qt))) |
             (static_cast<uint64_t>(static_cast<uint32_t>(_mm256_movemask_epi8(_mm256_cmpeq_epi8(v1, v_qt)))) << 32);
#elif defined(__aarch64__)
        const uint8_t* up = reinterpret_cast<const uint8_t*>(q);
        uint8x16_t a0 = vld1q_u8(up);
        uint8x16_t a1 = vld1q_u8(up + 16);
        uint8x16_t a2 = vld1q_u8(up + 32);
        uint8x16_t a3 = vld1q_u8(up + 48);
        n = neon_movemask64(vceqq_u8(a0, v_nl), vceqq_u8(a1, v_nl), vceqq_u8(a2, v_nl), vceqq_u8(a3, v_nl));
        c = neon_movemask64(vceqq_u8(a0, v_cm), vceqq_u8(a1, v_cm), vceqq_u8(a2, v_cm), vceqq_u8(a3, v_cm));
        qq = neon_movemask64(vceqq_u8(a0, v_qt), vceqq_u8(a1, v_qt), vceqq_u8(a2, v_qt), vceqq_u8(a3, v_qt));
#else
        n = c = qq = 0;
        for (unsigned k = 0; k < 64U; ++k) {
            n |= static_cast<uint64_t>(q[k] == '\n') << k;
            c |= static_cast<uint64_t>(q[k] == ',') << k;
            qq |= static_cast<uint64_t>(q[k] == '"') << k;
        }
#endif
        uint64_t d = c | n;
        uint64_t fs = (d << 1) | (dp >> 63);
        uint64_t qs = (qq << 1) | (qp >> 63);
        uint64_t fss = (d << 2) | (dp >> 62);
        uint64_t op = qq & fs;

        uint64_t sum = 0;
#if defined(__x86_64__) || defined(_M_X64)
        unsigned long long sum_ull = 0;
        uint8_t bout = _subborrow_u64(bin, static_cast<unsigned long long>(op),
                                      static_cast<unsigned long long>(d), &sum_ull);
        sum = static_cast<uint64_t>(sum_ull);
#else
        __uint128_t diff = static_cast<__uint128_t>(op) - static_cast<__uint128_t>(d) - static_cast<__uint128_t>(bin);
        sum = static_cast<uint64_t>(diff);
        uint8_t bout = static_cast<uint8_t>((diff >> 64) != 0);
#endif
        uint64_t od = sum & d;
        uint64_t bad = (qs & ~(fss ^ d)) | (d & (od ^ qs));
        dp = d;
        qp = qq;
        bin = bout;

        size_t cnt = static_cast<size_t>(__builtin_popcountll(d));
        if (__builtin_expect((bad != 0ULL) | (cnt > 8U), 0)) {
            kb += k6_block_slow(d | bad, qs, n, bad, base, e + kb, fl + kb);
            continue;
        }

        uint32_t* ep = e + kb;
        uint64_t f = d;
#if defined(__BMI2__)
        for (int i = 0; i < 8; ++i) {
            ep[i] = base + static_cast<uint32_t>(__builtin_ctzll(f));
            f &= f - 1ULL;
        }
        uint64_t fl8 = _pdep_u64(_pext_u64(qs, d), K6_L1) +
                       (_pdep_u64(_pext_u64(n, d), K6_L1) << 1);
        std::memcpy(fl + kb, &fl8, 8);
#else
        uint8_t* flp = fl + kb;
        for (int i = 0; i < 8; ++i) {
            uint32_t tz = static_cast<uint32_t>(__builtin_ctzll(f)) & 63U;
            ep[i] = base + tz;
            flp[i] = static_cast<uint8_t>(((qs >> tz) & 1ULL) | (((n >> tz) & 1ULL) << 1));
            f &= f - 1ULL;
        }
#endif
        kb += cnt;
    }

    uint32_t big = static_cast<uint32_t>(nblks << 6);
    for (size_t i = 0; i < K6_NSENT; ++i) {
        e[kb + i] = big;
        fl[kb + i] = K6_F_BAD;
    }
    return kb;
}

template <size_t N, typename Fn>
static FAST_ALWAYS_INLINE void scan_csv_chunk(const char* data, size_t size, Fn&& on_row) {
    const char* cur = data;
    const char* const end = data + size;
    K6Ring ring;
    Field fields[N];

    // Leave >= 64 readable bytes past scanned segment so AVX2 33B loads on simple lines never cross EOF!
    while (cur + 128 <= end) {
        const size_t avail = static_cast<size_t>(end - cur);
        const size_t nblks = std::min<size_t>((avail - 64) >> 6, K6_NB);
        const size_t cnt = k6_stage1(cur, nblks, ring);
        const uint32_t* e = ring.e;
        const uint8_t* fl = ring.fl;

        size_t k = 1;
        while (true) {
            bool simple = false;
            if constexpr (N == 7) {
                uint64_t x;
                std::memcpy(&x, fl + k, 8);
                simple = ((x & 0x0006060606060606ULL) == 0x0002000000000000ULL);
            } else if constexpr (N == 4) {
                uint32_t x;
                std::memcpy(&x, fl + k, 4);
                simple = ((x & 0x06060606U) == 0x02000000U);
            } else if constexpr (N == 2) {
                uint16_t x;
                std::memcpy(&x, fl + k, 2);
                simple = ((x & 0x0606U) == 0x0200U);
            } else {
                simple = true;
                for (size_t j = 0; j + 1 < N; ++j) {
                    if ((fl[k + j] & (K6_F_NL | K6_F_BAD)) != 0) simple = false;
                }
                if ((fl[k + N - 1] & (K6_F_NL | K6_F_BAD)) != K6_F_NL) simple = false;
            }

            if (__builtin_expect(simple, 1)) {
                uint32_t prev = e[k - 1] + 1U;
                for (size_t j = 0; j < N; ++j) {
                    uint32_t q = fl[k + j] & 1U;
                    uint32_t ej = e[k + j];
                    fields[j] = Field{cur + prev + q, (ej - prev) - (q << 1)};
                    prev = ej + 1U;
                }
                on_row(fields);
                k += N;
                continue;
            }

            size_t j = k;
            while (j < cnt && (fl[j] & K6_F_NL) == 0) ++j;
            if (__builtin_expect(j < cnt, 1)) {
                uint32_t ls = e[k - 1] + 1U;
                if (split_line_exact<N>(cur + ls, cur + e[j], fields)) {
                    on_row(fields);
                }
                k = j + 1;
                continue;
            }
            break;
        }

        uint32_t ls = e[k - 1] + 1U;
        if (__builtin_expect(ls == 0U, 0)) {
            const char* nl = static_cast<const char*>(
                std::memchr(cur + (nblks << 6), '\n', static_cast<size_t>(end - (cur + (nblks << 6)))));
            const char* le = nl ? nl : end;
            if (split_line_exact<N>(cur, le, fields)) on_row(fields);
            cur = nl ? (nl + 1) : end;
        } else {
            cur += ls;
        }
    }

    if (cur < end) {
        scan_csv_chunk_scalar<N>(cur, static_cast<size_t>(end - cur), on_row);
    }
}

} // namespace fast
