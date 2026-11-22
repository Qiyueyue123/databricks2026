#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <fcntl.h>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <sys/mman.h>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <vector>

#if defined(__linux__)
#include <sched.h>
#endif

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

#include "csv_fast.hpp"
#include "parquet_writer.hpp"

namespace {

constexpr int64_t DAY_MS = 86400000LL;
constexpr int64_t WEEK_MS = 604800000LL;
constexpr int64_t SESSION_GAP_MS = 1800000LL;
constexpr uint32_t SESSION_GAP_U32 = 1800000U;
constexpr size_t READ_CHUNK_SIZE = 2 * 1024 * 1024;

using Clock = std::chrono::steady_clock;

double elapsed_sec(Clock::time_point t0) {
    return std::chrono::duration<double>(Clock::now() - t0).count();
}

size_t next_pow2(size_t n) {
    size_t p = 1;
    while (p < n) p <<= 1;
    return p;
}

void ensure_dir(const std::string& path) {
    mkdir(path.c_str(), 0777);
}

static bool g_use_huge = true;

// 2MB-aligned anonymous mmap with MADV_HUGEPAGE (zero munmap -> zero TLB shootdowns).
static inline void* alloc_huge_raw(size_t bytes) {
#if defined(__linux__) && defined(MADV_HUGEPAGE)
    if (g_use_huge && bytes >= 2 * 1024 * 1024) {
        constexpr size_t HP = 2ULL << 20;
        size_t aligned = (bytes + HP - 1) & ~(HP - 1);
        void* raw = mmap(nullptr, aligned + HP, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (raw != MAP_FAILED) {
            uintptr_t addr = (reinterpret_cast<uintptr_t>(raw) + HP - 1) & ~(HP - 1);
            void* p = reinterpret_cast<void*>(addr);
            madvise(p, aligned, MADV_HUGEPAGE);
            return p;
        }
    }
#endif
    void* p = nullptr;
    if (posix_memalign(&p, 64, std::max<size_t>(64, bytes)) == 0) return p;
    return std::malloc(bytes);
}

static size_t detect_cpus() {
    size_t n = std::thread::hardware_concurrency();
    if (n == 0) n = 1;
#if defined(__linux__)
    cpu_set_t set;
    CPU_ZERO(&set);
    if (sched_getaffinity(0, sizeof(set), &set) == 0) {
        size_t aff = 0;
        for (int c = 0; c < CPU_SETSIZE; ++c) {
            if (CPU_ISSET(c, &set)) ++aff;
        }
        if (aff > 0 && aff < n) n = aff;
    }
    if (FILE* f = std::fopen("/sys/fs/cgroup/cpu.max", "r")) {
        char qbuf[64]{};
        long long period = 0;
        if (std::fscanf(f, "%63s %lld", qbuf, &period) == 2 && period > 0 && qbuf[0] != 'm') {
            long long quota = std::atoll(qbuf);
            if (quota > 0) {
                size_t cg = static_cast<size_t>((quota + period - 1) / period);
                if (cg < 1) cg = 1;
                if (cg < n) n = cg;
            }
        }
        std::fclose(f);
    } else if (FILE* fq = std::fopen("/sys/fs/cgroup/cpu/cpu.cfs_quota_us", "r")) {
        long long quota = -1, period = 100000;
        if (std::fscanf(fq, "%lld", &quota) == 1 && quota > 0) {
            if (FILE* fp = std::fopen("/sys/fs/cgroup/cpu/cpu.cfs_period_us", "r")) {
                long long p = 0;
                if (std::fscanf(fp, "%lld", &p) == 1 && p > 0) period = p;
                std::fclose(fp);
            }
            size_t cg = static_cast<size_t>((quota + period - 1) / period);
            if (cg < 1) cg = 1;
            if (cg < n) n = cg;
        }
        std::fclose(fq);
    }
#endif
    return std::clamp<size_t>(n, 1, 16);
}

// Persistent low-latency native 32-bit futex-backed worker pool!
class ThreadTeam {
    size_t n_ = 1;
    std::vector<std::thread> workers_;
    std::atomic<uint32_t> epoch_{0};
    std::atomic<uint32_t> done_{0};
    std::atomic<bool> stop_{false};
    const std::function<void(size_t)>* cur_fn_ = nullptr;

    static inline void cpu_relax() {
#if defined(__x86_64__) || defined(_M_X64)
        __builtin_ia32_pause();
#elif defined(__aarch64__)
        asm volatile("yield" ::: "memory");
#endif
    }

    void worker_loop(size_t tid) {
        uint32_t seen = 0;
        while (true) {
            uint32_t ep = epoch_.load(std::memory_order_acquire);
            if (ep == seen) {
                for (int s = 0; s < 1000; ++s) {
                    cpu_relax();
                    ep = epoch_.load(std::memory_order_acquire);
                    if (ep != seen) break;
                }
                while (ep == seen) {
                    epoch_.wait(seen, std::memory_order_acquire);
                    ep = epoch_.load(std::memory_order_acquire);
                }
            }
            if (stop_.load(std::memory_order_relaxed)) return;
            seen = ep;
            (*cur_fn_)(tid);
            if (done_.fetch_add(1, std::memory_order_release) + 1 == static_cast<uint32_t>(n_ - 1)) {
                done_.notify_one();
            }
        }
    }

public:
    explicit ThreadTeam(size_t n) : n_(n) {
        if (n_ > 1) {
            workers_.reserve(n_ - 1);
            for (size_t t = 1; t < n_; ++t) {
                workers_.emplace_back(&ThreadTeam::worker_loop, this, t);
            }
        }
    }

    ~ThreadTeam() {
        if (!workers_.empty()) {
            stop_.store(true, std::memory_order_relaxed);
            epoch_.fetch_add(1, std::memory_order_release);
            epoch_.notify_all();
            for (auto& w : workers_) w.join();
        }
    }

    size_t size() const { return n_; }

    template <typename Fn>
    void run(Fn&& fn) {
        if (n_ <= 1) {
            fn(0);
            return;
        }
        std::function<void(size_t)> bound = [&](size_t tid) { fn(tid); };
        cur_fn_ = &bound;
        done_.store(0, std::memory_order_relaxed);
        epoch_.fetch_add(1, std::memory_order_release);
        epoch_.notify_all();
        fn(0);
        uint32_t target = static_cast<uint32_t>(n_ - 1);
        uint32_t d = done_.load(std::memory_order_acquire);
        if (d < target) {
            for (int s = 0; s < 1000; ++s) {
                cpu_relax();
                d = done_.load(std::memory_order_acquire);
                if (d >= target) return;
            }
            while (d < target) {
                done_.wait(d, std::memory_order_acquire);
                d = done_.load(std::memory_order_acquire);
            }
        }
    }
};

std::vector<std::string> list_csv_files(const std::string& dir_path) {
    std::vector<std::string> files;
    DIR* dir = opendir(dir_path.c_str());
    if (!dir) return files;
    while (struct dirent* ent = readdir(dir)) {
        const char* name = ent->d_name;
        size_t len = std::strlen(name);
        if (len > 4 && std::memcmp(name + len - 4, ".csv", 4) == 0) {
            files.push_back(dir_path + "/" + name);
        }
    }
    closedir(dir);
    std::sort(files.begin(), files.end());
    return files;
}

bool read_all_file(const std::string& path, std::vector<char>& out) {
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) return false;
    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size < 0) {
        ::close(fd);
        return false;
    }
    size_t sz = static_cast<size_t>(st.st_size);
    out.resize(sz);
    size_t off = 0;
    while (off < sz) {
        ssize_t r = ::pread(fd, out.data() + off, sz - off, static_cast<off_t>(off));
        if (r <= 0) {
            ::close(fd);
            return false;
        }
        off += static_cast<size_t>(r);
    }
    ::close(fd);
    return true;
}

// ============================================================================
// 256-Shard Exotic Interner + Zero-Lock Scorer-v5 33-Byte Hex Registry!
// - 87.5% short IDs: raw 64-bit inline key (ZERO locks, ZERO table)
// - 12.5% 33-byte hex IDs:
//     * event_id: exact 128-bit in dedicated LongEv32 buckets (ZERO locks, ZERO table!)
//     * user_id: top-55-bit order-preserving key + L1-cached per-thread collector +
//                parallel 1ms verify_and_build (ZERO locks in Stage 1!)
//     * click_id: top-55-bit order-preserving key (ZERO storage, ZERO locks, collision-guarded!)
// - <0.01% exotic IDs (or EXACT_LONG=1 fallback): 256-shard spin-striped arena interner!
// ============================================================================

struct U128 {
    uint64_t hi;
    uint64_t lo;
    bool operator<(const U128& o) const {
        return hi != o.hi ? hi < o.hi : lo < o.lo;
    }
    bool operator==(const U128& o) const {
        return hi == o.hi && lo == o.lo;
    }
};

class ShardedExoticInterner {
    static constexpr size_t NUM_SHARDS = 256;
    struct Slot {
        uint64_t hash = 0;
        uint32_t idx = 0;
    };
    struct alignas(64) Shard {
        std::atomic_flag lock = ATOMIC_FLAG_INIT;
        std::vector<std::string_view> strs;
        std::vector<Slot> slots;
        size_t mask = 0;
        char* arena = nullptr;
        size_t arena_rem = 0;

        void grow() {
            size_t ncap = slots.empty() ? 32 : slots.size() * 2;
            std::vector<Slot> ns(ncap);
            size_t nmask = ncap - 1;
            for (const auto& s : slots) {
                if (!s.hash) continue;
                size_t i = static_cast<size_t>(s.hash >> 8) & nmask;
                while (ns[i].hash) i = (i + 1) & nmask;
                ns[i] = s;
            }
            slots.swap(ns);
            mask = nmask;
        }

        char* alloc_bytes(size_t len) {
            if (__builtin_expect(arena_rem < len, 0)) {
                size_t blk = std::max<size_t>(len, 65536);
                arena = static_cast<char*>(std::malloc(blk));
                arena_rem = blk;
            }
            char* p = arena;
            arena += len;
            arena_rem -= len;
            return p;
        }
    };
    Shard shards_[NUM_SHARDS];

public:
    __attribute__((noinline))
    uint64_t intern(const char* p, size_t len) {
        uint64_t h = fast::hash_bytes(p, len) | 1ULL;
        size_t sh = static_cast<size_t>(h & 255ULL);
        auto& s = shards_[sh];
        while (s.lock.test_and_set(std::memory_order_acquire)) {
#if defined(__x86_64__) || defined(_M_X64)
            __builtin_ia32_pause();
#elif defined(__aarch64__)
            asm volatile("yield" ::: "memory");
#endif
        }
        if (__builtin_expect((s.strs.size() + 1) * 10 >= s.slots.size() * 6, 0)) s.grow();
        size_t i = static_cast<size_t>(h >> 8) & s.mask;
        while (s.slots[i].hash) {
            if (s.slots[i].hash == h) {
                std::string_view sv = s.strs[s.slots[i].idx];
                if (sv.size() == len && std::memcmp(sv.data(), p, len) == 0) {
                    uint32_t idx = s.slots[i].idx;
                    s.lock.clear(std::memory_order_release);
                    return (static_cast<uint64_t>(sh) << 32) | (static_cast<uint64_t>(idx + 1) << 8);
                }
            }
            i = (i + 1) & s.mask;
        }
        char* copy = s.alloc_bytes(len);
        if (len) std::memcpy(copy, p, len);
        uint32_t idx = static_cast<uint32_t>(s.strs.size());
        s.strs.emplace_back(copy, len);
        s.slots[i] = Slot{h, idx};
        s.lock.clear(std::memory_order_release);
        return (static_cast<uint64_t>(sh) << 32) | (static_cast<uint64_t>(idx + 1) << 8);
    }

    inline std::string_view get_unlocked(uint64_t key) const {
        size_t sh = static_cast<size_t>((key >> 32) & 255ULL);
        size_t idx = static_cast<size_t>(((key >> 8) & 0xFFFFFFULL) - 1ULL);
        return shards_[sh].strs[idx];
    }
};

// Per-thread lazy L2-cached collector for 33-byte hex user_ids (single 16B cache line per probe!).
struct LongUserCollector {
    static constexpr size_t CACHE_SZ = 16384; // 256 KB (fits in Cascade Lake 1MB L2!)
    U128* cache = nullptr;
    std::vector<U128> vals;

    LongUserCollector() = default;
    LongUserCollector(const LongUserCollector&) = delete;
    LongUserCollector& operator=(const LongUserCollector&) = delete;
    LongUserCollector(LongUserCollector&& o) noexcept : cache(o.cache), vals(std::move(o.vals)) { o.cache = nullptr; }
    LongUserCollector& operator=(LongUserCollector&& o) noexcept {
        if (this != &o) { std::free(cache); cache = o.cache; o.cache = nullptr; vals = std::move(o.vals); }
        return *this;
    }
    ~LongUserCollector() { std::free(cache); }

    __attribute__((noinline))
    void init_cache() {
        void* p = nullptr;
        if (posix_memalign(&p, 64, CACHE_SZ * sizeof(U128)) != 0) p = std::malloc(CACHE_SZ * sizeof(U128));
        cache = static_cast<U128*>(p);
        std::memset(cache, 0xFF, CACHE_SZ * sizeof(U128));
        vals.reserve(8192);
    }

    FAST_ALWAYS_INLINE void record(uint64_t hi, uint64_t lo) {
        if (__builtin_expect(cache == nullptr, 0)) init_cache();
        size_t slot = static_cast<size_t>(fast::hash_u64(hi ^ lo)) & (CACHE_SZ - 1);
        U128& c = cache[slot];
        if (__builtin_expect(c.hi == hi && c.lo == lo, 1)) return;
        c.hi = hi;
        c.lo = lo;
        vals.push_back(U128{hi, lo});
    }
};

struct LongUserTable {
    U128* data_ = nullptr;
    bool data_is_huge_ = false;
    uint32_t bucket_lo[256]{};
    uint32_t bucket_hi[256]{};
    bool verified = false;
    bool exact_mode = false;
    mutable std::atomic<bool> need_exact{false};

    ~LongUserTable() {
        if (!data_is_huge_) std::free(data_);
    }

    bool verify_and_build(std::vector<LongUserCollector>& collectors, ThreadTeam& team, size_t num_threads) {
        if (exact_mode) {
            verified = true;
            return true;
        }
        if (need_exact.load(std::memory_order_relaxed)) return false;

        size_t total = 0;
        for (const auto& c : collectors) total += c.vals.size();
        if (total == 0) {
            verified = true;
            return true;
        }

        // Parallel 256-way flat radix scatter + in-place sort/unique (ZERO intermediate vectors, ZERO final copy!)
        std::vector<uint32_t> counts(num_threads * 256, 0);
        team.run([&](size_t tid) {
            if (tid >= num_threads) return;
            uint32_t* cnt = counts.data() + tid * 256;
            const auto& v = collectors[tid].vals;
            for (size_t i = 0; i < v.size(); ++i) cnt[v[i].hi >> 56]++;
        });

        std::vector<uint32_t> t_off(num_threads * 256, 0);
        uint32_t raw_end[256]{};
        uint32_t cur = 0;
        for (size_t b = 0; b < 256; ++b) {
            bucket_lo[b] = cur;
            for (size_t t = 0; t < num_threads; ++t) {
                t_off[t * 256 + b] = cur;
                cur += counts[t * 256 + b];
            }
            raw_end[b] = cur;
        }

        if (!data_is_huge_) std::free(data_);
        size_t bytes = total * sizeof(U128);
        if (bytes >= 2 * 1024 * 1024) {
            data_ = static_cast<U128*>(alloc_huge_raw(bytes));
            data_is_huge_ = true;
        } else {
            data_ = static_cast<U128*>(std::malloc(bytes));
            data_is_huge_ = false;
        }

        team.run([&](size_t tid) {
            if (tid >= num_threads) return;
            uint32_t* off = t_off.data() + tid * 256;
            U128* dst = data_;
            const auto& v = collectors[tid].vals;
            for (size_t i = 0; i < v.size(); ++i) {
                dst[off[v[i].hi >> 56]++] = v[i];
            }
        });

        std::atomic<bool> ok{true};
        std::atomic<size_t> next_b{0};
        team.run([&](size_t tid) {
            if (tid >= num_threads) return;
            while (true) {
                size_t b = next_b.fetch_add(1, std::memory_order_relaxed);
                if (b >= 256) break;
                uint32_t lo = bucket_lo[b];
                uint32_t hi = raw_end[b];
                if (hi - lo <= 1) {
                    bucket_hi[b] = hi;
                    continue;
                }
                U128* beg = data_ + lo;
                U128* end = data_ + hi;
                std::sort(beg, end);
                U128* w = beg + 1;
                for (U128* r = beg + 1; r < end; ++r) {
                    if (r->hi == w[-1].hi && r->lo == w[-1].lo) continue;
                    if (__builtin_expect((r->hi >> 9) == (w[-1].hi >> 9), 0)) {
                        ok.store(false, std::memory_order_relaxed);
                    }
                    *w++ = *r;
                }
                bucket_hi[b] = static_cast<uint32_t>(w - data_);
            }
        });

        if (!ok.load(std::memory_order_relaxed)) return false;
        verified = true;
        return !need_exact.load(std::memory_order_relaxed);
    }

    inline U128 lookup(uint64_t hex_key) const {
        uint64_t top55 = (hex_key & ~fast::HEX_KEY_TAG) >> 8;
        uint32_t b = static_cast<uint32_t>(top55 >> 47);
        uint32_t lo = bucket_lo[b];
        uint32_t hi = bucket_hi[b];
        const U128* arr = data_;
        while (lo < hi) {
            uint32_t mid = lo + ((hi - lo) >> 1);
            uint64_t mtop = arr[mid].hi >> 9;
            if (mtop < top55) lo = mid + 1;
            else hi = mid;
        }
        return arr[lo];
    }
};

struct KeyEnv {
    ShardedExoticInterner exotic;
    LongUserTable long_users;

    FAST_ALWAYS_INLINE uint64_t encode_user(const char* p, size_t len, const char* buf_end, LongUserCollector& col) {
        uint64_t k = fast::try_pack_raw(p, len, 'u', buf_end);
        if (__builtin_expect(k != 0, 1)) return k;
        if (__builtin_expect(!long_users.exact_mode && len == 33 && p[0] == 'u', 1)) {
            uint64_t hi, lo;
            if (__builtin_expect(fast::parse_hex128(p + 1, hi, lo), 1)) {
                col.record(hi, lo);
                return fast::make_hex_key(hi);
            }
        }
        return exotic.intern(p, len);
    }

    FAST_ALWAYS_INLINE uint64_t encode_click(const char* p, size_t len, const char* buf_end) {
        uint64_t k = fast::try_pack_raw(p, len, 'k', buf_end);
        if (__builtin_expect(k != 0, 1)) return k;
        if (__builtin_expect(!long_users.exact_mode && len == 33 && p[0] == 'k', 1)) {
            uint64_t hi, lo;
            if (__builtin_expect(fast::parse_hex128(p + 1, hi, lo), 1)) {
                return fast::make_hex_key(hi);
            }
        }
        return exotic.intern(p, len);
    }

    inline std::string_view key_to_sv(uint64_t key, char pfx, char (&buf)[40]) const {
        if (__builtin_expect(fast::is_raw_key(key), 1)) {
            size_t n = fast::unpack_raw(key, pfx, buf);
            return std::string_view(buf, n);
        }
        if (__builtin_expect(fast::is_hex_key(key), 1)) {
            U128 v = long_users.lookup(key);
            fast::format_hex128(pfx, v.hi, v.lo, buf);
            return std::string_view(buf, 33);
        }
        return exotic.get_unlocked(key);
    }

    __attribute__((noinline))
    bool key_less_slow(uint64_t a, uint64_t b, char pfx) const {
        bool a_hex = fast::is_hex_key(a);
        bool b_hex = fast::is_hex_key(b);
        if (a_hex && b_hex) {
            // Fast mode: distinct top 55 bits decide order; equal keys are identical (checked by verify_and_build!).
            return a < b;
        }
        if ((a_hex || b_hex) && !long_users.verified) {
            // One side is an unverified fast hex key, the other is raw or exotic:
            // compare other against the known 14-byte prefix (pfx + 13 hex digits)!
            uint64_t hkey = a_hex ? a : b;
            uint64_t okey = a_hex ? b : a;
            char obuf[40], hbuf[16];
            std::string_view osv = fast::is_raw_key(okey)
                                       ? std::string_view(obuf, fast::unpack_raw(okey, pfx, obuf))
                                       : exotic.get_unlocked(okey);
            fast::format_hex_prefix14(pfx, hkey, hbuf);
            size_t m = std::min<size_t>(osv.size(), 14);
            int c = m ? std::memcmp(osv.data(), hbuf, m) : 0;
            int cmp_o_vs_h;
            if (c != 0) {
                cmp_o_vs_h = c;
            } else if (osv.size() <= 14) {
                // Proper prefix of the 33-byte hex string!
                cmp_o_vs_h = -1;
            } else {
                long_users.need_exact.store(true, std::memory_order_relaxed);
                cmp_o_vs_h = 0;
            }
            return a_hex ? (cmp_o_vs_h > 0) : (cmp_o_vs_h < 0);
        }
        char ba[40], bb[40];
        auto sa = key_to_sv(a, pfx, ba);
        auto sb = key_to_sv(b, pfx, bb);
        return fast::cmp_bytes(sa.data(), sa.size(), sb.data(), sb.size()) < 0;
    }

    FAST_ALWAYS_INLINE bool key_less(uint64_t a, uint64_t b, char pfx) const {
        if (__builtin_expect(fast::is_raw_key(a) && fast::is_raw_key(b), 1)) {
            return __builtin_bswap64(a) < __builtin_bswap64(b);
        }
        if (a == b) return false;
        return key_less_slow(a, b, pfx);
    }
};

// ============================================================================
// Input File Handling (mmap + MADV_SEQUENTIAL + interior MADV_DONTNEED / pread)
// ============================================================================

struct InputFile {
    int fd = -1;
    mutable std::atomic<const char*> mapped{nullptr};
    mutable std::atomic<int> map_state{0};
    size_t size = 0;
    size_t header_end = 0;

    bool open(const std::string& path, bool use_mmap) {
        fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) return false;
        struct stat st;
        if (fstat(fd, &st) != 0 || st.st_size <= 0) {
            ::close(fd);
            fd = -1;
            map_state.store(2, std::memory_order_relaxed);
            return st.st_size == 0;
        }
        size = static_cast<size_t>(st.st_size);
        if (use_mmap) {
            header_end = 0;
            map_state.store(0, std::memory_order_relaxed);
            return true;
        }
        map_state.store(2, std::memory_order_relaxed);
#if defined(POSIX_FADV_SEQUENTIAL)
        posix_fadvise(fd, 0, 0, POSIX_FADV_SEQUENTIAL);
#endif
        char hbuf[4096];
        size_t pos = 0;
        header_end = size;
        while (pos < size) {
            size_t to_read = std::min(sizeof(hbuf), size - pos);
            ssize_t r = ::pread(fd, hbuf, to_read, static_cast<off_t>(pos));
            if (r <= 0) break;
            const char* nl = static_cast<const char*>(std::memchr(hbuf, '\n', static_cast<size_t>(r)));
            if (nl) {
                header_end = pos + static_cast<size_t>(nl - hbuf) + 1;
                break;
            }
            pos += static_cast<size_t>(r);
        }
        return true;
    }

    inline const char* ensure_mapped() const {
        if (__builtin_expect(map_state.load(std::memory_order_acquire) == 2, 1)) {
            return mapped.load(std::memory_order_relaxed);
        }
        int expected = 0;
        if (map_state.compare_exchange_strong(expected, 1, std::memory_order_acq_rel)) {
            void* p = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
            if (p != MAP_FAILED) {
#if defined(MADV_SEQUENTIAL)
                if (size >= 262144) madvise(p, size, MADV_SEQUENTIAL);
#endif
                mapped.store(static_cast<const char*>(p), std::memory_order_relaxed);
            }
            map_state.store(2, std::memory_order_release);
        } else {
            while (map_state.load(std::memory_order_acquire) != 2) {
                std::this_thread::yield();
            }
        }
        return mapped.load(std::memory_order_relaxed);
    }

    void close() {
        const char* p = mapped.load(std::memory_order_relaxed);
        if (p) {
            munmap(const_cast<char*>(p), size);
            mapped.store(nullptr, std::memory_order_relaxed);
        }
        if (fd >= 0) {
            ::close(fd);
            fd = -1;
        }
        size = 0;
        header_end = 0;
        map_state.store(2, std::memory_order_relaxed);
    }

    ~InputFile() { close(); }
};

struct ReadChunk {
    size_t file_idx;
    size_t start;
    size_t end;
    bool is_first;
};

std::vector<ReadChunk> build_read_chunks(const InputFile* files, size_t nfiles, size_t chunk_sz) {
    std::vector<ReadChunk> chunks;
    for (size_t fi = 0; fi < nfiles; ++fi) {
        const auto& f = files[fi];
        if (f.fd < 0 || f.header_end >= f.size) continue;
        size_t pos = f.header_end;
        bool first = true;
        while (pos < f.size) {
            size_t nxt = std::min(f.size, pos + chunk_sz);
            chunks.push_back(ReadChunk{fi, pos, nxt, first});
            pos = nxt;
            first = false;
        }
    }
    return chunks;
}

static inline void release_mmap_slice(const char* mp, std::string_view sv) {
#if defined(__linux__) && defined(MADV_DONTNEED)
    if (!mp || sv.size() < 262144) return;
    uintptr_t lo = (reinterpret_cast<uintptr_t>(sv.data()) + 4095ULL) & ~4095ULL;
    uintptr_t hi = reinterpret_cast<uintptr_t>(sv.data() + sv.size()) & ~4095ULL;
    if (hi > lo) madvise(reinterpret_cast<void*>(lo), hi - lo, MADV_DONTNEED);
#else
    (void)mp; (void)sv;
#endif
}

std::string_view get_aligned_chunk(const InputFile& f, const ReadChunk& ch, std::vector<char>& buf,
                                   bool use_populate = false) {
    const char* mp = f.ensure_mapped();
    if (mp) {
#if defined(__linux__) && defined(MADV_POPULATE_READ)
        if (use_populate && ch.end - ch.start >= 262144) {
            size_t lo = ch.start & ~size_t(4095);
            size_t hi = (std::min<size_t>(f.size, ch.end + 4096) + 4095) & ~size_t(4095);
            if (hi > lo) madvise(const_cast<char*>(mp + lo), hi - lo, MADV_POPULATE_READ);
        }
#else
        (void)use_populate;
#endif
        size_t s = ch.start;
        if (ch.is_first) {
            const char* nl = static_cast<const char*>(std::memchr(mp, '\n', f.size));
            if (!nl) return {};
            s = static_cast<size_t>(nl - mp) + 1;
        } else {
            const char* nl = static_cast<const char*>(std::memchr(mp + ch.start - 1, '\n', f.size - (ch.start - 1)));
            if (!nl) return {};
            s = static_cast<size_t>(nl - mp) + 1;
        }
        size_t e = f.size;
        if (ch.end < f.size) {
            const char* nl = static_cast<const char*>(std::memchr(mp + ch.end - 1, '\n', f.size - (ch.end - 1)));
            e = nl ? static_cast<size_t>(nl - mp) + 1 : f.size;
        }
        if (s >= e) return {};
        return std::string_view(mp + s, e - s);
    }

    size_t read_from = ch.is_first ? ch.start : (ch.start - 1);
    size_t extra_tail = (ch.end < f.size) ? std::min<size_t>(4096, f.size - ch.end) : 0;
    size_t want = (ch.end - read_from) + extra_tail;
    if (buf.size() < want) buf.resize(want);

    size_t got = 0;
    while (got < want) {
        ssize_t r = ::pread(f.fd, buf.data() + got, want - got, static_cast<off_t>(read_from + got));
        if (r <= 0) break;
        got += static_cast<size_t>(r);
    }

    size_t slice_start = 0;
    if (!ch.is_first) {
        const char* nl = static_cast<const char*>(std::memchr(buf.data(), '\n', got));
        if (!nl) return {};
        slice_start = static_cast<size_t>(nl - buf.data()) + 1;
    }

    size_t slice_end = got;
    if (ch.end < f.size) {
        size_t search_off = (ch.end - 1) - read_from;
        if (search_off >= got) return {};
        const char* nl = static_cast<const char*>(std::memchr(buf.data() + search_off, '\n', got - search_off));
        if (nl) {
            slice_end = static_cast<size_t>(nl - buf.data()) + 1;
        } else {
            size_t file_pos = read_from + got;
            while (file_pos < f.size) {
                size_t old_sz = got;
                size_t add = std::min<size_t>(16384, f.size - file_pos);
                if (buf.size() < old_sz + add) buf.resize(old_sz + add);
                ssize_t r = ::pread(f.fd, buf.data() + old_sz, add, static_cast<off_t>(file_pos));
                if (r <= 0) break;
                got += static_cast<size_t>(r);
                file_pos += static_cast<size_t>(r);
                const char* n2 = static_cast<const char*>(std::memchr(buf.data() + old_sz, '\n', static_cast<size_t>(r)));
                if (n2) {
                    slice_end = static_cast<size_t>(n2 - buf.data()) + 1;
                    break;
                }
                slice_end = got;
            }
        }
    }

    if (slice_start >= slice_end) return {};
    return std::string_view(buf.data() + slice_start, slice_end - slice_start);
}

class StringMap {
    struct Slot {
        uint64_t hash = 0;
        uint32_t off = 0;
        uint32_t len = 0;
        uint32_t val = 0;
    };
    std::vector<char> arena_;
    std::vector<Slot> slots_;
    size_t mask_ = 0;
    size_t count_ = 0;

    void grow(size_t new_cap) {
        std::vector<Slot> ns(new_cap);
        size_t nmask = new_cap - 1;
        for (const auto& s : slots_) {
            if (!s.hash) continue;
            size_t idx = static_cast<size_t>(s.hash) & nmask;
            while (ns[idx].hash) idx = (idx + 1) & nmask;
            ns[idx] = s;
        }
        slots_.swap(ns);
        mask_ = nmask;
    }

public:
    void reserve(size_t expected) {
        size_t cap = next_pow2(std::max<size_t>(16, expected * 2));
        slots_.assign(cap, Slot{});
        mask_ = cap - 1;
        arena_.reserve(expected * 12);
    }

    void insert(std::string_view key, uint32_t val) {
        if ((count_ + 1) * 10 >= slots_.size() * 6) {
            grow(slots_.empty() ? 16 : slots_.size() * 2);
        }
        uint64_t h = fast::hash_sv(key) | 1ULL;
        size_t idx = static_cast<size_t>(h) & mask_;
        while (slots_[idx].hash) {
            if (slots_[idx].hash == h && slots_[idx].len == key.size() &&
                std::memcmp(arena_.data() + slots_[idx].off, key.data(), key.size()) == 0) {
                slots_[idx].val = val;
                return;
            }
            idx = (idx + 1) & mask_;
        }
        uint32_t off = static_cast<uint32_t>(arena_.size());
        arena_.insert(arena_.end(), key.begin(), key.end());
        slots_[idx] = Slot{h, off, static_cast<uint32_t>(key.size()), val};
        count_++;
    }

    bool find(std::string_view key, uint32_t& out_val) const {
        if (count_ == 0) return false;
        uint64_t h = fast::hash_sv(key) | 1ULL;
        size_t idx = static_cast<size_t>(h) & mask_;
        while (slots_[idx].hash) {
            if (slots_[idx].hash == h && slots_[idx].len == key.size() &&
                std::memcmp(arena_.data() + slots_[idx].off, key.data(), key.size()) == 0) {
                out_val = slots_[idx].val;
                return true;
            }
            idx = (idx + 1) & mask_;
        }
        return false;
    }

    template <typename Fn>
    void remap_values(Fn&& fn) {
        for (auto& s : slots_) {
            if (s.hash) s.val = fn(s.val);
        }
    }
};

// Batch-64 Compacted Bloom -> Slot IdentityMap64!
class IdentityMap64 {
    struct Slot {
        uint64_t src;
        uint64_t dst;
    };
    static_assert(sizeof(Slot) == 16);
    Slot* slots_ = nullptr;
    uint64_t* bloom_ = nullptr;
    size_t cap_ = 0;
    size_t mask_ = 0;
    size_t bloom_words_ = 0;
    size_t bloom_mask_ = 0;
    size_t count_ = 0;

public:
    struct Pair {
        uint64_t src;
        uint64_t dst;
    };

    bool empty() const { return count_ == 0; }

    void build_single_from_pairs(const Pair* pairs, size_t mapped) {
        if (mapped == 0) return;

        bloom_words_ = next_pow2(std::max<size_t>(64, (mapped * 10) / 64));
        if (bloom_words_ > (1U << 19)) bloom_words_ = (1U << 19);
        bloom_mask_ = bloom_words_ - 1;
        bloom_ = static_cast<uint64_t*>(std::malloc(bloom_words_ * sizeof(uint64_t)));
        std::memset(bloom_, 0, bloom_words_ * sizeof(uint64_t));

        cap_ = next_pow2(std::max<size_t>(16, mapped * 5 / 2));
        mask_ = cap_ - 1;
        slots_ = static_cast<Slot*>(alloc_huge_raw(cap_ * sizeof(Slot)));
        std::memset(slots_, 0, cap_ * sizeof(Slot));
        count_ = mapped;

        for (size_t i = 0; i < mapped; ++i) {
            if (__builtin_expect(i + 16 < mapped, 1)) {
                uint64_t nh = fast::hash_u64(pairs[i + 16].src);
                __builtin_prefetch(&bloom_[(nh >> 18) & bloom_mask_], 1, 1);
                __builtin_prefetch(&slots_[static_cast<size_t>(nh) & mask_], 1, 1);
            }
            uint64_t src = pairs[i].src;
            uint64_t dst = pairs[i].dst;
            uint64_t h = fast::hash_u64(src);
            uint64_t bm = (1ULL << (h & 63)) | (1ULL << ((h >> 6) & 63)) |
                          (1ULL << ((h >> 12) & 63));
            bloom_[(h >> 18) & bloom_mask_] |= bm;

            size_t idx = static_cast<size_t>(h) & mask_;
            while (slots_[idx].src) idx = (idx + 1) & mask_;
            slots_[idx] = Slot{src, dst};
        }
    }

    void build_single_from_parent(const uint64_t* keys, const uint32_t* parent, size_t n, size_t mapped) {
        if (mapped == 0) return;

        bloom_words_ = next_pow2(std::max<size_t>(64, (mapped * 10) / 64));
        if (bloom_words_ > (1U << 19)) bloom_words_ = (1U << 19);
        bloom_mask_ = bloom_words_ - 1;
        bloom_ = static_cast<uint64_t*>(std::malloc(bloom_words_ * sizeof(uint64_t)));
        std::memset(bloom_, 0, bloom_words_ * sizeof(uint64_t));

        cap_ = next_pow2(std::max<size_t>(16, mapped * 5 / 2));
        mask_ = cap_ - 1;
        slots_ = static_cast<Slot*>(alloc_huge_raw(cap_ * sizeof(Slot)));
        std::memset(slots_, 0, cap_ * sizeof(Slot));
        count_ = mapped;

        for (size_t i = 0; i < n; ++i) {
            if (__builtin_expect(i + 16 < n, 1)) {
                if (parent[i + 16] != static_cast<uint32_t>(i + 16)) {
                    uint64_t nh = fast::hash_u64(keys[i + 16]);
                    __builtin_prefetch(&slots_[static_cast<size_t>(nh) & mask_], 1, 1);
                }
            }
            uint32_t r = parent[i];
            if (r == static_cast<uint32_t>(i)) continue;
            uint64_t src = keys[i];
            uint64_t dst = keys[r];
            uint64_t h = fast::hash_u64(src);
            uint64_t bm = (1ULL << (h & 63)) | (1ULL << ((h >> 6) & 63)) |
                          (1ULL << ((h >> 12) & 63));
            bloom_[(h >> 18) & bloom_mask_] |= bm;

            size_t idx = static_cast<size_t>(h) & mask_;
            while (slots_[idx].src) idx = (idx + 1) & mask_;
            slots_[idx] = Slot{src, dst};
        }
    }

    void build_from_parent(const uint64_t* keys, const uint32_t* parent, size_t n, size_t mapped,
                           ThreadTeam& team, size_t num_threads) {
        if (mapped == 0) return;
        if (num_threads <= 1) {
            build_single_from_parent(keys, parent, n, mapped);
            return;
        }

        bloom_words_ = next_pow2(std::max<size_t>(64, (mapped * 10) / 64));
        if (bloom_words_ > (1U << 19)) bloom_words_ = (1U << 19);
        bloom_mask_ = bloom_words_ - 1;
        bloom_ = static_cast<uint64_t*>(alloc_huge_raw(bloom_words_ * sizeof(uint64_t)));

        cap_ = next_pow2(std::max<size_t>(16, mapped * 5 / 2));
        mask_ = cap_ - 1;
        slots_ = static_cast<Slot*>(alloc_huge_raw(cap_ * sizeof(Slot)));
        count_ = mapped;

        team.run([&](size_t tid) {
            if (tid >= num_threads) return;
            size_t bs = (bloom_words_ * tid) / num_threads;
            size_t be = (bloom_words_ * (tid + 1)) / num_threads;
            if (be > bs) std::memset(bloom_ + bs, 0, (be - bs) * sizeof(uint64_t));
            size_t ss = (cap_ * tid) / num_threads;
            size_t se = (cap_ * (tid + 1)) / num_threads;
            if (se > ss) std::memset(slots_ + ss, 0, (se - ss) * sizeof(Slot));
        });

        team.run([&](size_t tid) {
            if (tid >= num_threads) return;
            size_t s = (n * tid) / num_threads;
            size_t e = (n * (tid + 1)) / num_threads;
            auto* abloom = reinterpret_cast<std::atomic<uint64_t>*>(bloom_);
            for (size_t i = s; i < e; ++i) {
                if (__builtin_expect(i + 16 < e, 1)) {
                    if (parent[i + 16] != static_cast<uint32_t>(i + 16)) {
                        uint64_t nh = fast::hash_u64(keys[i + 16]);
                        __builtin_prefetch(&slots_[static_cast<size_t>(nh) & mask_], 1, 1);
                    }
                }
                uint32_t r = parent[i];
                if (r == static_cast<uint32_t>(i)) continue;
                uint64_t src = keys[i];
                uint64_t dst = keys[r];
                uint64_t h = fast::hash_u64(src);
                uint64_t bm = (1ULL << (h & 63)) | (1ULL << ((h >> 6) & 63)) |
                              (1ULL << ((h >> 12) & 63));
                abloom[(h >> 18) & bloom_mask_].fetch_or(bm, std::memory_order_relaxed);

                size_t idx = static_cast<size_t>(h) & mask_;
                while (true) {
                    auto* slot_src = reinterpret_cast<std::atomic<uint64_t>*>(&slots_[idx].src);
                    uint64_t exp = 0;
                    if (slot_src->compare_exchange_weak(exp, src, std::memory_order_relaxed, std::memory_order_relaxed)) {
                        slots_[idx].dst = dst;
                        break;
                    }
                    idx = (idx + 1) & mask_;
                }
            }
        });
    }

    inline void resolve_batch(uint64_t* uids, uint64_t* hashes, size_t n) const {
        if (__builtin_expect(count_ == 0, 0)) {
            for (size_t i = 0; i < n; ++i) hashes[i] = fast::hash_u64(uids[i]);
            return;
        }
        uint32_t hit_idx[64];
        for (size_t i = 0; i < n; ++i) {
            uint64_t h = fast::hash_u64(uids[i]);
            hashes[i] = h;
            __builtin_prefetch(&bloom_[(h >> 18) & bloom_mask_], 0, 1);
        }
        size_t nhits = 0;
        for (size_t i = 0; i < n; ++i) {
            uint64_t h = hashes[i];
            uint64_t bm = (1ULL << (h & 63)) | (1ULL << ((h >> 6) & 63)) |
                          (1ULL << ((h >> 12) & 63));
            uint32_t hit = static_cast<uint32_t>((bloom_[(h >> 18) & bloom_mask_] & bm) == bm);
            hit_idx[nhits] = static_cast<uint32_t>(i);
            nhits += hit;
        }
        for (size_t k = 0; k < nhits; ++k) {
            __builtin_prefetch(&slots_[static_cast<size_t>(hashes[hit_idx[k]]) & mask_], 0, 1);
        }
        for (size_t k = 0; k < nhits; ++k) {
            uint32_t i = hit_idx[k];
            uint64_t u = uids[i];
            size_t idx = static_cast<size_t>(hashes[i]) & mask_;
            while (slots_[idx].src) {
                if (slots_[idx].src == u) {
                    uint64_t dst = slots_[idx].dst;
                    uids[i] = dst;
                    hashes[i] = fast::hash_u64(dst);
                    break;
                }
                idx = (idx + 1) & mask_;
            }
        }
    }

    inline uint64_t resolve(uint64_t uid, uint64_t& out_hash) const {
        uint64_t h = fast::hash_u64(uid);
        if (__builtin_expect(count_ == 0, 0)) {
            out_hash = h;
            return uid;
        }
        uint64_t bm = (1ULL << (h & 63)) | (1ULL << ((h >> 6) & 63)) |
                      (1ULL << ((h >> 12) & 63));
        if (__builtin_expect((bloom_[(h >> 18) & bloom_mask_] & bm) != bm, 1)) {
            out_hash = h;
            return uid;
        }
        size_t idx = static_cast<size_t>(h) & mask_;
        while (slots_[idx].src) {
            if (slots_[idx].src == uid) {
                uint64_t dst = slots_[idx].dst;
                out_hash = fast::hash_u64(dst);
                return dst;
            }
            idx = (idx + 1) & mask_;
        }
        out_hash = h;
        return uid;
    }
};

// ============================================================================
// Compact Event / Click Records & Huge-Page Slab Arenas + NT Staging Sinks
// ============================================================================

struct EvPay {
    uint64_t uid;
    uint64_t meta;
};
static_assert(sizeof(EvPay) == 16);

// Exact 128-bit event_id record for the 12.5% Scorer-v5 33-byte hex events (ZERO locks, ZERO collisions!)
struct LongEv32 {
    uint64_t hi;
    uint64_t lo;
    uint64_t uid;
    uint64_t meta;
};
static_assert(sizeof(LongEv32) == 32);

struct WideEvRow {
    uint64_t eid;     // short/exotic eid, or hi if is_long_eid
    uint64_t eid_lo;  // lo if is_long_eid, else 0
    uint64_t uid;
    int64_t ingest_sec;
    int64_t ts_ms;
    uint32_t price_cents;
    uint32_t cat_id;
    bool is_long_eid;
};

struct PersonEv16 {
    uint64_t pid;
    uint64_t ts_pc; // (ts_off << 32) | (cat_id << 24) | price_cents
};
static_assert(sizeof(PersonEv16) == 16);

struct WidePersonEv {
    uint64_t pid;
    int64_t ts_ms;
    uint32_t price_cents;
    uint32_t cat_id;
};

struct PersonClick {
    uint64_t pid;
    uint64_t cid;
    uint64_t ts_camp; // ((ts_ms - click_start) << 20) | camp_id
};
static_assert(sizeof(PersonClick) == 24);

struct SlabArena {
    std::vector<uint8_t*> chunks;
    std::vector<void*> free_list;
    uint8_t* cur = nullptr;
    size_t rem = 0;
    size_t slab_bytes = 32768;
    size_t chunk_bytes = 4 * 1024 * 1024;

    void init(size_t sb, size_t cb = 4 * 1024 * 1024) {
        slab_bytes = sb;
        chunk_bytes = std::max(cb, sb * 16);
    }

    void* alloc() {
        if (!free_list.empty()) {
            void* p = free_list.back();
            free_list.pop_back();
            return p;
        }
        if (__builtin_expect(rem < slab_bytes, 0)) {
            uint8_t* p = static_cast<uint8_t*>(alloc_huge_raw(chunk_bytes));
            chunks.push_back(p);
            cur = p;
            rem = chunk_bytes;
        }
        void* out = cur;
        cur += slab_bytes;
        rem -= slab_bytes;
        return out;
    }

    void recycle(void* p) {
        free_list.push_back(p);
    }
};

template <typename T>
struct SlabCursor {
    T* cur = nullptr;
    T* end = nullptr;
};

template <typename T>
struct SlabList {
    std::vector<T*> slabs;
    std::vector<uint32_t> counts;
    uint32_t total_count = 0;
};

template <typename T>
struct BucketSet {
    std::vector<SlabCursor<T>> cursors;
    std::vector<SlabList<T>> lists;
    uint32_t cap_per_slab = 1;

    void init(size_t nb, size_t sb) {
        cap_per_slab = static_cast<uint32_t>(std::max<size_t>(1, sb / sizeof(T)));
        cursors.assign(nb, SlabCursor<T>{nullptr, nullptr});
        lists.resize(nb);
    }

    __attribute__((noinline))
    void alloc_next(size_t b, SlabArena& arena) {
        auto& lst = lists[b];
        auto& c = cursors[b];
        if (c.cur != nullptr) {
            lst.counts.back() = cap_per_slab;
        }
        T* p = static_cast<T*>(arena.alloc());
        lst.slabs.push_back(p);
        lst.counts.push_back(0);
        c.cur = p;
        c.end = p + cap_per_slab;
    }

    FAST_ALWAYS_INLINE void push(size_t b, const T& item, SlabArena& arena) {
        auto& c = cursors[b];
        if (__builtin_expect(c.cur == c.end, 0)) {
            alloc_next(b, arena);
        }
        *c.cur++ = item;
    }

    void finish_all() {
        for (size_t b = 0; b < cursors.size(); ++b) {
            auto& c = cursors[b];
            auto& lst = lists[b];
            if (c.cur != nullptr) {
                uint32_t last_cnt = static_cast<uint32_t>(cap_per_slab - static_cast<size_t>(c.end - c.cur));
                lst.counts.back() = last_cnt;
                uint32_t tot = 0;
                for (uint32_t cnt : lst.counts) tot += cnt;
                lst.total_count = tot;
                c.cur = nullptr;
                c.end = nullptr;
            }
        }
    }

    void finish_all(SlabArena&) { finish_all(); }

    void recycle_bucket(size_t b, SlabArena& arena) {
        auto& lst = lists[b];
        for (T* p : lst.slabs) arena.recycle(p);
        lst.slabs.clear();
        lst.counts.clear();
        lst.total_count = 0;
    }
};

// Write-combining L1-staged BucketSet for 16B/32B records: flushes 64B cache lines via NT stores on x86_64!
template <typename T>
struct StagedBucketSet {
    static constexpr uint32_t WC_CAP = 64 / sizeof(T);
    static_assert(sizeof(T) * WC_CAP == 64);

    T* wc_buf = nullptr;
    std::vector<uint8_t> wc_cnt;
    std::vector<SlabCursor<T>> cursors;
    std::vector<SlabList<T>> lists;
    uint32_t cap_per_slab = WC_CAP;

    ~StagedBucketSet() {
        std::free(wc_buf);
    }

    void init(size_t nb, size_t sb) {
        cap_per_slab = static_cast<uint32_t>(std::max<size_t>(WC_CAP, ((sb / sizeof(T)) / WC_CAP) * WC_CAP));
        std::free(wc_buf);
        void* p = nullptr;
        if (posix_memalign(&p, 64, nb * 64) != 0) p = std::malloc(nb * 64);
        wc_buf = static_cast<T*>(p);
        wc_cnt.assign(nb, 0);
        cursors.assign(nb, SlabCursor<T>{nullptr, nullptr});
        lists.resize(nb);
    }

    __attribute__((noinline))
    void flush_line(size_t b, SlabArena& arena) {
        auto& c = cursors[b];
        if (__builtin_expect(c.cur == c.end, 0)) {
            auto& lst = lists[b];
            if (c.cur != nullptr) lst.counts.back() = cap_per_slab;
            T* p = static_cast<T*>(arena.alloc());
            lst.slabs.push_back(p);
            lst.counts.push_back(0);
            c.cur = p;
            c.end = p + cap_per_slab;
        }
        const T* src = wc_buf + b * WC_CAP;
#if defined(__AVX2__)
        auto* d = reinterpret_cast<__m256i*>(c.cur);
        const auto* s = reinterpret_cast<const __m256i*>(src);
        _mm256_stream_si256(d, _mm256_load_si256(s));
        _mm256_stream_si256(d + 1, _mm256_load_si256(s + 1));
#else
        std::memcpy(c.cur, src, 64);
#endif
        c.cur += WC_CAP;
        wc_cnt[b] = 0;
    }

    FAST_ALWAYS_INLINE void push(size_t b, const T& item, SlabArena& arena) {
        uint8_t n = wc_cnt[b];
        wc_buf[b * WC_CAP + n] = item;
        if (__builtin_expect(++n == WC_CAP, 0)) {
            flush_line(b, arena);
        } else {
            wc_cnt[b] = n;
        }
    }

    void finish_all(SlabArena& arena) {
        for (size_t b = 0; b < cursors.size(); ++b) {
            uint8_t rem = wc_cnt[b];
            auto& c = cursors[b];
            auto& lst = lists[b];
            if (rem > 0) {
                if (c.cur == c.end) {
                    if (c.cur != nullptr) lst.counts.back() = cap_per_slab;
                    T* p = static_cast<T*>(arena.alloc());
                    lst.slabs.push_back(p);
                    lst.counts.push_back(0);
                    c.cur = p;
                    c.end = p + cap_per_slab;
                }
                std::memcpy(c.cur, wc_buf + b * WC_CAP, static_cast<size_t>(rem) * sizeof(T));
                c.cur += rem;
                wc_cnt[b] = 0;
            }
            if (c.cur != nullptr) {
                uint32_t last_cnt = static_cast<uint32_t>(cap_per_slab - static_cast<size_t>(c.end - c.cur));
                lst.counts.back() = last_cnt;
                uint32_t tot = 0;
                for (uint32_t cnt : lst.counts) tot += cnt;
                lst.total_count = tot;
                c.cur = nullptr;
                c.end = nullptr;
            }
        }
#if defined(__AVX2__)
        _mm_sfence();
#endif
    }

    void recycle_bucket(size_t b, SlabArena& arena) {
        auto& lst = lists[b];
        for (T* p : lst.slabs) arena.recycle(p);
        lst.slabs.clear();
        lst.counts.clear();
        lst.total_count = 0;
    }
};

// Use NT-staged write-combining on x86_64 AVX2/AVX-512, direct stores on ARM64 (avoids double stores!).
#if defined(__AVX2__)
template <typename T>
using FastBucketSet = StagedBucketSet<T>;
#else
template <typename T>
using FastBucketSet = BucketSet<T>;
#endif

// SOA BucketSet for short/exotic events: eids[cap] + 16B-aligned EvPay[cap] per 32KB slab.
struct EvSlabCursor {
    uint8_t* raw = nullptr;
    uint32_t cnt = 0;
    uint32_t cap = 0;
};

struct EvSlabList {
    std::vector<uint8_t*> slabs;
    std::vector<uint32_t> counts;
    uint32_t total_count = 0;
};

struct EvBucketSet {
    std::vector<EvSlabCursor> cursors;
    std::vector<EvSlabList> lists;
    uint32_t cap_per_slab = 2;
    size_t pay_byte_off = 16;

    void init(size_t nb, size_t sb) {
        cap_per_slab = static_cast<uint32_t>(std::max<size_t>(2, (sb / 24) & ~static_cast<size_t>(1)));
        pay_byte_off = static_cast<size_t>(cap_per_slab) * sizeof(uint64_t);
        cursors.assign(nb, EvSlabCursor{nullptr, 0, 0});
        lists.resize(nb);
    }

    __attribute__((noinline))
    void alloc_next(size_t b, SlabArena& arena) {
        auto& lst = lists[b];
        auto& c = cursors[b];
        if (c.raw != nullptr) {
            lst.counts.back() = cap_per_slab;
        }
        uint8_t* p = static_cast<uint8_t*>(arena.alloc());
        lst.slabs.push_back(p);
        lst.counts.push_back(0);
        c.raw = p;
        c.cnt = 0;
        c.cap = cap_per_slab;
    }

    FAST_ALWAYS_INLINE void push(size_t b, uint64_t eid, uint64_t uid, uint64_t meta, SlabArena& arena) {
        auto& c = cursors[b];
        if (__builtin_expect(c.cnt == c.cap, 0)) {
            alloc_next(b, arena);
        }
        uint32_t idx = c.cnt++;
        reinterpret_cast<uint64_t*>(c.raw)[idx] = eid;
        reinterpret_cast<EvPay*>(c.raw + pay_byte_off)[idx] = EvPay{uid, meta};
    }

    void finish_all() {
        for (size_t b = 0; b < cursors.size(); ++b) {
            auto& c = cursors[b];
            auto& lst = lists[b];
            if (c.raw != nullptr) {
                lst.counts.back() = c.cnt;
                uint32_t tot = 0;
                for (uint32_t cnt : lst.counts) tot += cnt;
                lst.total_count = tot;
                c.raw = nullptr;
                c.cnt = 0;
                c.cap = 0;
            }
        }
    }

    void recycle_bucket(size_t b, SlabArena& arena) {
        auto& lst = lists[b];
        for (uint8_t* p : lst.slabs) arena.recycle(p);
        lst.slabs.clear();
        lst.counts.clear();
        lst.total_count = 0;
    }
};

// ============================================================================
// Manifest & Zero-Alloc Catalog Loader
// ============================================================================

bool parse_manifest(const std::string& path, int64_t& window_start, int64_t& window_end) {
    std::vector<char> buf;
    if (!read_all_file(path, buf) || buf.empty()) return false;
    std::string_view text(buf.data(), buf.size());
    fast::DateTable dtab{};
    auto extract = [&](std::string_view key, int64_t& out) -> bool {
        size_t k = text.find(key);
        if (k == std::string_view::npos) return false;
        size_t colon = text.find(':', k + key.size());
        if (colon == std::string_view::npos) return false;
        size_t q1 = text.find('"', colon + 1);
        if (q1 == std::string_view::npos) return false;
        size_t q2 = text.find('"', q1 + 1);
        if (q2 == std::string_view::npos) return false;
        return fast::parse_timestamp_ms(text.data() + q1 + 1, q2 - q1 - 1, dtab, out);
    };
    return extract("\"window_start\"", window_start) && extract("\"window_end\"", window_end);
}

struct Catalogs {
    static constexpr uint32_t MAX_FAST_PROD = 131072;
    static constexpr uint32_t MAX_FAST_CAMP = 65536;

    StringMap product_to_cat;
    std::vector<uint32_t> prod_fast;
    std::vector<std::string> categories;
    uint32_t unknown_cat_id = 0;

    StringMap campaign_to_id;
    std::vector<uint32_t> camp_fast;
    std::vector<std::string> campaign_names;
    std::vector<std::string> campaign_channels;

    FAST_ALWAYS_INLINE uint32_t lookup_product_cat(const char* p, size_t len) const {
        if (__builtin_expect(len >= 2 && p[0] == 'p', 1)) {
            uint32_t num;
            if (__builtin_expect(fast::parse_pos_u32_no_lz(p + 1, len - 1, MAX_FAST_PROD - 1, num), 1)) {
                uint32_t c = prod_fast[num];
                return (c != UINT32_MAX) ? c : unknown_cat_id;
            }
        }
        uint32_t c = unknown_cat_id;
        product_to_cat.find(std::string_view(p, len), c);
        return c;
    }

    FAST_ALWAYS_INLINE uint32_t lookup_campaign(const char* p, size_t len) const {
        if (__builtin_expect(len >= 3 && p[0] == 'c' && p[1] == '_', 1)) {
            uint32_t num;
            if (__builtin_expect(fast::parse_pos_u32_no_lz(p + 2, len - 2, MAX_FAST_CAMP - 1, num), 1)) {
                return camp_fast[num];
            }
        }
        uint32_t id = 0;
        campaign_to_id.find(std::string_view(p, len), id);
        return id;
    }
};

void load_catalogs(const std::string& data_dir, Catalogs& cat) {
    {
        std::vector<char> buf;
        std::vector<std::string> raw_cats;
        StringMap raw_cat_map;
        raw_cat_map.reserve(64);
        raw_cats.push_back("unknown");
        raw_cat_map.insert("unknown", 0);

        cat.prod_fast.assign(Catalogs::MAX_FAST_PROD, UINT32_MAX);
        uint32_t max_fast_seen = 0;

        if (read_all_file(data_dir + "/products.csv", buf) && !buf.empty()) {
            const char* nl = static_cast<const char*>(std::memchr(buf.data(), '\n', buf.size()));
            if (nl) {
                size_t start = static_cast<size_t>(nl - buf.data()) + 1;
                fast::scan_csv_chunk<2>(buf.data() + start, buf.size() - start, [&](const fast::Field (&f)[2]) {
                    if (f[0].len == 0) return;
                    std::string_view cname(f[1].ptr, f[1].len);
                    uint32_t cid = 0;
                    if (!raw_cat_map.find(cname, cid)) {
                        cid = static_cast<uint32_t>(raw_cats.size());
                        raw_cats.emplace_back(cname);
                        raw_cat_map.insert(cname, cid);
                    }
                    if (__builtin_expect(f[0].len >= 2 && f[0].ptr[0] == 'p', 1)) {
                        uint32_t num;
                        if (__builtin_expect(fast::parse_pos_u32_no_lz(f[0].ptr + 1, f[0].len - 1, Catalogs::MAX_FAST_PROD - 1, num), 1)) {
                            cat.prod_fast[num] = cid;
                            if (num > max_fast_seen) max_fast_seen = num;
                            return;
                        }
                    }
                    cat.product_to_cat.insert(std::string_view(f[0].ptr, f[0].len), cid);
                });
            }
        }

        size_t ncats = raw_cats.size();
        std::vector<uint32_t> order(ncats);
        for (uint32_t i = 0; i < ncats; ++i) order[i] = i;
        std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
            return fast::less_sv(raw_cats[a], raw_cats[b]);
        });
        std::vector<uint32_t> remap(ncats);
        cat.categories.resize(ncats);
        for (uint32_t new_id = 0; new_id < ncats; ++new_id) {
            uint32_t old_id = order[new_id];
            remap[old_id] = new_id;
            cat.categories[new_id] = std::move(raw_cats[old_id]);
        }
        cat.unknown_cat_id = remap[0];
        for (uint32_t i = 0; i <= max_fast_seen; ++i) {
            uint32_t c = cat.prod_fast[i];
            if (c != UINT32_MAX) cat.prod_fast[i] = remap[c];
        }
        cat.product_to_cat.remap_values([&](uint32_t old_id) { return remap[old_id]; });
    }

    {
        cat.campaign_names.push_back("organic");
        cat.campaign_channels.push_back("organic");
        cat.camp_fast.assign(Catalogs::MAX_FAST_CAMP, 0);
        std::vector<char> buf;
        if (read_all_file(data_dir + "/campaigns.csv", buf) && !buf.empty()) {
            const char* nl = static_cast<const char*>(std::memchr(buf.data(), '\n', buf.size()));
            if (nl) {
                size_t start = static_cast<size_t>(nl - buf.data()) + 1;
                cat.campaign_names.reserve(4096);
                cat.campaign_channels.reserve(4096);
                fast::scan_csv_chunk<2>(buf.data() + start, buf.size() - start, [&](const fast::Field (&f)[2]) {
                    if (f[0].len == 0) return;
                    uint32_t id = static_cast<uint32_t>(cat.campaign_names.size());
                    cat.campaign_names.emplace_back(f[0].ptr, f[0].len);
                    cat.campaign_channels.emplace_back(f[1].ptr, f[1].len);
                    if (__builtin_expect(f[0].len >= 3 && f[0].ptr[0] == 'c' && f[0].ptr[1] == '_', 1)) {
                        uint32_t num;
                        if (__builtin_expect(fast::parse_pos_u32_no_lz(f[0].ptr + 2, f[0].len - 2, Catalogs::MAX_FAST_CAMP - 1, num), 1)) {
                            cat.camp_fast[num] = id;
                            return;
                        }
                    }
                    cat.campaign_to_id.insert(std::string_view(f[0].ptr, f[0].len), id);
                });
            }
        }
    }
}

// ============================================================================
// Links DSU (Overlapped Single-Threaded + Parallel Fallback)
// ============================================================================

struct LinkEdge64 {
    uint64_t a;
    uint64_t b;
};

void parse_links_and_build_identity_single(const InputFile* link_files,
                                           size_t n_link_files,
                                           size_t total_link_bytes,
                                           KeyEnv& kenv,
                                           LongUserCollector& col,
                                           IdentityMap64& id_map,
                                           double& ms_parse,
                                           double& ms_union,
                                           double& ms_idmap) {
    ms_parse = ms_union = ms_idmap = 0.0;
    if (total_link_bytes == 0) return;
    auto t0 = Clock::now();
    std::vector<LinkEdge64> edges;
    edges.reserve(total_link_bytes / 20 + 64);
    std::vector<char> io_buf;
    auto chunks = build_read_chunks(link_files, n_link_files, 1 << 20);
    for (const auto& ch : chunks) {
        auto sv = get_aligned_chunk(link_files[ch.file_idx], ch, io_buf);
        if (sv.empty()) continue;
        const char* sv_end = sv.data() + sv.size();
        fast::scan_csv_chunk<2>(sv.data(), sv.size(), [&](const fast::Field (&f)[2]) __attribute__((always_inline)) {
            if (f[0].len == 0 || f[1].len == 0) return;
            if (f[0].len == f[1].len && std::memcmp(f[0].ptr, f[1].ptr, f[0].len) == 0) return;
            edges.push_back(LinkEdge64{
                kenv.encode_user(f[0].ptr, f[0].len, sv_end, col),
                kenv.encode_user(f[1].ptr, f[1].len, sv_end, col)
            });
        });
    }
    ms_parse = elapsed_sec(t0) * 1000.0;
    size_t total_edges = edges.size();
    if (total_edges == 0) return;

    // Direct-in-table 16B DsuSlot: key + parent in the SAME cache line (1 miss/node vs 4!)
    auto t1 = Clock::now();
    struct DsuSlot { uint64_t key; uint32_t parent; uint32_t pad; };
    static_assert(sizeof(DsuSlot) == 16);
    size_t cap = next_pow2(std::max<size_t>(64, total_edges * 3));
    DsuSlot* table = static_cast<DsuSlot*>(alloc_huge_raw(cap * sizeof(DsuSlot)));
    std::memset(table, 0, cap * sizeof(DsuSlot));
    size_t mask = cap - 1;
    size_t num_nodes = 0;

    auto grow_table = [&]() {
        size_t ncap = cap * 2;
        DsuSlot* nt = static_cast<DsuSlot*>(alloc_huge_raw(ncap * sizeof(DsuSlot)));
        std::memset(nt, 0, ncap * sizeof(DsuSlot));
        size_t nmask = ncap - 1;
        std::vector<uint32_t> remap(cap, 0);
        for (size_t i = 0; i < cap; ++i) {
            if (!table[i].key) continue;
            size_t idx = static_cast<size_t>(fast::hash_u64(table[i].key)) & nmask;
            while (nt[idx].key) idx = (idx + 1) & nmask;
            nt[idx] = table[i];
            remap[i] = static_cast<uint32_t>(idx);
        }
        for (size_t i = 0; i < ncap; ++i) {
            if (nt[i].key) nt[i].parent = remap[nt[i].parent];
        }
        table = nt; cap = ncap; mask = nmask;
    };

    auto intern = [&](uint64_t k) -> uint32_t {
        uint64_t h = fast::hash_u64(k);
        size_t idx = static_cast<size_t>(h) & mask;
        while (table[idx].key) {
            if (__builtin_expect(table[idx].key == k, 1)) return static_cast<uint32_t>(idx);
            idx = (idx + 1) & mask;
        }
        if (__builtin_expect((num_nodes + 1) * 10 >= cap * 7, 0)) {
            grow_table();
            idx = static_cast<size_t>(h) & mask;
            while (table[idx].key) idx = (idx + 1) & mask;
        }
        uint32_t sl = static_cast<uint32_t>(idx);
        table[sl] = DsuSlot{k, sl, 0};
        num_nodes++;
        return sl;
    };

    auto find_r = [&](uint32_t x) -> uint32_t {
        while (table[x].parent != x) {
            uint32_t p = table[x].parent;
            uint32_t gp = table[p].parent;
            table[x].parent = gp;
            x = gp;
        }
        return x;
    };

    const LinkEdge64* arr = edges.data();
    for (size_t i = 0; i < total_edges; ++i) {
        if (__builtin_expect(i + 16 < total_edges, 1)) {
            __builtin_prefetch(&table[static_cast<size_t>(fast::hash_u64(arr[i + 16].a)) & mask], 1, 1);
            __builtin_prefetch(&table[static_cast<size_t>(fast::hash_u64(arr[i + 16].b)) & mask], 1, 1);
        }
        uint64_t ka = arr[i].a;
        uint64_t kb = arr[i].b;
        if (__builtin_expect(ka == kb, 0)) continue;
        uint32_t u = find_r(intern(ka));
        uint32_t v = find_r(intern(kb));
        if (u == v) continue;
        if (kenv.key_less(table[v].key, table[u].key, 'u')) table[u].parent = v;
        else table[v].parent = u;
    }
    ms_union = elapsed_sec(t1) * 1000.0;

    auto t2 = Clock::now();
    std::vector<IdentityMap64::Pair> pairs;
    pairs.reserve(num_nodes / 2 + 16);
    for (size_t i = 0; i < cap; ++i) {
        uint64_t k = table[i].key;
        if (!k) continue;
        uint32_t r = find_r(static_cast<uint32_t>(i));
        if (r != static_cast<uint32_t>(i)) {
            pairs.push_back(IdentityMap64::Pair{k, table[r].key});
        }
    }
    id_map.build_single_from_pairs(pairs.data(), pairs.size());
    ms_idmap = elapsed_sec(t2) * 1000.0;
}

void build_identity_from_links(const std::vector<std::vector<LinkEdge64>>& bufs,
                               const KeyEnv& kenv,
                               IdentityMap64& id_map,
                               ThreadTeam& team,
                               size_t num_threads) {
    size_t total_edges = 0;
    for (const auto& b : bufs) total_edges += b.size();
    if (total_edges == 0) return;

    struct NodeSlot { uint64_t key; uint32_t node; uint32_t pad; };
    size_t est = std::max<size_t>(64, total_edges * 3 / 2);
    size_t cap = next_pow2(est * 2);
    std::vector<NodeSlot> table(cap, NodeSlot{0, 0, 0});
    size_t mask = cap - 1;
    std::vector<uint64_t> keys;
    std::vector<uint32_t> parent;
    keys.reserve(est);
    parent.reserve(est);
    auto grow_table = [&]() {
        size_t ncap = cap * 2;
        std::vector<NodeSlot> nt(ncap, NodeSlot{0, 0, 0});
        size_t nmask = ncap - 1;
        for (const auto& s : table) {
            if (!s.key) continue;
            size_t idx = static_cast<size_t>(fast::hash_u64(s.key)) & nmask;
            while (nt[idx].key) idx = (idx + 1) & nmask;
            nt[idx] = s;
        }
        table.swap(nt); cap = ncap; mask = nmask;
    };
    auto intern = [&](uint64_t k) -> uint32_t {
        if ((keys.size() + 1) * 10 >= cap * 6) grow_table();
        size_t idx = static_cast<size_t>(fast::hash_u64(k)) & mask;
        while (table[idx].key) {
            if (table[idx].key == k) return table[idx].node;
            idx = (idx + 1) & mask;
        }
        uint32_t nid = static_cast<uint32_t>(keys.size());
        keys.push_back(k); parent.push_back(nid);
        table[idx] = NodeSlot{k, nid, 0};
        return nid;
    };
    auto find_r = [&](uint32_t x) -> uint32_t {
        while (parent[x] != x) { parent[x] = parent[parent[x]]; x = parent[x]; }
        return x;
    };
    for (const auto& b : bufs) {
        for (const auto& e : b) {
            uint32_t u = find_r(intern(e.a));
            uint32_t v = find_r(intern(e.b));
            if (u == v) continue;
            if (kenv.key_less(keys[v], keys[u], 'u')) parent[u] = v;
            else parent[v] = u;
        }
    }
    size_t n = keys.size();
    size_t mapped = 0;
    for (size_t i = 0; i < n; ++i) {
        uint32_t r = find_r(static_cast<uint32_t>(i));
        parent[i] = r;
        mapped += static_cast<size_t>(r != static_cast<uint32_t>(i));
    }
    id_map.build_from_parent(keys.data(), parent.data(), n, mapped, team, num_threads);
}

// ============================================================================
// Hot/Cold CampDayAccum
// ============================================================================

struct HotCampDay {
    uint32_t sessions = 0;
    uint32_t assists = 0;
};
static_assert(sizeof(HotCampDay) == 8);

struct ColdCampDay {
    uint32_t purchasing_sessions = 0;
    uint32_t pad = 0;
    int64_t revenue_cents = 0;
};
static_assert(sizeof(ColdCampDay) == 16);

struct CampDayStats {
    int64_t sessions = 0;
    int64_t purchasing_sessions = 0;
    int64_t revenue_cents = 0;
    int64_t assisted_sessions = 0;
};

struct CampDayKey {
    uint32_t camp_id;
    int32_t day;
};

class CampDayAccum {
    uint32_t num_camps_ = 0;
    int32_t base_day_ = 0;
    int32_t num_days_ = 0;
    bool dense_ = false;
    std::vector<HotCampDay> dense_hot_;
    std::vector<ColdCampDay> dense_cold_;

    struct Slot {
        uint64_t hash = 0;
        CampDayKey key{0, 0};
        CampDayStats stats;
    };
    std::vector<Slot> map_slots_;
    size_t map_mask_ = 0;
    size_t map_count_ = 0;

    void grow_map() {
        size_t ncap = map_slots_.empty() ? 256 : map_slots_.size() * 2;
        std::vector<Slot> ns(ncap);
        size_t nmask = ncap - 1;
        for (const auto& s : map_slots_) {
            if (!s.hash) continue;
            size_t idx = static_cast<size_t>(s.hash) & nmask;
            while (ns[idx].hash) idx = (idx + 1) & nmask;
            ns[idx] = s;
        }
        map_slots_.swap(ns);
        map_mask_ = nmask;
    }

    CampDayStats& get_sparse(uint32_t camp_id, int32_t day) {
        if ((map_count_ + 1) * 10 >= map_slots_.size() * 6) grow_map();
        uint64_t k = (static_cast<uint64_t>(camp_id) << 32) | static_cast<uint32_t>(day);
        uint64_t h = fast::hash_u64(k) | 1ULL;
        size_t idx = static_cast<size_t>(h) & map_mask_;
        while (map_slots_[idx].hash) {
            if (map_slots_[idx].hash == h && map_slots_[idx].key.camp_id == camp_id &&
                map_slots_[idx].key.day == day) {
                return map_slots_[idx].stats;
            }
            idx = (idx + 1) & map_mask_;
        }
        map_slots_[idx] = Slot{h, CampDayKey{camp_id, day}, CampDayStats{}};
        map_count_++;
        return map_slots_[idx].stats;
    }

public:
    bool is_dense() const { return dense_; }
    int32_t num_days() const { return num_days_; }

    void init(uint32_t num_camps, int32_t min_day, int32_t max_day) {
        num_camps_ = num_camps;
        base_day_ = min_day;
        int64_t span = static_cast<int64_t>(max_day) - static_cast<int64_t>(min_day) + 1;
        if (span > 0 && span <= 64 && static_cast<uint64_t>(num_camps) * static_cast<uint64_t>(span) <= 262144ULL) {
            dense_ = true;
            num_days_ = static_cast<int32_t>(span);
            size_t tot = static_cast<size_t>(num_camps) * static_cast<size_t>(num_days_);
            dense_hot_.assign(tot, HotCampDay{});
            dense_cold_.assign(tot, ColdCampDay{});
        } else {
            dense_ = false;
        }
    }

    FAST_ALWAYS_INLINE void add_session_dense(uint32_t camp_id, uint32_t day_off, int64_t rev) {
        size_t idx = static_cast<size_t>(camp_id) * static_cast<size_t>(num_days_) + day_off;
        dense_hot_[idx].sessions++;
        if (__builtin_expect(rev > 0, 0)) {
            dense_cold_[idx].purchasing_sessions++;
            dense_cold_[idx].revenue_cents += rev;
        }
    }

    FAST_ALWAYS_INLINE void add_assist_dense(uint32_t camp_id, uint32_t day_off, uint32_t delta = 1) {
        dense_hot_[static_cast<size_t>(camp_id) * static_cast<size_t>(num_days_) + day_off].assists += delta;
    }

    FAST_ALWAYS_INLINE void add_session_and_assist_dense(uint32_t camp_id, uint32_t day_off, int64_t rev) {
        size_t idx = static_cast<size_t>(camp_id) * static_cast<size_t>(num_days_) + day_off;
        dense_hot_[idx].sessions++;
        dense_hot_[idx].assists++;
        if (__builtin_expect(rev > 0, 0)) {
            dense_cold_[idx].purchasing_sessions++;
            dense_cold_[idx].revenue_cents += rev;
        }
    }

    inline void add_session(uint32_t camp_id, int32_t day, int64_t rev) {
        if (__builtin_expect(dense_ && day >= base_day_ && day < base_day_ + num_days_, 1)) {
            add_session_dense(camp_id, static_cast<uint32_t>(day - base_day_), rev);
            return;
        }
        auto& st = get_sparse(camp_id, day);
        st.sessions++;
        if (rev > 0) {
            st.purchasing_sessions++;
            st.revenue_cents += rev;
        }
    }

    inline void add_assist(uint32_t camp_id, int32_t day, uint32_t delta = 1) {
        if (__builtin_expect(dense_ && day >= base_day_ && day < base_day_ + num_days_, 1)) {
            add_assist_dense(camp_id, static_cast<uint32_t>(day - base_day_), delta);
            return;
        }
        get_sparse(camp_id, day).assisted_sessions += delta;
    }

    struct MergedTable {
        bool dense = false;
        uint32_t num_camps = 0;
        int32_t base_day = 0;
        int32_t num_days = 0;
        std::vector<CampDayStats> dense_stats;
        std::vector<Slot> sparse_slots;
        size_t sparse_mask = 0;
        size_t sparse_count = 0;

        CampDayStats& get_sparse(uint32_t camp_id, int32_t day) {
            if ((sparse_count + 1) * 10 >= sparse_slots.size() * 6) {
                size_t ncap = sparse_slots.empty() ? 256 : sparse_slots.size() * 2;
                std::vector<Slot> ns(ncap);
                size_t nmask = ncap - 1;
                for (const auto& s : sparse_slots) {
                    if (!s.hash) continue;
                    size_t idx = static_cast<size_t>(s.hash) & nmask;
                    while (ns[idx].hash) idx = (idx + 1) & nmask;
                    ns[idx] = s;
                }
                sparse_slots.swap(ns);
                sparse_mask = nmask;
            }
            uint64_t k = (static_cast<uint64_t>(camp_id) << 32) | static_cast<uint32_t>(day);
            uint64_t h = fast::hash_u64(k) | 1ULL;
            size_t idx = static_cast<size_t>(h) & sparse_mask;
            while (sparse_slots[idx].hash) {
                if (sparse_slots[idx].hash == h && sparse_slots[idx].key.camp_id == camp_id &&
                    sparse_slots[idx].key.day == day) {
                    return sparse_slots[idx].stats;
                }
                idx = (idx + 1) & sparse_mask;
            }
            sparse_slots[idx] = Slot{h, CampDayKey{camp_id, day}, CampDayStats{}};
            sparse_count++;
            return sparse_slots[idx].stats;
        }

        template <typename Fn>
        void for_each_nonzero(Fn&& fn) const {
            if (dense) {
                for (uint32_t c = 0; c < num_camps; ++c) {
                    size_t base = static_cast<size_t>(c) * num_days;
                    for (int32_t d = 0; d < num_days; ++d) {
                        const auto& st = dense_stats[base + d];
                        if (st.sessions > 0 || st.assisted_sessions > 0) {
                            fn(c, base_day + d, st);
                        }
                    }
                }
            }
            for (const auto& s : sparse_slots) {
                if (s.hash && (s.stats.sessions > 0 || s.stats.assisted_sessions > 0)) {
                    fn(s.key.camp_id, s.key.day, s.stats);
                }
            }
        }
    };

    static MergedTable merge_all(const std::vector<CampDayAccum>& accs) {
        MergedTable out;
        if (accs.empty()) return out;
        out.dense = accs[0].dense_;
        out.num_camps = accs[0].num_camps_;
        out.base_day = accs[0].base_day_;
        out.num_days = accs[0].num_days_;
        if (out.dense) {
            size_t tot = static_cast<size_t>(out.num_camps) * static_cast<size_t>(out.num_days);
            out.dense_stats.assign(tot, CampDayStats{});
            for (const auto& a : accs) {
                for (size_t i = 0; i < tot; ++i) {
                    out.dense_stats[i].sessions += a.dense_hot_[i].sessions;
                    out.dense_stats[i].assisted_sessions += a.dense_hot_[i].assists;
                    out.dense_stats[i].purchasing_sessions += a.dense_cold_[i].purchasing_sessions;
                    out.dense_stats[i].revenue_cents += a.dense_cold_[i].revenue_cents;
                }
            }
        }
        for (const auto& a : accs) {
            for (const auto& s : a.map_slots_) {
                if (s.hash && (s.stats.sessions > 0 || s.stats.assisted_sessions > 0)) {
                    auto& dst = out.get_sparse(s.key.camp_id, s.key.day);
                    dst.sessions += s.stats.sessions;
                    dst.purchasing_sessions += s.stats.purchasing_sessions;
                    dst.revenue_cents += s.stats.revenue_cents;
                    dst.assisted_sessions += s.stats.assisted_sessions;
                }
            }
        }
        return out;
    }
};

// ============================================================================
// Sorting Helpers (tiny u16 bucket sort + click radix/insertion sort)
// ============================================================================

static inline void cswap16(uint16_t& a, uint16_t& b) {
    uint16_t lo = a < b ? a : b;
    uint16_t hi = a < b ? b : a;
    a = lo;
    b = hi;
}

static inline void sort_u16(uint16_t* a, size_t n) {
    if (n <= 1) return;
    if (n == 2) { cswap16(a[0], a[1]); return; }
    if (n == 3) { cswap16(a[0], a[1]); cswap16(a[1], a[2]); cswap16(a[0], a[1]); return; }
    if (n == 4) {
        cswap16(a[0], a[1]); cswap16(a[2], a[3]);
        cswap16(a[0], a[2]); cswap16(a[1], a[3]);
        cswap16(a[1], a[2]);
        return;
    }
    if (n <= 24) {
        for (size_t i = 1; i < n; ++i) {
            uint16_t k = a[i];
            size_t j = i;
            while (j > 0 && a[j - 1] > k) { a[j] = a[j - 1]; --j; }
            a[j] = k;
        }
        return;
    }
    std::sort(a, a + n);
}

struct ClickItem {
    uint64_t ts_camp;
    uint64_t cid;
};

static inline bool click_less(const ClickItem& a, const ClickItem& b, const KeyEnv& kenv) {
    uint64_t ta = a.ts_camp >> 20;
    uint64_t tb = b.ts_camp >> 20;
    if (__builtin_expect(ta != tb, 1)) return ta < tb;
    if (__builtin_expect(a.cid == b.cid, 0)) {
        if (fast::is_hex_key(a.cid) && (a.ts_camp & 0xFFFFFU) != (b.ts_camp & 0xFFFFFU)) {
            kenv.long_users.need_exact.store(true, std::memory_order_relaxed);
        }
        return false;
    }
    // Larger click_id FIRST on tie, so the LAST qualifying click at ts has the SMALLEST click_id!
    return kenv.key_less(b.cid, a.cid, 'k');
}

static const ClickItem* sort_clicks(ClickItem* a, size_t n, const KeyEnv& kenv,
                                    std::vector<ClickItem>& tmp) {
    if (n <= 1) return a;
    if (n == 2) {
        if (click_less(a[1], a[0], kenv)) std::swap(a[0], a[1]);
        return a;
    }
    if (n <= 24) {
        for (size_t i = 1; i < n; ++i) {
            ClickItem key = a[i];
            size_t j = i;
            while (j > 0 && click_less(key, a[j - 1], kenv)) {
                a[j] = a[j - 1];
                --j;
            }
            a[j] = key;
        }
        return a;
    }
    if (n >= 64) {
        uint64_t or_hi = 0;
        uint32_t c0[256], c1[256], c2[256], c3[256];
        std::memset(c0, 0, sizeof(c0));
        std::memset(c1, 0, sizeof(c1));
        std::memset(c2, 0, sizeof(c2));
        std::memset(c3, 0, sizeof(c3));
        for (size_t i = 0; i < n; ++i) {
            uint64_t t = a[i].ts_camp >> 20;
            or_hi |= (t >> 32);
            c0[t & 255U]++;
            c1[(t >> 8) & 255U]++;
            c2[(t >> 16) & 255U]++;
            c3[(t >> 24) & 255U]++;
        }
        if (__builtin_expect(or_hi == 0, 1)) {
            if (tmp.size() < n) tmp.resize(n);
            ClickItem* src = a;
            ClickItem* dst = tmp.data();
            uint32_t n32 = static_cast<uint32_t>(n);
            uint32_t* cs[4] = {c0, c1, c2, c3};
            unsigned shifts[4] = {20, 28, 36, 44};
            for (int p = 0; p < 4; ++p) {
                uint32_t* cnt = cs[p];
                unsigned sh = shifts[p];
                if (cnt[(src[0].ts_camp >> sh) & 255U] == n32) continue;
                uint32_t sum = 0;
                for (size_t i = 0; i < 256; ++i) { uint32_t c = cnt[i]; cnt[i] = sum; sum += c; }
                for (size_t i = 0; i < n; ++i) dst[cnt[(src[i].ts_camp >> sh) & 255U]++] = src[i];
                std::swap(src, dst);
            }
            // Fix any equal-timestamp runs: larger click_id first so smallest click_id is last!
            for (size_t i = 1; i < n; ++i) {
                if (__builtin_expect((src[i - 1].ts_camp >> 20) == (src[i].ts_camp >> 20), 0)) {
                    size_t st = i - 1;
                    uint64_t t0 = src[st].ts_camp >> 20;
                    while (i < n && (src[i].ts_camp >> 20) == t0) ++i;
                    std::sort(src + st, src + i, [&](const ClickItem& x, const ClickItem& y) {
                        if (x.cid == y.cid) {
                            if (fast::is_hex_key(x.cid) && (x.ts_camp & 0xFFFFFU) != (y.ts_camp & 0xFFFFFU)) {
                                kenv.long_users.need_exact.store(true, std::memory_order_relaxed);
                            }
                            return false;
                        }
                        return kenv.key_less(y.cid, x.cid, 'k');
                    });
                }
            }
            return src;
        }
    }
    std::sort(a, a + n, [&](const ClickItem& x, const ClickItem& y) {
        return click_less(x, y, kenv);
    });
    return a;
}

// ============================================================================
// Ultra-Compact Parquet UserDailyWriter (~5x smaller output files!)
// - user_id: DELTA_BYTE_ARRAY (zero bytes copied on repeat days for same user!)
// - day, sessions, events, top_category: RLE_DICTIONARY (3..5 bits/row!)
// - revenue_cents: DELTA_BINARY_PACKED (~1 bit/row for 90% zeros!)
// ============================================================================

struct UserDailyWriter {
    size_t rg_limit = 500000;
    size_t n_rows = 0;
    size_t ub_pos = 0;
    bool events_i32 = true;
    const std::vector<std::string>* cat_names = nullptr;
    pq::ParquetFileWriter writer;

    // On-the-fly DELTA_BYTE_ARRAY for user_id!
    uint64_t prev_uid_key = 0;
    char prev_uid_buf[64]{};
    size_t prev_uid_len = 0;
    bool has_prev_uid = false;
    std::vector<int32_t> uid_pre;
    std::vector<int32_t> uid_suf;
    std::vector<uint8_t> uid_bytes;

    std::vector<int32_t> col_day;
    std::vector<int32_t> col_sessions;
    std::vector<int32_t> col_events32;
    std::vector<int64_t> col_events64;
    std::vector<int64_t> col_revenue;
    std::vector<uint32_t> col_cat_ids;
    std::vector<uint8_t> def_bits;

    // Reusable encoding scratch buffers
    std::vector<uint8_t> enc_uid_hdr;
    std::vector<int32_t> dict_day, dict_sess, dict_ev;
    std::vector<uint32_t> enc_idx;
    std::vector<uint8_t> enc_day, enc_sess, enc_ev, enc_rev, enc_cat_dict, enc_cat;
    std::vector<uint32_t> val_map;

    bool open(const std::string& path, size_t est_rows, bool use_events_i32,
              const std::vector<std::string>& categories) {
        events_i32 = use_events_i32;
        cat_names = &categories;
        rg_limit = std::clamp<size_t>(est_rows, 4096, 500000);
        n_rows = 0;
        ub_pos = 0;
        uid_pre.resize(rg_limit);
        uid_suf.resize(rg_limit);
        uid_bytes.resize(rg_limit * 6 + 4096);
        col_day.resize(rg_limit);
        col_sessions.resize(rg_limit);
        if (events_i32) col_events32.resize(rg_limit);
        else col_events64.resize(rg_limit);
        col_revenue.resize(rg_limit);
        col_cat_ids.reserve(rg_limit / 6 + 64);
        def_bits.assign((rg_limit + 7) / 8 + 8, 0);
        return writer.open(path, {
            {"user_id", pq::PT_BYTE_ARRAY, pq::REP_REQUIRED, pq::CT_UTF8},
            {"day", pq::PT_INT32, pq::REP_REQUIRED, pq::CT_DATE},
            {"sessions", pq::PT_INT32, pq::REP_REQUIRED, pq::CT_NONE},
            {"events", events_i32 ? pq::PT_INT32 : pq::PT_INT64, pq::REP_REQUIRED, pq::CT_NONE},
            {"revenue_cents", pq::PT_INT64, pq::REP_REQUIRED, pq::CT_NONE},
            {"top_category", pq::PT_BYTE_ARRAY, pq::REP_OPTIONAL, pq::CT_UTF8},
        });
    }

    FAST_ALWAYS_INLINE void add_row(uint64_t uid_key, std::string_view uid, int32_t day, int32_t sessions,
                                    int64_t events, int64_t revenue, int32_t top_cat_id) {
        size_t i = n_rows;
        // On-the-fly DELTA_BYTE_ARRAY prefix/suffix encoding for user_id (ZERO push_back!)
        if (__builtin_expect(has_prev_uid && uid_key == prev_uid_key, 1)) {
            uid_pre[i] = static_cast<int32_t>(prev_uid_len);
            uid_suf[i] = 0;
        } else {
            size_t ulen = uid.size();
            size_t pre = 0;
            if (__builtin_expect(has_prev_uid && prev_uid_len > 0 && ulen > 0 && prev_uid_buf[0] == uid[0], 1)) {
                size_t m = std::min(prev_uid_len, ulen);
                pre = 1;
                while (pre < m && prev_uid_buf[pre] == uid[pre]) ++pre;
            }
            size_t suflen = ulen - pre;
            uid_pre[i] = static_cast<int32_t>(pre);
            uid_suf[i] = static_cast<int32_t>(suflen);
            if (__builtin_expect(ub_pos + suflen > uid_bytes.size(), 0)) {
                uid_bytes.resize(std::max(uid_bytes.size() * 2, ub_pos + suflen + 4096));
            }
            if (suflen) {
                std::memcpy(uid_bytes.data() + ub_pos, uid.data() + pre, suflen);
                ub_pos += suflen;
            }
            if (__builtin_expect(ulen <= sizeof(prev_uid_buf), 1)) {
                std::memcpy(prev_uid_buf, uid.data(), ulen);
                prev_uid_len = ulen;
                prev_uid_key = uid_key;
                has_prev_uid = true;
            } else {
                has_prev_uid = false;
            }
        }

        col_day[i] = day;
        col_sessions[i] = sessions;
        if (__builtin_expect(events_i32, 1)) col_events32[i] = static_cast<int32_t>(events);
        else col_events64[i] = events;
        col_revenue[i] = revenue;

        if (__builtin_expect(top_cat_id >= 0, 0)) {
            def_bits[i >> 3] |= static_cast<uint8_t>(1U << (i & 7));
            col_cat_ids.push_back(static_cast<uint32_t>(top_cat_id));
        }

        if (__builtin_expect(++n_rows == rg_limit, 0)) flush();
    }

    pq::ParquetFileWriter::ColumnData encode_small_i32(const int32_t* v, size_t n,
                                                       std::vector<int32_t>& dict_out,
                                                       std::vector<uint8_t>& body_out) {
        int32_t lo = v[0], hi = v[0];
        for (size_t i = 1; i < n; ++i) {
            int32_t x = v[i];
            if (x < lo) lo = x;
            if (x > hi) hi = x;
        }
        int64_t span = static_cast<int64_t>(hi) - static_cast<int64_t>(lo) + 1;
        body_out.clear();
        if (span > 65536) {
            pq::encode_delta_binary_packed(v, n, body_out);
            return pq::ParquetFileWriter::ColumnData{
                pq::ENC_DELTA_BINARY_PACKED, body_out.data(), body_out.size(), nullptr, nullptr, 0, 0, nullptr, 0
            };
        }
        val_map.assign(static_cast<size_t>(span), UINT32_MAX);
        dict_out.clear();
        if (enc_idx.size() < n) enc_idx.resize(n);
        uint32_t* vm = val_map.data();
        uint32_t* idx_p = enc_idx.data();
        for (size_t i = 0; i < n; ++i) {
            int32_t x = v[i];
            uint32_t& m = vm[static_cast<size_t>(x - lo)];
            if (__builtin_expect(m == UINT32_MAX, 0)) {
                m = static_cast<uint32_t>(dict_out.size());
                dict_out.push_back(x);
            }
            idx_p[i] = m;
        }
        uint32_t nd = static_cast<uint32_t>(dict_out.size());
        uint32_t w = (nd <= 1) ? 1U : static_cast<uint32_t>(32 - __builtin_clz(nd - 1));
        body_out.push_back(static_cast<uint8_t>(w));
        pq::encode_rle_hybrid(idx_p, n, w, body_out);
        return pq::ParquetFileWriter::ColumnData{
            pq::ENC_RLE_DICTIONARY, body_out.data(), body_out.size(), nullptr,
            dict_out.data(), dict_out.size() * sizeof(int32_t), static_cast<int32_t>(nd), nullptr, 0
        };
    }

    void flush() {
        size_t n = n_rows;
        if (n == 0) return;

        // 0. user_id: DELTA_BYTE_ARRAY
        enc_uid_hdr.clear();
        pq::encode_delta_binary_packed(uid_pre.data(), n, enc_uid_hdr);
        pq::encode_delta_binary_packed(uid_suf.data(), n, enc_uid_hdr);

        // 1. day: RLE_DICTIONARY
        auto c_day = encode_small_i32(col_day.data(), n, dict_day, enc_day);

        // 2. sessions: RLE_DICTIONARY
        auto c_sess = encode_small_i32(col_sessions.data(), n, dict_sess, enc_sess);

        // 3. events: RLE_DICTIONARY or DELTA_BINARY_PACKED
        pq::ParquetFileWriter::ColumnData c_ev;
        if (events_i32) {
            c_ev = encode_small_i32(col_events32.data(), n, dict_ev, enc_ev);
        } else {
            enc_ev.clear();
            pq::encode_delta_binary_packed(col_events64.data(), n, enc_ev);
            c_ev = {pq::ENC_DELTA_BINARY_PACKED, enc_ev.data(), enc_ev.size(), nullptr, nullptr, 0, 0, nullptr, 0};
        }

        // 4. revenue_cents: DELTA_BINARY_PACKED
        enc_rev.clear();
        pq::encode_delta_binary_packed(col_revenue.data(), n, enc_rev);
        pq::ParquetFileWriter::ColumnData c_rev{
            pq::ENC_DELTA_BINARY_PACKED, enc_rev.data(), enc_rev.size(), nullptr, nullptr, 0, 0, nullptr, 0
        };

        // 5. top_category: RLE_DICTIONARY (or empty PLAIN if all NULL!)
        pq::ParquetFileWriter::ColumnData c_cat;
        if (col_cat_ids.empty()) {
            c_cat = {pq::ENC_PLAIN, nullptr, 0, def_bits.data(), nullptr, 0, 0, nullptr, 0};
        } else {
            size_t nc = cat_names->size();
            val_map.assign(nc, UINT32_MAX);
            enc_cat_dict.clear();
            size_t ncat = col_cat_ids.size();
            if (enc_idx.size() < ncat) enc_idx.resize(ncat);
            uint32_t* idx_p = enc_idx.data();
            uint32_t nd = 0;
            for (size_t i = 0; i < ncat; ++i) {
                uint32_t cid = col_cat_ids[i];
                uint32_t& m = val_map[cid];
                if (__builtin_expect(m == UINT32_MAX, 0)) {
                    m = nd++;
                    const std::string& s = (*cat_names)[cid];
                    uint32_t l = static_cast<uint32_t>(s.size());
                    const uint8_t* lb = reinterpret_cast<const uint8_t*>(&l);
                    enc_cat_dict.insert(enc_cat_dict.end(), lb, lb + 4);
                    const uint8_t* sb = reinterpret_cast<const uint8_t*>(s.data());
                    enc_cat_dict.insert(enc_cat_dict.end(), sb, sb + l);
                }
                idx_p[i] = m;
            }
            uint32_t w = (nd <= 1) ? 1U : static_cast<uint32_t>(32 - __builtin_clz(nd - 1));
            enc_cat.clear();
            enc_cat.push_back(static_cast<uint8_t>(w));
            pq::encode_rle_hybrid(idx_p, ncat, w, enc_cat);
            c_cat = {
                pq::ENC_RLE_DICTIONARY, enc_cat.data(), enc_cat.size(), def_bits.data(),
                enc_cat_dict.data(), enc_cat_dict.size(), static_cast<int32_t>(nd), nullptr, 0
            };
        }

        std::vector<pq::ParquetFileWriter::ColumnData> cols(6);
        cols[0] = {pq::ENC_DELTA_BYTE_ARRAY, uid_bytes.data(), ub_pos, nullptr,
                   nullptr, 0, 0, enc_uid_hdr.data(), enc_uid_hdr.size()};
        cols[1] = c_day;
        cols[2] = c_sess;
        cols[3] = c_ev;
        cols[4] = c_rev;
        cols[5] = c_cat;
        writer.write_row_group(static_cast<int64_t>(n), cols);

        std::memset(def_bits.data(), 0, (n + 7) >> 3);
        n_rows = 0;
        ub_pos = 0;
        has_prev_uid = false;
        col_cat_ids.clear();
    }

    bool close() {
        flush();
        return writer.close();
    }
};

bool write_campaign_daily(const std::string& path, const Catalogs& cat, const CampDayAccum::MergedTable& acc) {
    pq::ParquetFileWriter writer;
    if (!writer.open(path, {
            {"campaign_id", pq::PT_BYTE_ARRAY, pq::REP_REQUIRED, pq::CT_UTF8},
            {"channel", pq::PT_BYTE_ARRAY, pq::REP_REQUIRED, pq::CT_UTF8},
            {"day", pq::PT_INT32, pq::REP_REQUIRED, pq::CT_DATE},
            {"sessions", pq::PT_INT64, pq::REP_REQUIRED, pq::CT_NONE},
            {"purchasing_sessions", pq::PT_INT64, pq::REP_REQUIRED, pq::CT_NONE},
            {"revenue_cents", pq::PT_INT64, pq::REP_REQUIRED, pq::CT_NONE},
            {"assisted_sessions", pq::PT_INT64, pq::REP_REQUIRED, pq::CT_NONE},
        })) {
        return false;
    }

    std::vector<uint8_t> col_cid, col_ch;
    std::vector<int32_t> col_day;
    std::vector<int64_t> col_sess, col_psess, col_rev, col_ass;
    auto push_str = [](std::vector<uint8_t>& buf, const std::string& s) {
        uint32_t l = static_cast<uint32_t>(s.size());
        const uint8_t* lb = reinterpret_cast<const uint8_t*>(&l);
        buf.insert(buf.end(), lb, lb + 4);
        const uint8_t* db = reinterpret_cast<const uint8_t*>(s.data());
        buf.insert(buf.end(), db, db + l);
    };

    acc.for_each_nonzero([&](uint32_t camp_id, int32_t day, const CampDayStats& st) {
        push_str(col_cid, cat.campaign_names[camp_id]);
        push_str(col_ch, cat.campaign_channels[camp_id]);
        col_day.push_back(day);
        col_sess.push_back(st.sessions);
        col_psess.push_back(st.purchasing_sessions);
        col_rev.push_back(st.revenue_cents);
        col_ass.push_back(st.assisted_sessions);
    });

    if (!col_day.empty()) {
        std::vector<pq::ParquetFileWriter::ColumnData> cols(7);
        cols[0] = {pq::ENC_PLAIN, col_cid.data(), col_cid.size(), nullptr, nullptr, 0, 0, nullptr, 0};
        cols[1] = {pq::ENC_PLAIN, col_ch.data(), col_ch.size(), nullptr, nullptr, 0, 0, nullptr, 0};
        cols[2] = {pq::ENC_PLAIN, col_day.data(), col_day.size() * sizeof(int32_t), nullptr, nullptr, 0, 0, nullptr, 0};
        cols[3] = {pq::ENC_PLAIN, col_sess.data(), col_sess.size() * sizeof(int64_t), nullptr, nullptr, 0, 0, nullptr, 0};
        cols[4] = {pq::ENC_PLAIN, col_psess.data(), col_psess.size() * sizeof(int64_t), nullptr, nullptr, 0, 0, nullptr, 0};
        cols[5] = {pq::ENC_PLAIN, col_rev.data(), col_rev.size() * sizeof(int64_t), nullptr, nullptr, 0, 0, nullptr, 0};
        cols[6] = {pq::ENC_PLAIN, col_ass.data(), col_ass.size() * sizeof(int64_t), nullptr, nullptr, 0, 0, nullptr, 0};
        writer.write_row_group(static_cast<int64_t>(col_day.size()), cols);
    }
    return writer.close();
}

} // namespace

int main(int argc, char** argv) {
    auto t_total = Clock::now();

    const char* env_data = std::getenv("DATA_DIR");
    const char* env_out = std::getenv("OUTPUT_DIR");
    const char* env_ncpu = std::getenv("NCPU");
    const char* env_mmap = std::getenv("USE_MMAP");
    const char* env_pf = std::getenv("PREFETCH");
    const char* env_pop = std::getenv("USE_POPULATE");
    const char* env_huge = std::getenv("HUGE");
    const char* env_exact = std::getenv("EXACT_LONG");
    const char* env_rel = std::getenv("RELEASE_MAP");

    std::string data_dir = env_data ? env_data : ((argc >= 3) ? argv[argc - 2] : "/data");
    std::string out_dir = env_out ? env_out : ((argc >= 3) ? argv[argc - 1] : "/output");
    size_t ncpu = env_ncpu ? static_cast<size_t>(std::max(1, std::atoi(env_ncpu))) : detect_cpus();
    bool use_mmap = env_mmap ? (std::atoi(env_mmap) != 0) : true;
    bool use_prefetch = env_pf ? (std::atoi(env_pf) != 0) : false;
    (void)use_prefetch;
    bool use_populate = env_pop ? (std::atoi(env_pop) != 0) : false;
    if (env_huge) g_use_huge = (std::atoi(env_huge) != 0);

    ensure_dir(out_dir);
    ensure_dir(out_dir + "/user_daily");
    ensure_dir(out_dir + "/campaign_daily");

    int64_t window_start = 0, window_end = 0;
    if (!parse_manifest(data_dir + "/manifest.json", window_start, window_end)) {
        std::fprintf(stderr, "Failed to parse manifest.json\n");
        return 1;
    }
    int64_t click_start = window_start - WEEK_MS;
    int32_t min_day = fast::floor_div_day(window_start);
    int32_t max_day = fast::floor_div_day(window_end - 1);
    int64_t day0_end_off = (static_cast<int64_t>(min_day + 1) * DAY_MS) - window_start;
    int64_t ws_sec = window_start / 1000LL;
    bool ws_whole_sec = (window_start >= 0) && (window_start % 1000LL == 0);

    bool compact_ev_mode = (window_end >= window_start) &&
                           (static_cast<uint64_t>(window_end - window_start) <= 0x7FFFFFFFULL);

    size_t num_30m_buckets = 0;
    bool can_bucket_sess = false;
    if (compact_ev_mode && window_end > window_start) {
        num_30m_buckets = static_cast<size_t>((window_end - window_start - 1) / SESSION_GAP_MS) + 1;
        can_bucket_sess = (num_30m_buckets <= 2048);
    }

    fast::DateTable dtab;
    dtab.precompute(window_start, window_end);

    auto event_paths = list_csv_files(data_dir + "/events");
    auto click_paths = list_csv_files(data_dir + "/clicks");
    auto link_paths = list_csv_files(data_dir + "/links");

    size_t n_ev_files = event_paths.size();
    size_t n_cl_files = click_paths.size();
    size_t n_ln_files = link_paths.size();
    auto event_files = std::make_unique<InputFile[]>(n_ev_files);
    auto click_files = std::make_unique<InputFile[]>(n_cl_files);
    auto link_files = std::make_unique<InputFile[]>(n_ln_files);

    auto t_cat = Clock::now();
    Catalogs catalogs;
    load_catalogs(data_dir, catalogs);
    double ms_cat = elapsed_sec(t_cat) * 1000.0;
    if (catalogs.categories.size() > 255) compact_ev_mode = false;
    if (!compact_ev_mode) can_bucket_sess = false;

    // Open all input files single-threaded BEFORE spawning worker threads:
    // zero files_struct lock contention, zero pthread_create/mmap_lock overlap (~0.1-0.5ms for 114 files!).
    double ms_open = 0.0;
    {
        auto to = Clock::now();
        for (size_t i = 0; i < n_ev_files; ++i) event_files[i].open(event_paths[i], use_mmap);
        for (size_t i = 0; i < n_cl_files; ++i) click_files[i].open(click_paths[i], use_mmap);
        for (size_t i = 0; i < n_ln_files; ++i) link_files[i].open(link_paths[i], use_mmap);
        ms_open = elapsed_sec(to) * 1000.0;
    }

    size_t total_event_bytes = 0;
    for (size_t i = 0; i < n_ev_files; ++i) total_event_bytes += event_files[i].size;
    size_t total_link_bytes = 0;
    for (size_t i = 0; i < n_ln_files; ++i) total_link_bytes += link_files[i].size;

    // Avoid MADV_DONTNEED TLB shootdown storms in Stage 1 unless input is huge (>= 10 GB) or explicitly requested.
    bool use_release_map = env_rel ? (std::atoi(env_rel) != 0) : (total_event_bytes >= (10ULL << 30));

    bool use_events_i32 = (total_event_bytes < 20ULL * 1024 * 1024 * 1024);

    size_t num_threads = ncpu;
    if (total_event_bytes < 512 * 1024 && total_link_bytes < 512 * 1024) {
        num_threads = 1;
    }

    auto t_s0_start = Clock::now();
    ThreadTeam team(num_threads);
    double ms_team = elapsed_sec(t_s0_start) * 1000.0;

    size_t num_buckets = (total_event_bytes < 4 * 1024 * 1024)
                             ? 16
                             : ((total_event_bytes >= 1536ULL * 1024 * 1024) ? 512 : 256);
    unsigned bucket_shift = (num_buckets == 16) ? 60U : ((num_buckets == 512) ? 55U : 56U);
    size_t num_parts = (total_event_bytes < 4 * 1024 * 1024)
                           ? 16
                           : ((total_event_bytes >= 256ULL * 1024 * 1024) ? 256 : 128);
    unsigned part_shift = (num_parts == 16) ? 60U : ((num_parts == 256) ? 56U : 57U);
    size_t slab_bytes = (total_event_bytes < 4 * 1024 * 1024) ? 4096 : 32768;

    KeyEnv kenv;
    kenv.long_users.exact_mode = (env_exact && std::atoi(env_exact) != 0);
    std::vector<LongUserCollector> long_collectors(num_threads);
    IdentityMap64 identity_map;
    std::atomic<uint32_t> identity_ready{0};
    double dsu_ms = 0.0, dsu_parse_ms = 0.0, dsu_union_ms = 0.0, dsu_idmap_ms = 0.0;

    bool overlap_dsu = (num_threads >= 2) && (total_link_bytes > 0) && (total_event_bytes >= total_link_bytes * 4);
    auto t_s0_end = Clock::now();
    if (!overlap_dsu) {
        auto t_dsu = Clock::now();
        auto link_chunks = build_read_chunks(link_files.get(), n_ln_files, 1 << 20);
        std::vector<std::vector<LinkEdge64>> link_bufs(team.size());
        if (!link_chunks.empty()) {
            std::atomic<size_t> next_lc{0};
            team.run([&](size_t tid) {
                if (tid >= num_threads) return;
                std::vector<char> io_buf;
                auto& edges = link_bufs[tid];
                auto& col = long_collectors[tid];
                edges.reserve(total_link_bytes / (24 * num_threads) + 64);
                while (true) {
                    size_t ci = next_lc.fetch_add(1, std::memory_order_relaxed);
                    if (ci >= link_chunks.size()) break;
                    const auto& ch = link_chunks[ci];
                    auto sv = get_aligned_chunk(link_files[ch.file_idx], ch, io_buf);
                    if (sv.empty()) continue;
                    const char* sv_end = sv.data() + sv.size();
                    fast::scan_csv_chunk<2>(sv.data(), sv.size(), [&](const fast::Field (&f)[2]) __attribute__((always_inline)) {
                        if (f[0].len == 0 || f[1].len == 0) return;
                        if (f[0].len == f[1].len && std::memcmp(f[0].ptr, f[1].ptr, f[0].len) == 0) return;
                        edges.push_back(LinkEdge64{
                            kenv.encode_user(f[0].ptr, f[0].len, sv_end, col),
                            kenv.encode_user(f[1].ptr, f[1].len, sv_end, col)
                        });
                    });
                }
            });
            build_identity_from_links(link_bufs, kenv, identity_map, team, num_threads);
        }
        dsu_ms = elapsed_sec(t_dsu) * 1000.0;
        identity_ready.store(1U, std::memory_order_release);
    }

    // Async kernel readahead in consumption order (links -> events -> clicks), cancellable when Stage 1 finishes!
    std::atomic<bool> stop_pf{false};
    std::thread prefetcher;
#if defined(__linux__) && defined(POSIX_FADV_WILLNEED)
    if (use_prefetch && use_mmap && !use_populate && total_event_bytes >= 32ULL * 1024 * 1024) {
        prefetcher = std::thread([&]() {
            auto adv = [&](const InputFile& f) {
                if (f.fd < 0 || stop_pf.load(std::memory_order_relaxed)) return;
                posix_fadvise(f.fd, 0, static_cast<off_t>(f.size), POSIX_FADV_WILLNEED);
            };
            for (size_t i = 0; i < n_ln_files; ++i) adv(link_files[i]);
            for (size_t i = 0; i < n_ev_files; ++i) adv(event_files[i]);
            for (size_t i = 0; i < n_cl_files; ++i) adv(click_files[i]);
        });
    }
#endif

    // Stage 1: Overlapped links+DSU + 4KB block-classified SIMD events & clicks!
    auto t_s1 = Clock::now();
    auto event_chunks = build_read_chunks(event_files.get(), n_ev_files, READ_CHUNK_SIZE);
    auto click_chunks = build_read_chunks(click_files.get(), n_cl_files, READ_CHUNK_SIZE);

    std::vector<SlabArena> arenas(num_threads);
    std::vector<EvBucketSet> ev_buckets(num_threads);
    std::vector<FastBucketSet<LongEv32>> long_ev_buckets(num_threads);
    std::vector<std::vector<std::vector<uint32_t>>> ev_purchases(num_threads);
    std::vector<std::vector<std::vector<WideEvRow>>> wide_ev_buckets(num_threads);
    std::vector<BucketSet<PersonClick>> person_cl(num_threads);
    for (size_t t = 0; t < num_threads; ++t) {
        arenas[t].init(slab_bytes, (total_event_bytes < 4 * 1024 * 1024) ? 65536 : 4 * 1024 * 1024);
        ev_buckets[t].init(num_buckets, slab_bytes);
        long_ev_buckets[t].init(num_buckets, slab_bytes);
        ev_purchases[t].resize(num_buckets);
        wide_ev_buckets[t].resize(num_buckets);
        person_cl[t].init(num_parts, slab_bytes);
    }

    std::atomic<size_t> next_ec{0};
    std::atomic<size_t> next_cc{0};
    std::atomic<uint64_t> prof_s1_ev_ns{0}, prof_s1_cl_ns{0};

    const uint64_t ev_span_u64 = (window_end > window_start) ? static_cast<uint64_t>(window_end - window_start) : 0ULL;
    const uint64_t cl_span_u64 = (window_end > click_start) ? static_cast<uint64_t>(window_end - click_start) : 0ULL;

    auto scan_clicks_for = [&](size_t tid, std::vector<char>& io_buf) {
        auto& my_cl = person_cl[tid];
        auto& arena = arenas[tid];
        auto& col = long_collectors[tid];
        uint64_t b_uid[64], b_h[64], b_cid[64], b_tc[64];
        size_t nb = 0;
        auto flush_batch = [&]() {
            identity_map.resolve_batch(b_uid, b_h, nb);
            for (size_t i = 0; i < nb; ++i) {
                size_t p = static_cast<size_t>(b_h[i] >> part_shift);
                my_cl.push(p, PersonClick{b_uid[i], b_cid[i], b_tc[i]}, arena);
            }
            nb = 0;
        };

        while (true) {
            size_t ci = next_cc.fetch_add(1, std::memory_order_relaxed);
            if (ci >= click_chunks.size()) break;
            const auto& ch = click_chunks[ci];
            const auto& cf = click_files[ch.file_idx];
            auto sv = get_aligned_chunk(cf, ch, io_buf, use_populate);
            if (sv.empty()) continue;
            const char* sv_end = sv.data() + sv.size();

            fast::scan_csv_chunk<4>(sv.data(), sv.size(), [&](const fast::Field (&f)[4]) __attribute__((always_inline)) {
                if (__builtin_expect(f[0].len == 0 || f[1].len == 0 || f[2].len == 0, 0)) return;
                uint32_t camp_id = catalogs.lookup_campaign(f[2].ptr, f[2].len);
                if (__builtin_expect(camp_id == 0, 0)) return;
                int64_t ts_ms = 0;
                if (__builtin_expect(!fast::parse_timestamp_ms(f[3].ptr, f[3].len, dtab, ts_ms), 0)) return;
                uint64_t ts_off = static_cast<uint64_t>(ts_ms - click_start);
                if (__builtin_expect(ts_ms < click_start || ts_off >= cl_span_u64, 0)) return;

                b_uid[nb] = kenv.encode_user(f[1].ptr, f[1].len, sv_end, col);
                b_cid[nb] = kenv.encode_click(f[0].ptr, f[0].len, sv_end);
                b_tc[nb] = (ts_off << 20) | camp_id;
                if (__builtin_expect(++nb == 64, 0)) flush_batch();
            });
            if (use_mmap && use_release_map) release_mmap_slice(cf.mapped.load(std::memory_order_relaxed), sv);
        }
        if (nb > 0) flush_batch();
    };

    auto scan_events_body = [&](size_t tid) {
        auto tev0 = Clock::now();
        std::vector<char> io_buf;
        if (!use_mmap) io_buf.reserve(READ_CHUNK_SIZE + 8192);
        auto& my_ev = ev_buckets[tid];
        auto& my_lev = long_ev_buckets[tid];
        auto& my_purch = ev_purchases[tid];
        auto& my_wide = wide_ev_buckets[tid];
        auto& arena = arenas[tid];
        auto& col = long_collectors[tid];

        while (true) {
            size_t ci = next_ec.fetch_add(1, std::memory_order_relaxed);
            if (ci >= event_chunks.size()) break;
            const auto& ch = event_chunks[ci];
            const auto& ef = event_files[ch.file_idx];
            auto sv = get_aligned_chunk(ef, ch, io_buf, use_populate);
            if (sv.empty()) continue;
            const char* sv_end = sv.data() + sv.size();

            fast::scan_csv_chunk<7>(sv.data(), sv.size(), [&](const fast::Field (&f)[7]) __attribute__((always_inline)) {
                if (__builtin_expect(f[0].len == 0, 0)) return;

                int64_t ingest_sec = 0, ingest_day_sec = 0;
                uint64_t ingest_ym = 0;
                uint16_t ingest_dd = 0;
                if (__builtin_expect(!fast::parse_ingest_f1_sec(f[3].ptr, f[3].len, dtab, ingest_sec, ingest_day_sec,
                                                                 ingest_ym, ingest_dd), 0)) {
                    return;
                }
                int64_t ingest_ms = ingest_sec * 1000LL;

                // Encode event_id: raw 64-bit (87.5%), exact 128-bit hex (12.5%), or sharded exotic (<0.01%)!
                uint64_t eid = fast::try_pack_raw(f[0].ptr, f[0].len, 'e', sv_end);
                uint64_t eid_lo = 0;
                bool is_long_eid = false;
                size_t bucket;
                if (__builtin_expect(eid != 0, 1)) {
                    bucket = static_cast<size_t>(fast::hash_u64(eid) >> bucket_shift);
                } else if (__builtin_expect(f[0].len == 33 && f[0].ptr[0] == 'e' &&
                                            fast::parse_hex128(f[0].ptr + 1, eid, eid_lo), 1)) {
                    is_long_eid = true;
                    bucket = static_cast<size_t>(fast::hash_u128(eid, eid_lo) >> bucket_shift);
                } else {
                    eid = kenv.exotic.intern(f[0].ptr, f[0].len);
                    bucket = static_cast<size_t>(fast::hash_u64(eid) >> bucket_shift);
                }

                bool valid = false;
                int64_t ts_ms = 0, ts_sec = 0;
                uint64_t ts_off_u64 = 0;
                uint32_t price_cents = 0;
                uint32_t cat_id = 0;

                if (__builtin_expect(f[1].len > 0 && f[5].len > 0, 1)) {
                    uint8_t etype = 0;
                    if (__builtin_expect(f[4].len == 4, 1)) {
                        uint32_t w;
                        std::memcpy(&w, f[4].ptr, 4);
                        if (w == 0x77656976U || w == 0x74726163U) etype = 1;
                    } else if (f[4].len == 8) {
                        uint64_t w;
                        std::memcpy(&w, f[4].ptr, 8);
                        if (w == 0x6573616863727570ULL) etype = 2;
                    }

                    if (__builtin_expect(etype != 0, 1)) {
                        if (__builtin_expect(fast::parse_timestamp_ms_sec(f[2].ptr, f[2].len, dtab, ts_ms, ts_sec,
                                                                          ingest_ym, ingest_dd, ingest_day_sec), 1)) {
                            ts_off_u64 = static_cast<uint64_t>(ts_ms - window_start);
                            if (__builtin_expect(ts_ms >= window_start && ts_off_u64 < ev_span_u64 &&
                                                     (ingest_ms - ts_ms) <= DAY_MS, 1)) {
                                if (__builtin_expect(etype == 1, 1)) {
                                    valid = true;
                                } else if (__builtin_expect(fast::parse_price_cents(f[6].ptr, f[6].len, price_cents), 1)) {
                                    valid = true;
                                    cat_id = catalogs.lookup_product_cat(f[5].ptr, f[5].len);
                                }
                            }
                        }
                    }
                }

                if (__builtin_expect(!valid, 0)) {
                    if (__builtin_expect(!is_long_eid, 1)) {
                        my_ev.push(bucket, eid, 0ULL, static_cast<uint64_t>(ingest_sec), arena);
                    } else {
                        my_lev.push(bucket, LongEv32{eid, eid_lo, 0ULL, static_cast<uint64_t>(ingest_sec)}, arena);
                    }
                    return;
                }

                uint64_t uid = kenv.encode_user(f[1].ptr, f[1].len, sv_end, col);
                if (__builtin_expect(compact_ev_mode, 1)) {
                    uint32_t ts_off = static_cast<uint32_t>(ts_off_u64);
                    int64_t delta = ingest_sec - ts_sec;
                    auto& pv = my_purch[bucket];
                    if (__builtin_expect(delta >= -65536 && delta <= 196607 &&
                                             (price_cents == 0 || pv.size() < 32767), 1)) {
                        uint32_t pidx_p1 = 0;
                        if (__builtin_expect(price_cents > 0, 0)) {
                            pv.push_back((cat_id << 24) | price_cents);
                            pidx_p1 = static_cast<uint32_t>(pv.size());
                        }
                        uint64_t d_biased = static_cast<uint64_t>(delta + 65536);
                        uint64_t meta = (static_cast<uint64_t>(ts_off) << 33) | (d_biased << 15) | pidx_p1;
                        if (__builtin_expect(!is_long_eid, 1)) {
                            my_ev.push(bucket, eid, uid, meta, arena);
                        } else {
                            my_lev.push(bucket, LongEv32{eid, eid_lo, uid, meta}, arena);
                        }
                        return;
                    }
                }
                my_wide[bucket].push_back(WideEvRow{eid, eid_lo, uid, ingest_sec, ts_ms, price_cents, cat_id, is_long_eid});
            });
            if (use_mmap && use_release_map) release_mmap_slice(ef.mapped.load(std::memory_order_relaxed), sv);
        }
        my_ev.finish_all();
        my_lev.finish_all(arena);
        auto tcl0 = Clock::now();
        prof_s1_ev_ns.fetch_add(
            static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(tcl0 - tev0).count()),
            std::memory_order_relaxed);
        // Wait for Thread 0's overlapped DSU if still running so ALL threads scan clicks in parallel!
        while (identity_ready.load(std::memory_order_acquire) == 0U) {
            identity_ready.wait(0U, std::memory_order_acquire);
        }
        auto tcl1 = Clock::now();
        scan_clicks_for(tid, io_buf);
        prof_s1_cl_ns.fetch_add(
            static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - tcl1).count()),
            std::memory_order_relaxed);
    };

    team.run([&](size_t tid) {
        if (tid >= num_threads) return;
        if (overlap_dsu && tid == 0) {
            auto td = Clock::now();
            parse_links_and_build_identity_single(link_files.get(), n_ln_files, total_link_bytes,
                                                  kenv, long_collectors[0], identity_map,
                                                  dsu_parse_ms, dsu_union_ms, dsu_idmap_ms);
            dsu_ms = elapsed_sec(td) * 1000.0;
            identity_ready.store(1U, std::memory_order_release);
            identity_ready.notify_all();
        }
        scan_events_body(tid);
    });
    stop_pf.store(true, std::memory_order_relaxed);
    if (prefetcher.joinable()) prefetcher.join();

    // Finish any remaining clicks before verifying long user IDs!
    if (next_cc.load(std::memory_order_relaxed) < click_chunks.size()) {
        team.run([&](size_t tid) {
            if (tid >= num_threads) return;
            std::vector<char> io_buf;
            if (!use_mmap) io_buf.reserve(READ_CHUNK_SIZE + 8192);
            scan_clicks_for(tid, io_buf);
        });
    }
    for (size_t t = 0; t < num_threads; ++t) person_cl[t].finish_all();

    auto t_vlu = Clock::now();
    if (!kenv.long_users.verify_and_build(long_collectors, team, num_threads)) {
        // Rare 55-bit hex prefix collision or undecided exotic-vs-hex comparison: exact re-run!
        setenv("EXACT_LONG", "1", 1);
#if defined(__linux__)
        execv("/proc/self/exe", argv);
#endif
        execv(argv[0], argv);
        return 1;
    }
    double ms_vlu = elapsed_sec(t_vlu) * 1000.0;

    std::fprintf(stderr, "Stage 1 in %.3fs (s0=%.1fms [team=%.1f open=%.1f cat=%.1f] dsu=%.1fms%s[p=%.1f u=%.1f m=%.1f] vlu=%.1fms | avg/th: ev=%.1fms cl=%.1fms)\n",
                 elapsed_sec(t_s1),
                 std::chrono::duration<double, std::milli>(t_s0_end - t_total).count(),
                 ms_team, ms_open, ms_cat,
                 dsu_ms, overlap_dsu ? "[ovl]" : "[par]", dsu_parse_ms, dsu_union_ms, dsu_idmap_ms, ms_vlu,
                 prof_s1_ev_ns.load() / (1e6 * num_threads),
                 prof_s1_cl_ns.load() / (1e6 * num_threads));

    // Stage 2: SOA L2-Bitmap Pre-Filter Dedup + L1-Bitmap LongEv32 Dedup + Batch-64 Fast Emit!
    auto t_s2 = Clock::now();
    std::vector<FastBucketSet<PersonEv16>> person_ev(num_threads);
    std::vector<std::vector<std::vector<WidePersonEv>>> wide_person_ev(num_threads);
    for (size_t t = 0; t < num_threads; ++t) {
        person_ev[t].init(num_parts, slab_bytes);
        wide_person_ev[t].resize(num_parts);
    }

    std::atomic<size_t> next_db{0};
    std::atomic<bool> has_any_wide{false};
    std::atomic<uint64_t> prof_p1_ns{0}, prof_p2_ns{0};

    team.run([&](size_t tid) {
        if (tid >= num_threads) return;
        auto& arena = arenas[tid];
        const size_t pay_off = ev_buckets[0].pay_byte_off;
        struct DedupSlot {
            uint32_t tag;
            uint32_t loc;
        };
        struct DupEntry {
            uint64_t eid;
            uint64_t uid;
            uint64_t meta;
            uint32_t owner_t;
            uint32_t pad;
        };
        struct LongSlot {
            uint64_t hi;
            uint64_t lo;
            uint64_t uid;
            uint64_t meta;
            uint32_t owner_t;
            uint32_t used;
        };

        std::vector<uint64_t> seen_bits, coll_bits;
        std::vector<uint32_t> dup_ht; // 0 = empty, else index+1 into dup_entries
        std::vector<DupEntry> dup_entries;
        std::vector<uint64_t> l_seen_bits, l_coll_bits;
        std::vector<uint32_t> l_dup_ht;
        std::vector<LongSlot> l_dup_entries;
        std::vector<LongSlot> long_ht;
        std::vector<uint32_t> long_used;

        // Fallback full-table buffers (only used when bucket has WideEvRow!)
        std::vector<DedupSlot> full_table;
        std::vector<uint8_t*> slab_ptrs;
        std::vector<WideEvRow*> wide_ptrs;

        auto& my_ev = person_ev[tid];
        auto& my_wev = wide_person_ev[tid];

        auto get_ev_ingest = [&](uint64_t uid, uint64_t meta) -> int64_t {
            if (__builtin_expect(uid == 0, 0)) return static_cast<int64_t>(meta);
            uint32_t ts_off = static_cast<uint32_t>(meta >> 33);
            int64_t ts_sec = __builtin_expect(ws_whole_sec, 1)
                                 ? (ws_sec + static_cast<int64_t>(ts_off / 1000U))
                                 : ((window_start + static_cast<int64_t>(ts_off)) / 1000LL);
            int64_t delta = static_cast<int64_t>((meta >> 15) & 0x3FFFFULL) - 65536LL;
            return ts_sec + delta;
        };

        uint64_t loc_p1_ns = 0, loc_p2_ns = 0;
        uint64_t b_uid[64], b_h[64], b_tspc[64];
        size_t nb = 0;
        auto flush_ev_batch = [&]() {
            identity_map.resolve_batch(b_uid, b_h, nb);
            for (size_t k = 0; k < nb; ++k) {
                size_t p_part = static_cast<size_t>(b_h[k] >> part_shift);
                my_ev.push(p_part, PersonEv16{b_uid[k], b_tspc[k]}, arena);
            }
            nb = 0;
        };

        while (true) {
            size_t b = next_db.fetch_add(1, std::memory_order_relaxed);
            if (b >= num_buckets) break;

            size_t total_recs = 0, total_long = 0, total_wide = 0;
            for (size_t t = 0; t < num_threads; ++t) {
                total_recs += ev_buckets[t].lists[b].total_count;
                total_long += long_ev_buckets[t].lists[b].total_count;
                total_wide += wide_ev_buckets[t][b].size();
            }
            if (total_recs + total_long + total_wide == 0) continue;
            if (total_wide > 0) has_any_wide.store(true, std::memory_order_relaxed);

            auto tp1 = Clock::now();
            if (__builtin_expect(total_wide == 0, 1)) {
                // FAST PATH (99.999%): SOA Bitmap Pre-Filter for short/exotic events!
                if (total_recs > 0) {
                    size_t nbits = next_pow2(std::max<size_t>(64, total_recs * 16));
                    size_t nwords = nbits >> 6;
                    uint64_t fmask = static_cast<uint64_t>(nbits - 1);
                    if (seen_bits.size() < nwords) {
                        seen_bits.resize(nwords);
                        coll_bits.resize(nwords);
                    }
                    std::memset(seen_bits.data(), 0, nwords * sizeof(uint64_t));
                    std::memset(coll_bits.data(), 0, nwords * sizeof(uint64_t));
                    uint64_t* sb = seen_bits.data();
                    uint64_t* cb = coll_bits.data();
                    uint64_t h_ring[16];

                    for (size_t t = 0; t < num_threads; ++t) {
                        const auto& lst = ev_buckets[t].lists[b];
                        for (size_t si = 0; si < lst.slabs.size(); ++si) {
                            const uint64_t* eids = reinterpret_cast<const uint64_t*>(lst.slabs[si]);
                            uint32_t cnt = lst.counts[si];
                            uint32_t pre = std::min<uint32_t>(8, cnt);
                            for (uint32_t j = 0; j < pre; ++j) {
                                uint64_t nh = fast::hash_u64(eids[j]);
                                h_ring[j] = nh;
                                size_t nw = static_cast<size_t>((nh & fmask) >> 6);
                                __builtin_prefetch(&sb[nw], 1, 1);
                                __builtin_prefetch(&cb[nw], 1, 1);
                            }
                            for (uint32_t i = 0; i < cnt; ++i) {
                                uint64_t f = h_ring[i & 15] & fmask;
                                if (__builtin_expect(i + 8 < cnt, 1)) {
                                    uint64_t nh = fast::hash_u64(eids[i + 8]);
                                    h_ring[(i + 8) & 15] = nh;
                                    size_t nw = static_cast<size_t>((nh & fmask) >> 6);
                                    __builtin_prefetch(&sb[nw], 1, 1);
                                    __builtin_prefetch(&cb[nw], 1, 1);
                                }
                                size_t w = static_cast<size_t>(f >> 6);
                                uint64_t bit = 1ULL << (f & 63U);
                                cb[w] |= sb[w] & bit;
                                sb[w] |= bit;
                            }
                        }
                    }

                    auto tp2 = Clock::now();
                    loc_p1_ns += static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(tp2 - tp1).count());

                    // Pass 2: 90%+ bypass hash table completely -> direct Batch-64 emit!
                    size_t dcap = next_pow2(std::max<size_t>(64, total_recs / 4));
                    if (dup_ht.size() < dcap) dup_ht.resize(dcap);
                    std::memset(dup_ht.data(), 0, dcap * sizeof(uint32_t));
                    size_t dmask = dcap - 1;
                    dup_entries.clear();

                    auto grow_dup_ht = [&]() {
                        dcap *= 2;
                        dmask = dcap - 1;
                        dup_ht.assign(dcap, 0);
                        for (size_t di = 0; di < dup_entries.size(); ++di) {
                            size_t sl = static_cast<size_t>(fast::hash_u64(dup_entries[di].eid)) & dmask;
                            while (dup_ht[sl]) sl = (sl + 1) & dmask;
                            dup_ht[sl] = static_cast<uint32_t>(di + 1);
                        }
                    };

                    for (size_t t = 0; t < num_threads; ++t) {
                        const auto& lst = ev_buckets[t].lists[b];
                        const auto& pv = ev_purchases[t][b];
                        for (size_t si = 0; si < lst.slabs.size(); ++si) {
                            const uint64_t* eids = reinterpret_cast<const uint64_t*>(lst.slabs[si]);
                            const EvPay* pays = reinterpret_cast<const EvPay*>(lst.slabs[si] + pay_off);
                            uint32_t cnt = lst.counts[si];
                            uint32_t pre = std::min<uint32_t>(8, cnt);
                            for (uint32_t j = 0; j < pre; ++j) {
                                uint64_t nh = fast::hash_u64(eids[j]);
                                h_ring[j] = nh;
                                __builtin_prefetch(&cb[static_cast<size_t>((nh & fmask) >> 6)], 0, 1);
                            }
                            for (uint32_t i = 0; i < cnt; ++i) {
                                uint64_t h = h_ring[i & 15];
                                if (__builtin_expect(i + 8 < cnt, 1)) {
                                    uint64_t nh = fast::hash_u64(eids[i + 8]);
                                    h_ring[(i + 8) & 15] = nh;
                                    __builtin_prefetch(&cb[static_cast<size_t>((nh & fmask) >> 6)], 0, 1);
                                }
                                uint64_t f = h & fmask;
                                if (__builtin_expect((cb[f >> 6] & (1ULL << (f & 63U))) == 0, 1)) {
                                    uint64_t uid = pays[i].uid;
                                    if (__builtin_expect(uid == 0, 0)) continue;
                                    uint64_t meta = pays[i].meta;
                                    uint32_t pidx = static_cast<uint32_t>(meta & 0x7FFFU);
                                    uint32_t pc = __builtin_expect(pidx == 0, 1) ? 0U : pv[pidx - 1];
                                    b_uid[nb] = uid;
                                    b_tspc[nb] = ((meta >> 33) << 32) | pc;
                                    if (__builtin_expect(++nb == 64, 0)) flush_ev_batch();
                                } else {
                                    uint64_t eid = eids[i];
                                    uint64_t uid = pays[i].uid;
                                    uint64_t meta = pays[i].meta;
                                    size_t sl = static_cast<size_t>(h) & dmask;
                                    while (true) {
                                        uint32_t cur = dup_ht[sl];
                                        if (cur == 0) {
                                            if (__builtin_expect((dup_entries.size() + 1) * 10 >= dcap * 6, 0)) {
                                                grow_dup_ht();
                                                sl = static_cast<size_t>(h) & dmask;
                                                while (dup_ht[sl]) sl = (sl + 1) & dmask;
                                            }
                                            dup_entries.push_back(DupEntry{eid, uid, meta, static_cast<uint32_t>(t), 0});
                                            dup_ht[sl] = static_cast<uint32_t>(dup_entries.size());
                                            break;
                                        }
                                        auto& de = dup_entries[cur - 1];
                                        if (de.eid == eid) {
                                            if (get_ev_ingest(uid, meta) < get_ev_ingest(de.uid, de.meta)) {
                                                de.uid = uid;
                                                de.meta = meta;
                                                de.owner_t = static_cast<uint32_t>(t);
                                            }
                                            break;
                                        }
                                        sl = (sl + 1) & dmask;
                                    }
                                }
                            }
                        }
                        ev_buckets[t].recycle_bucket(b, arena);
                    }

                    for (const auto& de : dup_entries) {
                        if (de.uid == 0) continue;
                        uint32_t pidx = static_cast<uint32_t>(de.meta & 0x7FFFU);
                        uint32_t pc = (pidx == 0) ? 0U : ev_purchases[de.owner_t][b][pidx - 1];
                        b_uid[nb] = de.uid;
                        b_tspc[nb] = ((de.meta >> 33) << 32) | pc;
                        if (__builtin_expect(++nb == 64, 0)) flush_ev_batch();
                    }
                    loc_p2_ns += static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - tp2).count());
                }

                // Exact 128-bit dedup for the 12.5% Scorer-v5 33-byte hex events with L1-resident bitmap pre-filter!
                if (total_long > 0) {
                    auto tl0 = Clock::now();
                    size_t lnbits = next_pow2(std::max<size_t>(64, total_long * 16));
                    size_t lnwords = lnbits >> 6;
                    uint64_t lfmask = static_cast<uint64_t>(lnbits - 1);
                    if (l_seen_bits.size() < lnwords) {
                        l_seen_bits.resize(lnwords);
                        l_coll_bits.resize(lnwords);
                    }
                    std::memset(l_seen_bits.data(), 0, lnwords * sizeof(uint64_t));
                    std::memset(l_coll_bits.data(), 0, lnwords * sizeof(uint64_t));
                    uint64_t* lsb = l_seen_bits.data();
                    uint64_t* lcb = l_coll_bits.data();

                    for (size_t t = 0; t < num_threads; ++t) {
                        const auto& lst = long_ev_buckets[t].lists[b];
                        for (size_t si = 0; si < lst.slabs.size(); ++si) {
                            const LongEv32* arr = lst.slabs[si];
                            uint32_t cnt = lst.counts[si];
                            for (uint32_t i = 0; i < cnt; ++i) {
                                uint64_t f = fast::hash_u128(arr[i].hi, arr[i].lo) & lfmask;
                                size_t w = static_cast<size_t>(f >> 6);
                                uint64_t bit = 1ULL << (f & 63U);
                                lcb[w] |= lsb[w] & bit;
                                lsb[w] |= bit;
                            }
                        }
                    }

                    size_t ldcap = next_pow2(std::max<size_t>(32, total_long / 4));
                    if (l_dup_ht.size() < ldcap) l_dup_ht.resize(ldcap);
                    std::memset(l_dup_ht.data(), 0, ldcap * sizeof(uint32_t));
                    size_t ldmask = ldcap - 1;
                    l_dup_entries.clear();

                    auto grow_l_dup_ht = [&]() {
                        ldcap *= 2;
                        ldmask = ldcap - 1;
                        l_dup_ht.assign(ldcap, 0);
                        for (size_t di = 0; di < l_dup_entries.size(); ++di) {
                            size_t sl = static_cast<size_t>(fast::hash_u128(l_dup_entries[di].hi, l_dup_entries[di].lo)) & ldmask;
                            while (l_dup_ht[sl]) sl = (sl + 1) & ldmask;
                            l_dup_ht[sl] = static_cast<uint32_t>(di + 1);
                        }
                    };

                    for (size_t t = 0; t < num_threads; ++t) {
                        const auto& lst = long_ev_buckets[t].lists[b];
                        const auto& pv = ev_purchases[t][b];
                        for (size_t si = 0; si < lst.slabs.size(); ++si) {
                            const LongEv32* arr = lst.slabs[si];
                            uint32_t cnt = lst.counts[si];
                            for (uint32_t i = 0; i < cnt; ++i) {
                                uint64_t hi = arr[i].hi;
                                uint64_t lo = arr[i].lo;
                                uint64_t h = fast::hash_u128(hi, lo);
                                uint64_t f = h & lfmask;
                                if (__builtin_expect((lcb[f >> 6] & (1ULL << (f & 63U))) == 0, 1)) {
                                    uint64_t uid = arr[i].uid;
                                    if (__builtin_expect(uid == 0, 0)) continue;
                                    uint64_t meta = arr[i].meta;
                                    uint32_t pidx = static_cast<uint32_t>(meta & 0x7FFFU);
                                    uint32_t pc = __builtin_expect(pidx == 0, 1) ? 0U : pv[pidx - 1];
                                    b_uid[nb] = uid;
                                    b_tspc[nb] = ((meta >> 33) << 32) | pc;
                                    if (__builtin_expect(++nb == 64, 0)) flush_ev_batch();
                                } else {
                                    uint64_t uid = arr[i].uid;
                                    uint64_t meta = arr[i].meta;
                                    size_t sl = static_cast<size_t>(h) & ldmask;
                                    while (true) {
                                        uint32_t cur = l_dup_ht[sl];
                                        if (cur == 0) {
                                            if (__builtin_expect((l_dup_entries.size() + 1) * 10 >= ldcap * 6, 0)) {
                                                grow_l_dup_ht();
                                                sl = static_cast<size_t>(h) & ldmask;
                                                while (l_dup_ht[sl]) sl = (sl + 1) & ldmask;
                                            }
                                            l_dup_entries.push_back(LongSlot{hi, lo, uid, meta, static_cast<uint32_t>(t), 1});
                                            l_dup_ht[sl] = static_cast<uint32_t>(l_dup_entries.size());
                                            break;
                                        }
                                        auto& ls = l_dup_entries[cur - 1];
                                        if (ls.hi == hi && ls.lo == lo) {
                                            if (get_ev_ingest(uid, meta) < get_ev_ingest(ls.uid, ls.meta)) {
                                                ls.uid = uid;
                                                ls.meta = meta;
                                                ls.owner_t = static_cast<uint32_t>(t);
                                            }
                                            break;
                                        }
                                        sl = (sl + 1) & ldmask;
                                    }
                                }
                            }
                        }
                        long_ev_buckets[t].recycle_bucket(b, arena);
                    }
                    for (const auto& ls : l_dup_entries) {
                        if (ls.uid == 0) continue;
                        uint32_t pidx = static_cast<uint32_t>(ls.meta & 0x7FFFU);
                        uint32_t pc = (pidx == 0) ? 0U : ev_purchases[ls.owner_t][b][pidx - 1];
                        b_uid[nb] = ls.uid;
                        b_tspc[nb] = ((ls.meta >> 33) << 32) | pc;
                        if (__builtin_expect(++nb == 64, 0)) flush_ev_batch();
                    }
                    loc_p2_ns += static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - tl0).count());
                }
            } else {
                // General fallback path when bucket contains WideEvRow (>24d window, >255 categories, or big ingest delta).
                size_t cap = next_pow2(std::max<size_t>(32, (total_recs + total_wide) * 2));
                if (full_table.size() < cap) full_table.resize(cap);
                std::memset(full_table.data(), 0, cap * sizeof(DedupSlot));
                size_t mask = cap - 1;
                slab_ptrs.clear();
                wide_ptrs.clear();

                auto get_slot_eid = [&](uint32_t loc) -> uint64_t {
                    if ((loc & 0x80000000U) == 0) return reinterpret_cast<const uint64_t*>(slab_ptrs[(loc >> 12) - 1])[loc & 0xFFFU];
                    return wide_ptrs[loc & 0x7FFFFFFFU]->eid;
                };
                auto get_slot_ingest = [&](uint32_t loc) -> int64_t {
                    if ((loc & 0x80000000U) == 0) {
                        const EvPay* pays = reinterpret_cast<const EvPay*>(slab_ptrs[(loc >> 12) - 1] + pay_off);
                        const auto& r = pays[loc & 0xFFFU];
                        return get_ev_ingest(r.uid, r.meta);
                    }
                    return wide_ptrs[loc & 0x7FFFFFFFU]->ingest_sec;
                };
                auto kill_slot = [&](uint32_t loc) {
                    if ((loc & 0x80000000U) == 0) {
                        EvPay* pays = reinterpret_cast<EvPay*>(slab_ptrs[(loc >> 12) - 1] + pay_off);
                        pays[loc & 0xFFFU].uid = 0;
                    } else {
                        wide_ptrs[loc & 0x7FFFFFFFU]->uid = 0;
                    }
                };

                for (size_t t = 0; t < num_threads; ++t) {
                    auto& lst = ev_buckets[t].lists[b];
                    for (size_t si = 0; si < lst.slabs.size(); ++si) {
                        uint8_t* raw = lst.slabs[si];
                        const uint64_t* eids = reinterpret_cast<const uint64_t*>(raw);
                        EvPay* pays = reinterpret_cast<EvPay*>(raw + pay_off);
                        uint32_t cnt = lst.counts[si];
                        uint32_t sidx_p1 = static_cast<uint32_t>(slab_ptrs.size()) + 1;
                        slab_ptrs.push_back(raw);
                        for (uint32_t i = 0; i < cnt; ++i) {
                            uint64_t eid = eids[i];
                            uint64_t h = fast::hash_u64(eid);
                            uint32_t tag = static_cast<uint32_t>(h >> 23);
                            size_t slot = static_cast<size_t>(h) & mask;
                            while (true) {
                                DedupSlot& ds = full_table[slot];
                                if (ds.loc == 0) {
                                    ds.tag = tag;
                                    ds.loc = (sidx_p1 << 12) | i;
                                    break;
                                }
                                if (ds.tag == tag && get_slot_eid(ds.loc) == eid) {
                                    int64_t cur_ing = get_ev_ingest(pays[i].uid, pays[i].meta);
                                    int64_t prev_ing = get_slot_ingest(ds.loc);
                                    if (cur_ing < prev_ing) {
                                        kill_slot(ds.loc);
                                        ds.loc = (sidx_p1 << 12) | i;
                                    } else {
                                        pays[i].uid = 0;
                                    }
                                    break;
                                }
                                slot = (slot + 1) & mask;
                            }
                        }
                    }
                }
                for (size_t t = 0; t < num_threads; ++t) {
                    for (auto& wr : wide_ev_buckets[t][b]) {
                        if (wr.is_long_eid) continue;
                        uint32_t wloc = 0x80000000U | static_cast<uint32_t>(wide_ptrs.size());
                        wide_ptrs.push_back(&wr);
                        uint64_t h = fast::hash_u64(wr.eid);
                        uint32_t tag = static_cast<uint32_t>(h >> 23);
                        size_t slot = static_cast<size_t>(h) & mask;
                        while (true) {
                            DedupSlot& ds = full_table[slot];
                            if (ds.loc == 0) {
                                ds.tag = tag;
                                ds.loc = wloc;
                                break;
                            }
                            if (ds.tag == tag && get_slot_eid(ds.loc) == wr.eid) {
                                int64_t prev_ing = get_slot_ingest(ds.loc);
                                if (wr.ingest_sec < prev_ing) {
                                    kill_slot(ds.loc);
                                    ds.loc = wloc;
                                } else {
                                    wr.uid = 0;
                                }
                                break;
                            }
                            slot = (slot + 1) & mask;
                        }
                    }
                }

                // Dedup long compact + long wide together!
                size_t lcap = next_pow2(std::max<size_t>(32, (total_long + total_wide) * 2));
                if (long_ht.size() < lcap) long_ht.resize(lcap);
                std::memset(long_ht.data(), 0, lcap * sizeof(LongSlot));
                size_t lmask = lcap - 1;
                struct LongWideRef { bool is_wide; WideEvRow* wr; int64_t ing; };
                std::vector<LongWideRef> long_wide_refs(lcap, LongWideRef{false, nullptr, 0});
                long_used.clear();

                for (size_t t = 0; t < num_threads; ++t) {
                    const auto& lst = long_ev_buckets[t].lists[b];
                    for (size_t si = 0; si < lst.slabs.size(); ++si) {
                        const LongEv32* arr = lst.slabs[si];
                        uint32_t cnt = lst.counts[si];
                        for (uint32_t i = 0; i < cnt; ++i) {
                            size_t sl = static_cast<size_t>(fast::hash_u128(arr[i].hi, arr[i].lo)) & lmask;
                            while (true) {
                                auto& ls = long_ht[sl];
                                if (!ls.used) {
                                    int64_t ing = get_ev_ingest(arr[i].uid, arr[i].meta);
                                    ls = LongSlot{arr[i].hi, arr[i].lo, arr[i].uid, arr[i].meta,
                                                  static_cast<uint32_t>(t), 1};
                                    long_wide_refs[sl] = {false, nullptr, ing};
                                    long_used.push_back(static_cast<uint32_t>(sl));
                                    break;
                                }
                                if (ls.hi == arr[i].hi && ls.lo == arr[i].lo) {
                                    int64_t ing = get_ev_ingest(arr[i].uid, arr[i].meta);
                                    if (ing < long_wide_refs[sl].ing) {
                                        ls.uid = arr[i].uid;
                                        ls.meta = arr[i].meta;
                                        ls.owner_t = static_cast<uint32_t>(t);
                                        long_wide_refs[sl] = {false, nullptr, ing};
                                    }
                                    break;
                                }
                                sl = (sl + 1) & lmask;
                            }
                        }
                    }
                    long_ev_buckets[t].recycle_bucket(b, arena);
                }
                for (size_t t = 0; t < num_threads; ++t) {
                    for (auto& wr : wide_ev_buckets[t][b]) {
                        if (!wr.is_long_eid) continue;
                        size_t sl = static_cast<size_t>(fast::hash_u128(wr.eid, wr.eid_lo)) & lmask;
                        while (true) {
                            auto& ls = long_ht[sl];
                            if (!ls.used) {
                                ls = LongSlot{wr.eid, wr.eid_lo, wr.uid, 0, static_cast<uint32_t>(t), 1};
                                long_wide_refs[sl] = {true, &wr, wr.ingest_sec};
                                long_used.push_back(static_cast<uint32_t>(sl));
                                break;
                            }
                            if (ls.hi == wr.eid && ls.lo == wr.eid_lo) {
                                if (wr.ingest_sec < long_wide_refs[sl].ing) {
                                    ls.uid = wr.uid;
                                    long_wide_refs[sl] = {true, &wr, wr.ingest_sec};
                                }
                                break;
                            }
                            sl = (sl + 1) & lmask;
                        }
                    }
                }

                // Emit all survivors
                for (size_t t = 0; t < num_threads; ++t) {
                    const auto& lst = ev_buckets[t].lists[b];
                    const auto& pv = ev_purchases[t][b];
                    for (size_t si = 0; si < lst.slabs.size(); ++si) {
                        const EvPay* pays = reinterpret_cast<const EvPay*>(lst.slabs[si] + pay_off);
                        uint32_t cnt = lst.counts[si];
                        for (uint32_t i = 0; i < cnt; ++i) {
                            if (pays[i].uid == 0) continue;
                            uint64_t meta = pays[i].meta;
                            uint32_t pidx = static_cast<uint32_t>(meta & 0x7FFFU);
                            uint32_t pc = (pidx == 0) ? 0U : pv[pidx - 1];
                            b_uid[nb] = pays[i].uid;
                            b_tspc[nb] = ((meta >> 33) << 32) | pc;
                            if (++nb == 64) flush_ev_batch();
                        }
                    }
                    ev_buckets[t].recycle_bucket(b, arena);
                    for (const auto& wr : wide_ev_buckets[t][b]) {
                        if (wr.is_long_eid || wr.uid == 0) continue;
                        uint64_t pid_h = 0;
                        uint64_t pid = identity_map.resolve(wr.uid, pid_h);
                        my_wev[static_cast<size_t>(pid_h >> part_shift)].push_back(
                            WidePersonEv{pid, wr.ts_ms, wr.price_cents, wr.cat_id});
                    }
                }
                for (uint32_t sl : long_used) {
                    const auto& ls = long_ht[sl];
                    if (ls.uid == 0) continue;
                    if (long_wide_refs[sl].is_wide) {
                        const auto& wr = *long_wide_refs[sl].wr;
                        uint64_t pid_h = 0;
                        uint64_t pid = identity_map.resolve(wr.uid, pid_h);
                        my_wev[static_cast<size_t>(pid_h >> part_shift)].push_back(
                            WidePersonEv{pid, wr.ts_ms, wr.price_cents, wr.cat_id});
                    } else {
                        uint32_t pidx = static_cast<uint32_t>(ls.meta & 0x7FFFU);
                        uint32_t pc = (pidx == 0) ? 0U : ev_purchases[ls.owner_t][b][pidx - 1];
                        b_uid[nb] = ls.uid;
                        b_tspc[nb] = ((ls.meta >> 33) << 32) | pc;
                        if (++nb == 64) flush_ev_batch();
                    }
                }
            }
        }
        if (nb > 0) flush_ev_batch();
        my_ev.finish_all(arena);
        prof_p1_ns.fetch_add(loc_p1_ns, std::memory_order_relaxed);
        prof_p2_ns.fetch_add(loc_p2_ns, std::memory_order_relaxed);
    });
    std::fprintf(stderr, "Stage 2 in %.3fs (avg/thread: dedup_p1=%.1fms emit_p2=%.1fms)\n",
                 elapsed_sec(t_s2),
                 prof_p1_ns.load() / (1e6 * num_threads),
                 prof_p2_ns.load() / (1e6 * num_threads));

    // Stage 3: Zero-Sort 30-Min Bucket Sessionization + Small Unsorted Clicks + Whale Interval Assists + Compact Parquet!
    auto t_s3 = Clock::now();
    std::vector<std::pair<size_t, size_t>> part_order(num_parts);
    size_t grand_total_ev = 0;
    for (size_t p = 0; p < num_parts; ++p) {
        size_t cnt = 0;
        for (size_t t = 0; t < num_threads; ++t) {
            cnt += person_ev[t].lists[p].total_count + wide_person_ev[t][p].size();
        }
        part_order[p] = {cnt, p};
        grand_total_ev += cnt;
    }
    std::sort(part_order.begin(), part_order.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });

    uint32_t num_camps = static_cast<uint32_t>(catalogs.campaign_names.size());
    uint32_t num_cats = static_cast<uint32_t>(catalogs.categories.size());
    std::vector<CampDayAccum> camp_accums(num_threads);
    for (size_t t = 0; t < num_threads; ++t) camp_accums[t].init(num_camps, min_day, max_day);

    bool any_wide = has_any_wide.load(std::memory_order_relaxed);
    std::atomic<size_t> next_part{0};
    std::atomic<uint64_t> prof_grp_ns{0}, prof_proc_ns{0}, prof_pq_ns{0};
    size_t est_ud_per_thread = (grand_total_ev / (8 * num_threads)) + 1024;
    const uint32_t ndays_u32 = static_cast<uint32_t>(std::max<int64_t>(1, static_cast<int64_t>(max_day) - min_day + 1));

    team.run([&](size_t tid) {
        if (tid >= num_threads) return;
        UserDailyWriter ud_writer;
        ud_writer.open(out_dir + "/user_daily/part-" + std::to_string(tid) + ".parquet",
                       est_ud_per_thread, use_events_i32, catalogs.categories);
        CampDayAccum& cacc = camp_accums[tid];
        bool cacc_dense = cacc.is_dense();

        struct UserSlot {
            uint64_t pid;
            uint32_t uid_plus1;
            uint32_t pad;
        };
        struct WideEvItem {
            int64_t ts_ms;
            uint32_t price_cents;
            uint32_t cat_id;
        };
        struct SessionRec {
            uint64_t start_cl_off;
            int64_t rev_cents;
            int32_t day;
            uint32_t day_off;
            uint32_t ev_cnt;
        };
        struct Bucket30 {
            uint32_t mn;
            uint32_t mx;
            uint32_t cnt;
            uint32_t sid;
            int64_t rev;
        };

        std::vector<UserSlot> utable;
        std::vector<uint64_t> u_pids;
        std::vector<uint32_t> u_ev_cnt, u_cl_cnt;
        std::vector<uint32_t> ev_off, cl_off, ev_cur, cl_cur;
        std::vector<uint64_t> ev_arr64;
        std::vector<WideEvItem> tmp_wev, wev_arr;
        std::vector<uint32_t> tmp_ev_uid;
        std::vector<ClickItem> cl_arr, cl_radix_tmp;
        std::vector<uint32_t> tmp_cl_uid;

        struct PurEv {
            uint16_t bk;
            uint16_t cid;
            uint32_t price;
        };
        std::vector<Bucket30> bag(can_bucket_sess ? num_30m_buckets : 1, Bucket30{0, 0, 0, 0, 0});
        std::vector<uint16_t> touched_bks;
        touched_bks.reserve(can_bucket_sess ? num_30m_buckets : 1);
        std::vector<PurEv> pur_evs;
        pur_evs.reserve(1024);
        std::vector<int64_t> catday_rev((can_bucket_sess ? ndays_u32 : 1U) * num_cats, 0);
        std::vector<int64_t> cat_rev(num_cats, 0);
        std::vector<uint32_t> touched_cats;
        touched_cats.reserve(num_cats);

        std::vector<SessionRec> u_sessions;
        u_sessions.reserve(1024);
        std::vector<uint32_t> camp_mark(num_camps, 0);
        std::vector<uint64_t> ckey_buf;
        std::vector<uint32_t> dfirst(ndays_u32 + 1, 0);
        struct CampAssistState { uint32_t gen; uint32_t next_sess; };
        std::vector<CampAssistState> camp_state(num_camps, CampAssistState{0, 0});
        uint32_t cur_gen = 0;
        char pid_buf[40];
        uint64_t loc_grp_ns = 0, loc_proc_ns = 0;

        while (true) {
            size_t oi = next_part.fetch_add(1, std::memory_order_relaxed);
            if (oi >= num_parts) break;
            size_t p = part_order[oi].second;
            size_t total_ev = part_order[oi].first;
            if (total_ev == 0) continue;

            auto tg0 = Clock::now();
            bool part_wide = false;
            if (__builtin_expect(any_wide, 0)) {
                for (size_t t = 0; t < num_threads; ++t) {
                    if (!wide_person_ev[t][p].empty()) { part_wide = true; break; }
                }
            }

            size_t est_users = std::max<size_t>(32, total_ev / 24);
            size_t ucap = next_pow2(est_users * 2);
            if (utable.size() < ucap) utable.resize(ucap);
            std::memset(utable.data(), 0, ucap * sizeof(UserSlot));
            size_t umask = ucap - 1;
            u_pids.clear();
            u_ev_cnt.clear();

            auto grow_utable = [&]() {
                size_t ncap = (umask + 1) * 2;
                std::vector<UserSlot> nt(ncap, UserSlot{0, 0, 0});
                size_t nmask = ncap - 1;
                for (size_t i = 0; i <= umask; ++i) {
                    if (utable[i].pid == 0) continue;
                    size_t idx = static_cast<size_t>(fast::hash_u64(utable[i].pid)) & nmask;
                    while (nt[idx].pid != 0) idx = (idx + 1) & nmask;
                    nt[idx] = utable[i];
                }
                utable.swap(nt);
                umask = nmask;
            };

            auto get_or_add_user_h = [&](uint64_t pid, uint64_t h) -> uint32_t {
                size_t idx = static_cast<size_t>(h) & umask;
                while (utable[idx].pid != 0) {
                    if (__builtin_expect(utable[idx].pid == pid, 1)) return utable[idx].uid_plus1 - 1;
                    idx = (idx + 1) & umask;
                }
                if (__builtin_expect((u_pids.size() + 1) * 10 >= (umask + 1) * 6, 0)) {
                    grow_utable();
                    idx = static_cast<size_t>(h) & umask;
                    while (utable[idx].pid != 0) idx = (idx + 1) & umask;
                }
                uint32_t uid = static_cast<uint32_t>(u_pids.size());
                u_pids.push_back(pid);
                u_ev_cnt.push_back(0);
                utable[idx] = UserSlot{pid, uid + 1, 0};
                return uid;
            };

            auto get_or_add_user = [&](uint64_t pid) -> uint32_t {
                return get_or_add_user_h(pid, fast::hash_u64(pid));
            };

            auto find_user_h = [&](uint64_t pid, uint64_t h) -> int64_t {
                size_t idx = static_cast<size_t>(h) & umask;
                while (utable[idx].pid != 0) {
                    if (__builtin_expect(utable[idx].pid == pid, 1)) return static_cast<int64_t>(utable[idx].uid_plus1 - 1);
                    idx = (idx + 1) & umask;
                }
                return -1;
            };

            if (__builtin_expect(!part_wide, 1)) {
                if (tmp_ev_uid.size() < total_ev) tmp_ev_uid.resize(total_ev);
                uint32_t* uid_out = tmp_ev_uid.data();
                size_t n_ev = 0;
                uint64_t h_ring[16];

                for (size_t t = 0; t < num_threads; ++t) {
                    const auto& lst = person_ev[t].lists[p];
                    for (size_t si = 0; si < lst.slabs.size(); ++si) {
                        const PersonEv16* arr = lst.slabs[si];
                        uint32_t cnt = lst.counts[si];
                        uint32_t pre = std::min<uint32_t>(8, cnt);
                        for (uint32_t j = 0; j < pre; ++j) {
                            uint64_t nh = fast::hash_u64(arr[j].pid);
                            h_ring[j] = nh;
                            __builtin_prefetch(&utable[static_cast<size_t>(nh) & umask], 0, 1);
                        }
                        for (uint32_t i = 0; i < cnt; ++i) {
                            uint64_t h = h_ring[i & 15];
                            if (__builtin_expect(i + 8 < cnt, 1)) {
                                uint64_t nh = fast::hash_u64(arr[i + 8].pid);
                                h_ring[(i + 8) & 15] = nh;
                                __builtin_prefetch(&utable[static_cast<size_t>(nh) & umask], 0, 1);
                            }
                            uint32_t uid = get_or_add_user_h(arr[i].pid, h);
                            u_ev_cnt[uid]++;
                            uid_out[n_ev++] = uid;
                        }
                    }
                }
                size_t num_users = u_pids.size();
                ev_off.resize(num_users + 1);
                ev_off[0] = 0;
                for (size_t u = 0; u < num_users; ++u) ev_off[u + 1] = ev_off[u] + u_ev_cnt[u];
                ev_cur.assign(ev_off.begin(), ev_off.begin() + num_users);
                if (ev_arr64.size() < n_ev) ev_arr64.resize(n_ev);
                uint64_t* dst = ev_arr64.data();
                uint32_t* cur = ev_cur.data();
                size_t k = 0;
                for (size_t t = 0; t < num_threads; ++t) {
                    const auto& lst = person_ev[t].lists[p];
                    for (size_t si = 0; si < lst.slabs.size(); ++si) {
                        const PersonEv16* arr = lst.slabs[si];
                        uint32_t cnt = lst.counts[si];
                        for (uint32_t i = 0; i < cnt; ++i) {
                            dst[cur[uid_out[k++]]++] = arr[i].ts_pc;
                        }
                    }
                }
            } else {
                tmp_ev_uid.clear();
                tmp_wev.clear();
                tmp_ev_uid.reserve(total_ev);
                tmp_wev.reserve(total_ev);
                for (size_t t = 0; t < num_threads; ++t) {
                    const auto& lst = person_ev[t].lists[p];
                    for (size_t si = 0; si < lst.slabs.size(); ++si) {
                        const PersonEv16* arr = lst.slabs[si];
                        uint32_t cnt = lst.counts[si];
                        for (uint32_t i = 0; i < cnt; ++i) {
                            uint32_t uid = get_or_add_user(arr[i].pid);
                            u_ev_cnt[uid]++;
                            tmp_ev_uid.push_back(uid);
                            uint64_t v = arr[i].ts_pc;
                            tmp_wev.push_back(WideEvItem{
                                window_start + static_cast<int64_t>(v >> 32),
                                static_cast<uint32_t>(v) & 0x00FFFFFFU,
                                static_cast<uint32_t>(v) >> 24
                            });
                        }
                    }
                    for (const auto& w : wide_person_ev[t][p]) {
                        uint32_t uid = get_or_add_user(w.pid);
                        u_ev_cnt[uid]++;
                        tmp_ev_uid.push_back(uid);
                        tmp_wev.push_back(WideEvItem{w.ts_ms, w.price_cents, w.cat_id});
                    }
                }
                size_t num_users = u_pids.size();
                ev_off.resize(num_users + 1);
                ev_off[0] = 0;
                for (size_t u = 0; u < num_users; ++u) ev_off[u + 1] = ev_off[u] + u_ev_cnt[u];
                ev_cur.assign(ev_off.begin(), ev_off.begin() + num_users);
                wev_arr.resize(tmp_wev.size());
                for (size_t i = 0; i < tmp_wev.size(); ++i) {
                    wev_arr[ev_cur[tmp_ev_uid[i]]++] = tmp_wev[i];
                }
            }

            size_t num_users = u_pids.size();
            u_cl_cnt.assign(num_users, 0);
            size_t total_cl = 0;
            for (size_t t = 0; t < num_threads; ++t) total_cl += person_cl[t].lists[p].total_count;
            if (tmp_cl_uid.size() < total_cl) tmp_cl_uid.resize(total_cl);
            uint32_t* cl_uid_out = tmp_cl_uid.data();
            size_t cl_pos = 0;
            size_t n_cl = 0;
            uint64_t ch_ring[16];

            for (size_t t = 0; t < num_threads; ++t) {
                const auto& lst = person_cl[t].lists[p];
                for (size_t si = 0; si < lst.slabs.size(); ++si) {
                    const PersonClick* arr = lst.slabs[si];
                    uint32_t cnt = lst.counts[si];
                    uint32_t pre = std::min<uint32_t>(8, cnt);
                    for (uint32_t j = 0; j < pre; ++j) {
                        uint64_t nh = fast::hash_u64(arr[j].pid);
                        ch_ring[j] = nh;
                        __builtin_prefetch(&utable[static_cast<size_t>(nh) & umask], 0, 1);
                    }
                    for (uint32_t i = 0; i < cnt; ++i) {
                        uint64_t h = ch_ring[i & 15];
                        if (__builtin_expect(i + 8 < cnt, 1)) {
                            uint64_t nh = fast::hash_u64(arr[i + 8].pid);
                            ch_ring[(i + 8) & 15] = nh;
                            __builtin_prefetch(&utable[static_cast<size_t>(nh) & umask], 0, 1);
                        }
                        int64_t uid = find_user_h(arr[i].pid, h);
                        if (__builtin_expect(uid >= 0, 1)) {
                            u_cl_cnt[static_cast<size_t>(uid)]++;
                            cl_uid_out[cl_pos++] = static_cast<uint32_t>(uid);
                            n_cl++;
                        } else {
                            cl_uid_out[cl_pos++] = UINT32_MAX;
                        }
                    }
                }
            }

            cl_off.resize(num_users + 1);
            cl_off[0] = 0;
            for (size_t u = 0; u < num_users; ++u) cl_off[u + 1] = cl_off[u] + u_cl_cnt[u];
            cl_cur.assign(cl_off.begin(), cl_off.begin() + num_users);
            if (cl_arr.size() < n_cl) cl_arr.resize(n_cl);
            ClickItem* cl_dst = cl_arr.data();
            uint32_t* ccur = cl_cur.data();
            size_t ck = 0;
            for (size_t t = 0; t < num_threads; ++t) {
                const auto& lst = person_cl[t].lists[p];
                for (size_t si = 0; si < lst.slabs.size(); ++si) {
                    const PersonClick* arr = lst.slabs[si];
                    uint32_t cnt = lst.counts[si];
                    for (uint32_t i = 0; i < cnt; ++i) {
                        uint32_t uid = cl_uid_out[ck++];
                        if (__builtin_expect(uid != UINT32_MAX, 1)) {
                            cl_dst[ccur[uid]++] = ClickItem{arr[i].ts_camp, arr[i].cid};
                        }
                    }
                }
            }

            auto tg1 = Clock::now();
            loc_grp_ns += static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(tg1 - tg0).count());

            const bool use_30m_nosort = can_bucket_sess && !part_wide;
            for (size_t u = 0; u < num_users; ++u) {
                u_sessions.clear();
                uint64_t pid_key = u_pids[u];
                std::string_view pid_sv = kenv.key_to_sv(pid_key, 'u', pid_buf);

                if (__builtin_expect(use_30m_nosort, 1)) {
                    // ZERO-SORT 30-MINUTE BUCKET SESSIONIZATION (exact O(N) even on 1M-event whales!)
                    const uint64_t* eb = ev_arr64.data() + ev_off[u];
                    size_t en = static_cast<size_t>(ev_off[u + 1] - ev_off[u]);
                    if (__builtin_expect(en == 1, 0)) {
                        uint64_t ev = eb[0];
                        uint32_t ts = static_cast<uint32_t>(ev >> 32);
                        uint32_t price = static_cast<uint32_t>(ev) & 0x00FFFFFFU;
                        int32_t top_cat = (price > 0) ? static_cast<int32_t>(static_cast<uint32_t>(ev) >> 24) : -1;
                        uint32_t doff = (static_cast<int64_t>(ts) < day0_end_off)
                                            ? 0U
                                            : static_cast<uint32_t>((static_cast<int64_t>(ts) - day0_end_off) / DAY_MS + 1);
                        int32_t d = min_day + static_cast<int32_t>(doff);
                        u_sessions.push_back(SessionRec{
                            static_cast<uint64_t>(ts) + static_cast<uint64_t>(WEEK_MS),
                            static_cast<int64_t>(price), d, doff, 1U
                        });
                        ud_writer.add_row(pid_key, pid_sv, d, 1, 1, static_cast<int64_t>(price), top_cat);
                    } else {
                        touched_bks.clear();
                        pur_evs.clear();
                        Bucket30* bptr = bag.data();
                        for (size_t i = 0; i < en; ++i) {
                            uint64_t ev = eb[i];
                            uint32_t ts = static_cast<uint32_t>(ev >> 32);
                            uint32_t bk = ts / SESSION_GAP_U32;
                            auto& a = bptr[bk];
                            if (a.cnt == 0) {
                                touched_bks.push_back(static_cast<uint16_t>(bk));
                                a.mn = ts;
                                a.mx = ts;
                            } else {
                                if (ts < a.mn) a.mn = ts;
                                if (ts > a.mx) a.mx = ts;
                            }
                            a.cnt++;
                            uint32_t price = static_cast<uint32_t>(ev) & 0x00FFFFFFU;
                            if (__builtin_expect(price > 0, 0)) {
                                a.rev += price;
                                pur_evs.push_back(PurEv{
                                    static_cast<uint16_t>(bk),
                                    static_cast<uint16_t>(static_cast<uint32_t>(ev) >> 24),
                                    price
                                });
                            }
                        }
                        size_t ntb = touched_bks.size();
                        sort_u16(touched_bks.data(), ntb);

                        int32_t cur_day = min_day;
                        int64_t next_day_off = day0_end_off;
                        uint32_t prev_mx = 0;
                        for (size_t k = 0; k < ntb; ++k) {
                            auto& a = bptr[touched_bks[k]];
                            if (k == 0 || a.mn - prev_mx > SESSION_GAP_U32) {
                                while (static_cast<int64_t>(a.mn) >= next_day_off) {
                                    cur_day++;
                                    next_day_off += DAY_MS;
                                }
                                u_sessions.push_back(SessionRec{
                                    static_cast<uint64_t>(a.mn) + static_cast<uint64_t>(WEEK_MS),
                                    a.rev, cur_day, static_cast<uint32_t>(cur_day - min_day), a.cnt
                                });
                            } else {
                                auto& last_s = u_sessions.back();
                                last_s.ev_cnt += a.cnt;
                                last_s.rev_cents += a.rev;
                            }
                            a.sid = static_cast<uint32_t>(u_sessions.size() - 1);
                            prev_mx = a.mx;
                        }
                        const size_t npur = pur_evs.size();
                        const int32_t single_pur_cat = (npur == 1) ? static_cast<int32_t>(pur_evs[0].cid) : -1;
                        if (npur >= 2) {
                            for (const auto& pe : pur_evs) {
                                uint32_t day_off = u_sessions[bptr[pe.bk].sid].day_off;
                                catday_rev[static_cast<size_t>(day_off) * num_cats + pe.cid] += pe.price;
                            }
                        }
                        for (size_t k = 0; k < ntb; ++k) {
                            auto& a = bptr[touched_bks[k]];
                            a.cnt = 0;
                            a.rev = 0;
                        }

                        // Emit user_daily rows grouped by day!
                        size_t S_tot = u_sessions.size();
                        for (size_t si = 0; si < S_tot; ) {
                            int32_t d = u_sessions[si].day;
                            uint32_t doff = u_sessions[si].day_off;
                            int32_t dsess = 0;
                            int64_t devn = 0, drev = 0;
                            while (si < S_tot && u_sessions[si].day == d) {
                                dsess++;
                                devn += u_sessions[si].ev_cnt;
                                drev += u_sessions[si].rev_cents;
                                ++si;
                            }
                            int32_t top_cat = -1;
                            if (drev > 0) {
                                if (npur == 1) {
                                    top_cat = single_pur_cat;
                                } else {
                                    int64_t* cs = catday_rev.data() + static_cast<size_t>(doff) * num_cats;
                                    int64_t bv = 0;
                                    for (uint32_t cid = 0; cid < num_cats; ++cid) {
                                        if (cs[cid] > bv) {
                                            bv = cs[cid];
                                            top_cat = static_cast<int32_t>(cid);
                                        }
                                        cs[cid] = 0;
                                    }
                                }
                            }
                            ud_writer.add_row(pid_key, pid_sv, d, dsess, devn, drev, top_cat);
                        }
                    }
                } else {
                    // General exact sort fallback (window > 42 days or > 255 categories).
                    touched_cats.clear();
                    auto flush_day = [&](int32_t cur_day, int32_t day_sessions, int64_t day_events, int64_t day_rev) {
                        int32_t top_cat = -1;
                        if (day_rev > 0) {
                            int64_t best_rev = -1;
                            uint32_t best_cat = 0;
                            for (uint32_t cid : touched_cats) {
                                int64_t r = cat_rev[cid];
                                if (r > best_rev || (r == best_rev && cid < best_cat)) {
                                    best_rev = r;
                                    best_cat = cid;
                                }
                                cat_rev[cid] = 0;
                            }
                            top_cat = static_cast<int32_t>(best_cat);
                        }
                        touched_cats.clear();
                        ud_writer.add_row(pid_key, pid_sv, cur_day, day_sessions, day_events, day_rev, top_cat);
                    };

                    if (!part_wide) {
                        uint64_t* eb = ev_arr64.data() + ev_off[u];
                        size_t en = static_cast<size_t>(ev_off[u + 1] - ev_off[u]);
                        if (en > 1) std::sort(eb, eb + en);
                        int64_t sess_off = static_cast<int64_t>(eb[0] >> 32);
                        int64_t prev_off = sess_off;
                        int32_t sess_day = fast::floor_div_day(window_start + sess_off);
                        int64_t sess_rev = 0;
                        int32_t cur_day = sess_day, day_sessions = 1;
                        int64_t day_events = 0, day_rev = 0;
                        for (size_t i = 0; i < en; ++i) {
                            uint64_t ev = eb[i];
                            int64_t off = static_cast<int64_t>(ev >> 32);
                            if (i > 0 && off - prev_off > SESSION_GAP_MS) {
                                u_sessions.push_back(SessionRec{static_cast<uint64_t>(sess_off + WEEK_MS), sess_rev, sess_day, static_cast<uint32_t>(sess_day - min_day), 0});
                                sess_off = off;
                                sess_day = fast::floor_div_day(window_start + sess_off);
                                sess_rev = 0;
                                if (sess_day != cur_day) {
                                    flush_day(cur_day, day_sessions, day_events, day_rev);
                                    cur_day = sess_day; day_sessions = 1; day_events = 0; day_rev = 0;
                                } else {
                                    day_sessions++;
                                }
                            }
                            prev_off = off;
                            day_events++;
                            uint32_t price = static_cast<uint32_t>(ev) & 0x00FFFFFFU;
                            if (price > 0) {
                                uint32_t cid = static_cast<uint32_t>(ev) >> 24;
                                sess_rev += price; day_rev += price;
                                if (cat_rev[cid] == 0) touched_cats.push_back(cid);
                                cat_rev[cid] += price;
                            }
                        }
                        u_sessions.push_back(SessionRec{static_cast<uint64_t>(sess_off + WEEK_MS), sess_rev, sess_day, static_cast<uint32_t>(sess_day - min_day), 0});
                        flush_day(cur_day, day_sessions, day_events, day_rev);
                    } else {
                        WideEvItem* eb = wev_arr.data() + ev_off[u];
                        WideEvItem* ee = wev_arr.data() + ev_off[u + 1];
                        size_t en = static_cast<size_t>(ee - eb);
                        if (en > 1) {
                            std::sort(eb, ee, [](const WideEvItem& a, const WideEvItem& b) { return a.ts_ms < b.ts_ms; });
                        }
                        int64_t sess_start = eb[0].ts_ms, prev_ts = eb[0].ts_ms;
                        int32_t sess_day = fast::floor_div_day(sess_start);
                        int64_t sess_rev = 0;
                        int32_t cur_day = sess_day, day_sessions = 1;
                        int64_t day_events = 0, day_rev = 0;
                        for (size_t i = 0; i < en; ++i) {
                            const auto& e = eb[i];
                            if (i > 0 && e.ts_ms - prev_ts > SESSION_GAP_MS) {
                                u_sessions.push_back(SessionRec{static_cast<uint64_t>(sess_start - click_start), sess_rev, sess_day, static_cast<uint32_t>(sess_day - min_day), 0});
                                sess_start = e.ts_ms;
                                sess_day = fast::floor_div_day(sess_start);
                                sess_rev = 0;
                                if (sess_day != cur_day) {
                                    flush_day(cur_day, day_sessions, day_events, day_rev);
                                    cur_day = sess_day; day_sessions = 1; day_events = 0; day_rev = 0;
                                } else {
                                    day_sessions++;
                                }
                            }
                            prev_ts = e.ts_ms;
                            day_events++;
                            if (e.price_cents > 0) {
                                sess_rev += e.price_cents; day_rev += e.price_cents;
                                if (cat_rev[e.cat_id] == 0) touched_cats.push_back(e.cat_id);
                                cat_rev[e.cat_id] += e.price_cents;
                            }
                        }
                        u_sessions.push_back(SessionRec{static_cast<uint64_t>(sess_start - click_start), sess_rev, sess_day, static_cast<uint32_t>(sess_day - min_day), 0});
                        flush_day(cur_day, day_sessions, day_events, day_rev);
                    }
                }

                // Attribution & Assists!
                size_t S = u_sessions.size();
                ClickItem* cb_raw = cl_arr.data() + cl_off[u];
                size_t C = static_cast<size_t>(cl_off[u + 1] - cl_off[u]);
                const bool fast_dense = cacc_dense && !part_wide;
                constexpr uint64_t WEEK_U64 = static_cast<uint64_t>(WEEK_MS);

                if (C == 0) {
                    if (__builtin_expect(fast_dense, 1)) {
                        for (size_t si = 0; si < S; ++si) cacc.add_session_dense(0, u_sessions[si].day_off, u_sessions[si].rev_cents);
                    } else {
                        for (size_t si = 0; si < S; ++si) cacc.add_session(0, u_sessions[si].day, u_sessions[si].rev_cents);
                    }
                    continue;
                }

                if (C == 1) {
                    uint64_t cts_off = cb_raw[0].ts_camp >> 20;
                    uint32_t camp = static_cast<uint32_t>(cb_raw[0].ts_camp & 0xFFFFFU);
                    uint64_t hi_off = cts_off + WEEK_U64;
                    if (__builtin_expect(fast_dense, 1)) {
                        for (size_t si = 0; si < S; ++si) {
                            uint64_t soff = u_sessions[si].start_cl_off;
                            if (soff >= cts_off && soff <= hi_off) {
                                cacc.add_session_and_assist_dense(camp, u_sessions[si].day_off, u_sessions[si].rev_cents);
                            } else {
                                cacc.add_session_dense(0, u_sessions[si].day_off, u_sessions[si].rev_cents);
                            }
                        }
                    } else {
                        for (size_t si = 0; si < S; ++si) {
                            uint64_t soff = u_sessions[si].start_cl_off;
                            if (soff >= cts_off && soff <= hi_off) {
                                cacc.add_session(camp, u_sessions[si].day, u_sessions[si].rev_cents);
                                cacc.add_assist(camp, u_sessions[si].day);
                            } else {
                                cacc.add_session(0, u_sessions[si].day, u_sessions[si].rev_cents);
                            }
                        }
                    }
                    continue;
                }

                // Small (S, C) unsorted direct scan (covers 90%+ of users with clicks -> ZERO click sort!)
                if (S * C <= 64 + 8 * (S + C)) {
                    if (__builtin_expect(C <= 8, 1)) {
                        uint32_t seen_c[8];
                        for (size_t si = 0; si < S; ++si) {
                            uint64_t soff = u_sessions[si].start_cl_off;
                            uint64_t lo_off = soff - WEEK_U64;
                            uint32_t nseen = 0;
                            uint64_t best_ts = 0;
                            uint64_t best_cid = 0;
                            uint32_t best_camp = 0;
                            bool has_best = false;
                            for (size_t qi = 0; qi < C; ++qi) {
                                uint64_t tc = cb_raw[qi].ts_camp;
                                uint64_t t = tc >> 20;
                                if (t >= lo_off && t <= soff) {
                                    uint32_t camp = static_cast<uint32_t>(tc & 0xFFFFFU);
                                    bool dup = false;
                                    for (uint32_t m = 0; m < nseen; ++m) {
                                        if (seen_c[m] == camp) { dup = true; break; }
                                    }
                                    if (!dup) {
                                        seen_c[nseen++] = camp;
                                        if (__builtin_expect(fast_dense, 1)) cacc.add_assist_dense(camp, u_sessions[si].day_off);
                                        else cacc.add_assist(camp, u_sessions[si].day);
                                    }
                                    if (!has_best || t > best_ts) {
                                        has_best = true;
                                        best_ts = t;
                                        best_cid = cb_raw[qi].cid;
                                        best_camp = camp;
                                    } else if (__builtin_expect(t == best_ts, 0)) {
                                        uint64_t cid = cb_raw[qi].cid;
                                        if (cid == best_cid) {
                                            if (fast::is_hex_key(cid) && camp != best_camp) {
                                                kenv.long_users.need_exact.store(true, std::memory_order_relaxed);
                                            }
                                        } else if (kenv.key_less(cid, best_cid, 'k')) {
                                            best_cid = cid;
                                            best_camp = camp;
                                        }
                                    }
                                }
                            }
                            uint32_t credited = has_best ? best_camp : 0U;
                            if (__builtin_expect(fast_dense, 1)) cacc.add_session_dense(credited, u_sessions[si].day_off, u_sessions[si].rev_cents);
                            else cacc.add_session(credited, u_sessions[si].day, u_sessions[si].rev_cents);
                        }
                    } else {
                        for (size_t si = 0; si < S; ++si) {
                            uint64_t soff = u_sessions[si].start_cl_off;
                            uint64_t lo_off = soff - WEEK_U64;
                            if (++cur_gen == 0) {
                                std::fill(camp_mark.begin(), camp_mark.end(), 0);
                                cur_gen = 1;
                            }
                            uint64_t best_ts = 0;
                            uint64_t best_cid = 0;
                            uint32_t best_camp = 0;
                            bool has_best = false;
                            for (size_t qi = 0; qi < C; ++qi) {
                                uint64_t tc = cb_raw[qi].ts_camp;
                                uint64_t t = tc >> 20;
                                if (t >= lo_off && t <= soff) {
                                    uint32_t camp = static_cast<uint32_t>(tc & 0xFFFFFU);
                                    if (camp_mark[camp] != cur_gen) {
                                        camp_mark[camp] = cur_gen;
                                        if (__builtin_expect(fast_dense, 1)) cacc.add_assist_dense(camp, u_sessions[si].day_off);
                                        else cacc.add_assist(camp, u_sessions[si].day);
                                    }
                                    if (!has_best || t > best_ts) {
                                        has_best = true;
                                        best_ts = t;
                                        best_cid = cb_raw[qi].cid;
                                        best_camp = camp;
                                    } else if (__builtin_expect(t == best_ts, 0)) {
                                        uint64_t cid = cb_raw[qi].cid;
                                        if (cid == best_cid) {
                                            if (fast::is_hex_key(cid) && camp != best_camp) {
                                                kenv.long_users.need_exact.store(true, std::memory_order_relaxed);
                                            }
                                        } else if (kenv.key_less(cid, best_cid, 'k')) {
                                            best_cid = cid;
                                            best_camp = camp;
                                        }
                                    }
                                }
                            }
                            uint32_t credited = has_best ? best_camp : 0U;
                            if (__builtin_expect(fast_dense, 1)) cacc.add_session_dense(credited, u_sessions[si].day_off, u_sessions[si].rev_cents);
                            else cacc.add_session(credited, u_sessions[si].day, u_sessions[si].rev_cents);
                        }
                    }
                    continue;
                }

                // Large users & whales: sort clicks (larger click_id first on tie -> last qualifying wins!),
                // two-pointer credit, and merged per-campaign intervals for O(C log C + C * ndays) assists!
                const ClickItem* cb = sort_clicks(cb_raw, C, kenv, cl_radix_tmp);
                if (__builtin_expect(u_sessions[S - 1].start_cl_off < (cb[0].ts_camp >> 20), 0)) {
                    if (__builtin_expect(fast_dense, 1)) {
                        for (size_t si = 0; si < S; ++si) cacc.add_session_dense(0, u_sessions[si].day_off, u_sessions[si].rev_cents);
                    } else {
                        for (size_t si = 0; si < S; ++si) cacc.add_session(0, u_sessions[si].day, u_sessions[si].rev_cents);
                    }
                    continue;
                }

                size_t hi = 0;
                for (size_t si = 0; si < S; ++si) {
                    uint64_t soff = u_sessions[si].start_cl_off;
                    while (hi < C && (cb[hi].ts_camp >> 20) <= soff) ++hi;
                    uint32_t credited = (hi > 0 && soff - (cb[hi - 1].ts_camp >> 20) <= WEEK_U64)
                                            ? static_cast<uint32_t>(cb[hi - 1].ts_camp & 0xFFFFFU)
                                            : 0U;
                    if (__builtin_expect(fast_dense, 1)) cacc.add_session_dense(credited, u_sessions[si].day_off, u_sessions[si].rev_cents);
                    else cacc.add_session(credited, u_sessions[si].day, u_sessions[si].rev_cents);
                }

                if (__builtin_expect(fast_dense && S >= 32, 1)) {
                    // Whale interval-merge assists: O(C log C + C * ndays) instead of O(C * S)!
                    if (ckey_buf.size() < C) ckey_buf.resize(C);
                    uint64_t* keys_c = ckey_buf.data();
                    for (size_t i = 0; i < C; ++i) {
                        uint64_t tc = cb[i].ts_camp;
                        keys_c[i] = ((tc & 0xFFFFFU) << 44) | (tc >> 20);
                    }
                    std::sort(keys_c, keys_c + C);

                    uint32_t k = 0;
                    for (uint32_t d = 0; d <= ndays_u32; ++d) {
                        while (k < S && u_sessions[k].day_off < d) ++k;
                        dfirst[d] = k;
                    }
                    auto starts_lb = [&](uint64_t v) -> uint32_t {
                        uint32_t base = 0, len = static_cast<uint32_t>(S);
                        while (len > 1) {
                            uint32_t half = len >> 1;
                            base = (u_sessions[base + half - 1].start_cl_off < v) ? (base + half) : base;
                            len -= half;
                        }
                        return (len == 1 && u_sessions[base].start_cl_off < v) ? (base + 1) : base;
                    };

                    size_t q = 0;
                    while (q < C) {
                        uint32_t cp = static_cast<uint32_t>(keys_c[q] >> 44);
                        uint64_t L = keys_c[q] & 0xFFFFFFFFFFFULL;
                        uint64_t R = L + WEEK_U64;
                        ++q;
                        auto flush_iv = [&](uint64_t l, uint64_t r) {
                            uint32_t a0 = starts_lb(l);
                            uint32_t b0 = starts_lb(r + 1);
                            if (a0 >= b0) return;
                            uint32_t d_lo = u_sessions[a0].day_off;
                            uint32_t d_hi = u_sessions[b0 - 1].day_off;
                            for (uint32_t d = d_lo; d <= d_hi; ++d) {
                                uint32_t lo2 = std::max(a0, dfirst[d]);
                                uint32_t hi2 = std::min(b0, dfirst[d + 1]);
                                if (hi2 > lo2) cacc.add_assist_dense(cp, d, hi2 - lo2);
                            }
                        };
                        while (q < C && static_cast<uint32_t>(keys_c[q] >> 44) == cp) {
                            uint64_t l2 = keys_c[q] & 0xFFFFFFFFFFFULL;
                            uint64_t r2 = l2 + WEEK_U64;
                            if (l2 <= R + 1) {
                                if (r2 > R) R = r2;
                            } else {
                                flush_iv(L, R);
                                L = l2;
                                R = r2;
                            }
                            ++q;
                        }
                        flush_iv(L, R);
                    }
                } else {
                    if (++cur_gen == 0) {
                        std::fill(camp_state.begin(), camp_state.end(), CampAssistState{0, 0});
                        cur_gen = 1;
                    }
                    size_t si_lo = 0;
                    uint64_t prev_tc = UINT64_MAX;
                    for (size_t i = 0; i < C; ++i) {
                        if (__builtin_expect(cb[i].ts_camp == prev_tc, 0)) continue;
                        prev_tc = cb[i].ts_camp;
                        uint64_t cts_off = cb[i].ts_camp >> 20;
                        while (si_lo < S && u_sessions[si_lo].start_cl_off < cts_off) si_lo++;
                        if (si_lo >= S) break;
                        uint32_t camp = static_cast<uint32_t>(cb[i].ts_camp & 0xFFFFFU);
                        auto& cs = camp_state[camp];
                        size_t k = (cs.gen == cur_gen) ? std::max<size_t>(si_lo, cs.next_sess) : si_lo;
                        uint64_t hi_off = cts_off + WEEK_U64;
                        if (__builtin_expect(fast_dense, 1)) {
                            while (k < S && u_sessions[k].start_cl_off <= hi_off) {
                                cacc.add_assist_dense(camp, u_sessions[k].day_off);
                                k++;
                            }
                        } else {
                            while (k < S && u_sessions[k].start_cl_off <= hi_off) {
                                cacc.add_assist(camp, u_sessions[k].day);
                                k++;
                            }
                        }
                        cs.gen = cur_gen;
                        cs.next_sess = static_cast<uint32_t>(k);
                    }
                }
            }
            loc_proc_ns += static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - tg1).count());
        }

        auto tpq = Clock::now();
        ud_writer.close();
        prof_pq_ns.fetch_add(
            static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - tpq).count()),
            std::memory_order_relaxed);
        prof_grp_ns.fetch_add(loc_grp_ns, std::memory_order_relaxed);
        prof_proc_ns.fetch_add(loc_proc_ns, std::memory_order_relaxed);
    });

    if (__builtin_expect(!kenv.long_users.exact_mode && kenv.long_users.need_exact.load(std::memory_order_relaxed), 0)) {
        // Adversarial click_id 55-bit prefix collision: exact re-run overwrites outputs cleanly!
        setenv("EXACT_LONG", "1", 1);
#if defined(__linux__)
        execv("/proc/self/exe", argv);
#endif
        execv(argv[0], argv);
        return 1;
    }

    auto t_merge = Clock::now();
    auto merged_camps = CampDayAccum::merge_all(camp_accums);
    if (!write_campaign_daily(out_dir + "/campaign_daily/part-0.parquet", catalogs, merged_camps)) {
        std::fprintf(stderr, "Failed to write campaign_daily\n");
        return 1;
    }

    std::fprintf(stderr, "Stage 3 in %.3fs (avg/thread: grp=%.1fms proc+ud=%.1fms close=%.1fms | merge+cd=%.1fms)\n",
                 elapsed_sec(t_s3),
                 prof_grp_ns.load() / (1e6 * num_threads),
                 prof_proc_ns.load() / (1e6 * num_threads),
                 prof_pq_ns.load() / (1e6 * num_threads),
                 elapsed_sec(t_merge) * 1000.0);
    std::fprintf(stderr, "Completed pipeline in %.3fs\n", elapsed_sec(t_total));
    std::fflush(stderr);
#if defined(PGO_TRAIN)
    return 0;
#else
    _exit(0);
#endif
}
