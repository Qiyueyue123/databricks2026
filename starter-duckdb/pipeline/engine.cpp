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

#include "csv_fast.hpp"
#include "parquet_writer.hpp"

namespace {

constexpr int64_t DAY_MS = 86400000LL;
constexpr int64_t WEEK_MS = 604800000LL;
constexpr int64_t SESSION_GAP_MS = 1800000LL;
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

static bool g_use_huge = false;

static inline void* alloc_huge_raw(size_t bytes) {
#if defined(__linux__)
    if (g_use_huge && bytes >= 2 * 1024 * 1024) {
        void* p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (p != MAP_FAILED) {
#if defined(MADV_HUGEPAGE)
            madvise(p, bytes, MADV_HUGEPAGE);
#endif
            return p;
        }
    }
#endif
    return std::malloc(bytes);
}

// Persistent low-latency futex-backed worker pool (C++20 atomic::wait -> ZERO spin-yield CPU stealing!).
class ThreadTeam {
    size_t n_ = 1;
    std::vector<std::thread> workers_;
    std::atomic<uint64_t> epoch_{0};
    std::atomic<uint64_t> done_{0};
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
        uint64_t seen = 0;
        while (true) {
            uint64_t ep = epoch_.load(std::memory_order_acquire);
            if (ep == seen) {
                for (int s = 0; s < 300; ++s) {
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
            if (done_.fetch_add(1, std::memory_order_release) + 1 == static_cast<uint64_t>(n_ - 1)) {
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
        uint64_t target = static_cast<uint64_t>(n_ - 1);
        uint64_t d = done_.load(std::memory_order_acquire);
        if (d < target) {
            for (int s = 0; s < 300; ++s) {
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

class FallbackInterner {
    mutable std::mutex mu_;
    struct Slot {
        uint64_t hash = 0;
        uint32_t idx = 0;
    };
    std::vector<std::string> strs_;
    std::vector<Slot> slots_;
    size_t mask_ = 0;

    void grow() {
        size_t ncap = slots_.empty() ? 64 : slots_.size() * 2;
        std::vector<Slot> ns(ncap);
        size_t nmask = ncap - 1;
        for (const auto& s : slots_) {
            if (!s.hash) continue;
            size_t i = static_cast<size_t>(s.hash) & nmask;
            while (ns[i].hash) i = (i + 1) & nmask;
            ns[i] = s;
        }
        slots_.swap(ns);
        mask_ = nmask;
    }

public:
    uint64_t intern(const char* p, size_t len) {
        std::lock_guard<std::mutex> lk(mu_);
        if ((strs_.size() + 1) * 10 >= slots_.size() * 6) grow();
        uint64_t h = fast::hash_bytes(p, len) | 1ULL;
        size_t i = static_cast<size_t>(h) & mask_;
        while (slots_[i].hash) {
            if (slots_[i].hash == h) {
                const auto& s = strs_[slots_[i].idx];
                if (s.size() == len && std::memcmp(s.data(), p, len) == 0) {
                    return (1ULL << 63) | slots_[i].idx;
                }
            }
            i = (i + 1) & mask_;
        }
        uint32_t idx = static_cast<uint32_t>(strs_.size());
        strs_.emplace_back(p, len);
        slots_[i] = Slot{h, idx};
        return (1ULL << 63) | idx;
    }

    std::string_view get_unlocked(uint64_t key) const {
        return strs_[static_cast<size_t>(key & 0x7FFFFFFFFFFFFFFFULL)];
    }

    std::string_view key_to_sv(uint64_t key, char (&buf)[16]) const {
        if (__builtin_expect((key >> 63) == 0, 1)) {
            size_t n = fast::unpack7(key, buf);
            return std::string_view(buf, n);
        }
        return get_unlocked(key);
    }

    bool key_less(uint64_t a, uint64_t b) const {
        if (__builtin_expect(((a | b) >> 63) == 0, 1)) {
            return a < b;
        }
        // The identity build can compare fallback IDs while event workers intern them.
        std::lock_guard<std::mutex> lock(mu_);
        char ba[16], bb[16];
        auto sa = key_to_sv(a, ba);
        auto sb = key_to_sv(b, bb);
        return fast::cmp_bytes(sa.data(), sa.size(), sb.data(), sb.size()) < 0;
    }
};

static inline uint64_t encode_key(const char* p, size_t len, FallbackInterner& fb) {
    uint64_t k = fast::try_pack7(p, len);
    if (__builtin_expect(k != 0, 1)) return k;
    return fb.intern(p, len);
}

static inline uint64_t encode_eid(const char* p, size_t len, FallbackInterner& fb) {
    uint64_t k = fast::try_pack7_unordered(p, len);
    if (__builtin_expect(k != 0, 1)) return k;
    return fb.intern(p, len);
}

struct InputFile {
    int fd = -1;
    mutable std::atomic<const char*> mapped{nullptr};
    mutable std::atomic<int> map_state{0}; // 0=unmapped, 1=mapping, 2=ready
    size_t size = 0;
    size_t header_end = 0;

    bool open(const std::string& path, bool use_mmap) {
        fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) return false;
        struct stat st;
        const int stat_result = fstat(fd, &st);
        if (stat_result != 0 || st.st_size <= 0) {
            ::close(fd);
            fd = -1;
            map_state.store(2, std::memory_order_relaxed);
            return stat_result == 0 && st.st_size == 0;
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

std::string_view get_aligned_chunk(const InputFile& f, const ReadChunk& ch, std::vector<char>& buf) {
    const char* mp = f.ensure_mapped();
    if (mp) {
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

// Batch-32 Compacted Bloom -> Slot IdentityMap64!
// - Parallel lock-free build across all threads (~1.5 ms on 10M!)
// - Batch-32 resolves 10M lookups in 9.2 ms (ZERO Bloom branch mispredicts, full MLP prefetch!)
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
    bool empty() const { return count_ == 0; }

    void build_single_from_parent(const uint64_t* keys, const uint32_t* parent, size_t n, size_t mapped) {
        if (mapped == 0) return;

        // Tight ~10 bits/key Bloom filter (~512 KB on 10M -> 100% L2-resident, ~1.2% FP rate!)
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

    // Batch-32 (or up to 64) compacted Bloom -> Slot resolver!
    // In-place: transforms uids[0..n-1] -> pids and writes hashes[0..n-1] = hash_u64(pid)!
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

// SOA Event Payload (16B aligned) stored separately from uint64_t eids[] inside each 32KB slab!
// Cuts Stage 2 dedup_p1 read bandwidth by 3x (80 MB vs 240 MB) and emit_p2 by 1.5x (160 MB vs 240 MB)!
struct EvPay {
    uint64_t uid;
    uint64_t meta;
};
static_assert(sizeof(EvPay) == 16);

struct WideEvRow {
    uint64_t eid;
    uint64_t uid;
    int64_t ingest_sec;
    int64_t ts_ms;
    uint32_t price_cents;
    uint32_t cat_id;
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

// Thread-local slab allocator backed by 4 MB hugepage-advised bump chunks + zero-fault recycler.
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
            uint8_t* p = nullptr;
#if defined(__linux__)
            if (g_use_huge && chunk_bytes >= 2 * 1024 * 1024) {
                void* mp = mmap(nullptr, chunk_bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
                if (mp != MAP_FAILED) {
#if defined(MADV_HUGEPAGE)
                    madvise(mp, chunk_bytes, MADV_HUGEPAGE);
#endif
                    p = static_cast<uint8_t*>(mp);
                }
            }
#endif
            if (!p) p = static_cast<uint8_t*>(std::malloc(chunk_bytes));
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

    inline void push(size_t b, const T& item, SlabArena& arena) {
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

    void recycle_bucket(size_t b, SlabArena& arena) {
        auto& lst = lists[b];
        for (T* p : lst.slabs) arena.recycle(p);
        lst.slabs.clear();
        lst.counts.clear();
        lst.total_count = 0;
    }
};

// SOA BucketSet for Events: each slab holds eids[cap] followed by 16B-aligned EvPay pays[cap].
// Cursor is only 16 bytes per bucket (256 buckets = 4 KB -> 100% L1D resident!).
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
        // Keep cap_per_slab even so cap_per_slab * 8 is 16-byte aligned for EvPay!
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

    inline void push(size_t b, uint64_t eid, uint64_t uid, uint64_t meta, SlabArena& arena) {
        auto& c = cursors[b];
        if (__builtin_expect(c.cnt == c.cap, 0)) {
            alloc_next(b, arena);
        }
        uint32_t idx = c.cnt++;
        reinterpret_cast<uint64_t*>(c.raw)[idx] = eid;
        reinterpret_cast<EvPay*>(c.raw + pay_byte_off)[idx] = EvPay{uid, meta};
    }

    inline uint64_t* eids_of(uint8_t* raw) const {
        return reinterpret_cast<uint64_t*>(raw);
    }

    inline EvPay* pays_of(uint8_t* raw) const {
        return reinterpret_cast<EvPay*>(raw + pay_byte_off);
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

    inline uint32_t lookup_product_cat(const char* p, size_t len) const {
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

    inline uint32_t lookup_campaign(const char* p, size_t len) const {
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

// Zero-alloc single-pass catalog loader (10x faster: no 150K std::string allocs, no StringMap inserts for p<n>/c_<n>!).
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

        // Sort the tiny list of unique categories so cid order == byte-wise lexicographic order!
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

struct LinkEdge64 {
    uint64_t a;
    uint64_t b;
};

// Fast single-threaded prefetched links parse + compact 8B-slot DSU + IdentityMap64 build.
// Runs on Thread 0 COMPLETELY OVERLAPPED with Threads 1..N-1 scanning events/*.csv!
void parse_links_and_build_identity_single(const InputFile* link_files,
                                           size_t n_link_files,
                                           size_t total_link_bytes,
                                           FallbackInterner& fb,
                                           IdentityMap64& id_map) {
    if (total_link_bytes == 0) return;
    std::vector<LinkEdge64> edges;
    edges.reserve(total_link_bytes / 20 + 64);
    std::vector<char> io_buf;
    auto chunks = build_read_chunks(link_files, n_link_files, 1 << 20);
    for (const auto& ch : chunks) {
        auto sv = get_aligned_chunk(link_files[ch.file_idx], ch, io_buf);
        if (sv.empty()) continue;
        fast::scan_csv_chunk<2>(sv.data(), sv.size(), [&](const fast::Field (&f)[2]) {
            if (f[0].len == 0 || f[1].len == 0) return;
            if (f[0].len == f[1].len && std::memcmp(f[0].ptr, f[1].ptr, f[0].len) == 0) return;
            edges.push_back(LinkEdge64{
                encode_key(f[0].ptr, f[0].len, fb),
                encode_key(f[1].ptr, f[1].len, fb)
            });
        });
    }
    size_t total_edges = edges.size();
    if (total_edges == 0) return;

    // Compact 8-byte NodeSlot (2x smaller -> 8.4 MB instead of 16.8 MB on 10M!)
    struct NodeSlot { uint32_t tag; uint32_t nid_p1; };
    size_t est = std::max<size_t>(64, total_edges * 7 / 5);
    size_t cap = next_pow2(std::max<size_t>(64, total_edges * 2));
    NodeSlot* table = static_cast<NodeSlot*>(std::malloc(cap * sizeof(NodeSlot)));
    std::memset(table, 0, cap * sizeof(NodeSlot));
    size_t mask = cap - 1;

    std::vector<uint64_t> keys;
    std::vector<uint32_t> parent;
    keys.reserve(est);
    parent.reserve(est);

    auto grow_table = [&]() {
        size_t ncap = cap * 2;
        NodeSlot* nt = static_cast<NodeSlot*>(std::malloc(ncap * sizeof(NodeSlot)));
        std::memset(nt, 0, ncap * sizeof(NodeSlot));
        size_t nmask = ncap - 1;
        for (size_t i = 0; i < cap; ++i) {
            if (!table[i].nid_p1) continue;
            uint64_t h = fast::hash_u64(keys[table[i].nid_p1 - 1]);
            size_t idx = static_cast<size_t>(h) & nmask;
            while (nt[idx].nid_p1) idx = (idx + 1) & nmask;
            nt[idx] = table[i];
        }
        std::free(table);
        table = nt; cap = ncap; mask = nmask;
    };

    auto intern = [&](uint64_t k) -> uint32_t {
        uint64_t h = fast::hash_u64(k);
        uint32_t tag = static_cast<uint32_t>(h >> 32);
        size_t idx = static_cast<size_t>(h) & mask;
        while (table[idx].nid_p1) {
            if (table[idx].tag == tag && keys[table[idx].nid_p1 - 1] == k) return table[idx].nid_p1 - 1;
            idx = (idx + 1) & mask;
        }
        if (__builtin_expect((keys.size() + 1) * 10 >= cap * 7, 0)) {
            grow_table();
            idx = static_cast<size_t>(h) & mask;
            while (table[idx].nid_p1) idx = (idx + 1) & mask;
        }
        uint32_t nid = static_cast<uint32_t>(keys.size());
        keys.push_back(k);
        parent.push_back(nid);
        table[idx] = NodeSlot{tag, nid + 1};
        return nid;
    };

    auto find_r = [&](uint32_t x) -> uint32_t {
        while (parent[x] != x) {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        return x;
    };

    const LinkEdge64* arr = edges.data();
    for (size_t i = 0; i < total_edges; ++i) {
        if (__builtin_expect(i + 16 < total_edges, 1)) {
            __builtin_prefetch(&table[static_cast<size_t>(fast::hash_u64(arr[i + 16].a)) & mask], 1, 1);
            __builtin_prefetch(&table[static_cast<size_t>(fast::hash_u64(arr[i + 16].b)) & mask], 1, 1);
        }
        uint32_t u = find_r(intern(arr[i].a));
        uint32_t v = find_r(intern(arr[i].b));
        if (u == v) continue;
        if (fb.key_less(keys[v], keys[u])) parent[u] = v;
        else parent[v] = u;
    }
    std::free(table);

    size_t n = keys.size();
    size_t mapped = 0;
    for (size_t i = 0; i < n; ++i) {
        uint32_t r = find_r(static_cast<uint32_t>(i));
        parent[i] = r;
        mapped += static_cast<size_t>(r != static_cast<uint32_t>(i));
    }
    id_map.build_single_from_parent(keys.data(), parent.data(), n, mapped);
}

// Lean Parallel 3-step DSU (used if links is huge relative to events):
// - Union directly by min_key (keys[root] IS the component min -> ZERO rank_sz, ZERO min_key, ZERO canonical!)
// - Uninitialized malloc buffers + parallel parent[i]=i init + 32-bucket L2-resident interning!
void build_identity_from_links(const std::vector<std::vector<LinkEdge64>>& bufs,
                               const FallbackInterner& fb,
                               IdentityMap64& id_map,
                               ThreadTeam& team,
                               size_t num_threads) {
    size_t total_edges = 0;
    for (const auto& b : bufs) total_edges += b.size();
    if (total_edges == 0) return;

    if (num_threads == 1 || total_edges < 8192) {
        struct NodeSlot { uint64_t key; uint32_t node; uint32_t pad; };
        size_t est = std::max<size_t>(64, total_edges * 3 / 2);
        size_t cap = next_pow2(est * 2);
        std::vector<NodeSlot> table(cap, NodeSlot{0, 0, 0});
        size_t mask = cap - 1;
        std::vector<uint64_t> keys;
        std::vector<uint32_t> parent;
        keys.reserve(est); parent.reserve(est);
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
                if (fb.key_less(keys[v], keys[u])) parent[u] = v;
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
        id_map.build_from_parent(keys.data(), parent.data(), n, mapped, team, 1);
        return;
    }

    constexpr size_t NB = 32;
    constexpr unsigned BSHIFT = 59;
    struct Endpoint {
        uint64_t key;
        uint32_t ref;
        uint32_t pad;
    };
    struct RefLocal {
        uint32_t ref;
        uint32_t local_nid;
    };

    std::vector<size_t> edge_off(bufs.size() + 1, 0);
    for (size_t t = 0; t < bufs.size(); ++t) edge_off[t + 1] = edge_off[t] + bufs[t].size();

    std::vector<std::vector<std::vector<Endpoint>>> ep_buckets(num_threads);
    for (size_t t = 0; t < num_threads; ++t) ep_buckets[t].resize(NB);

    // Step A: Scatter endpoints into 32 buckets in parallel.
    team.run([&](size_t tid) {
        if (tid >= num_threads) return;
        auto& my_b = ep_buckets[tid];
        size_t est = (bufs[tid].size() * 2) / NB + 32;
        for (size_t b = 0; b < NB; ++b) my_b[b].reserve(est);
        const auto& edges = bufs[tid];
        uint32_t base_e = static_cast<uint32_t>(edge_off[tid]);
        for (size_t i = 0; i < edges.size(); ++i) {
            uint64_t a = edges[i].a;
            uint64_t b = edges[i].b;
            uint32_t e = base_e + static_cast<uint32_t>(i);
            my_b[fast::hash_u64(a) >> BSHIFT].push_back(Endpoint{a, e << 1, 0});
            my_b[fast::hash_u64(b) >> BSHIFT].push_back(Endpoint{b, (e << 1) | 1U, 0});
        }
    });

    // Step B: Deduplicate each bucket in parallel inside L2 cache (~512 KB hash table!).
    std::vector<std::vector<uint64_t>> b_keys(NB);
    std::vector<std::vector<RefLocal>> b_refs(NB);
    std::atomic<size_t> next_b{0};

    team.run([&](size_t tid) {
        if (tid >= num_threads) return;
        struct LocalSlot { uint64_t key; uint32_t nid; uint32_t pad; };
        std::vector<LocalSlot> ltab;
        while (true) {
            size_t b = next_b.fetch_add(1, std::memory_order_relaxed);
            if (b >= NB) break;
            size_t tot = 0;
            for (size_t t = 0; t < num_threads; ++t) tot += ep_buckets[t][b].size();
            if (tot == 0) continue;

            size_t cap = next_pow2(std::max<size_t>(32, tot * 2));
            if (ltab.size() < cap) ltab.resize(cap);
            std::memset(ltab.data(), 0, cap * sizeof(LocalSlot));
            size_t mask = cap - 1;

            auto& out_k = b_keys[b];
            auto& out_r = b_refs[b];
            out_k.reserve(tot * 3 / 4);
            out_r.reserve(tot);

            for (size_t t = 0; t < num_threads; ++t) {
                const auto& vec = ep_buckets[t][b];
                size_t sz = vec.size();
                const Endpoint* arr = vec.data();
                for (size_t i = 0; i < sz; ++i) {
                    if (__builtin_expect(i + 8 < sz, 1)) {
                        __builtin_prefetch(&ltab[static_cast<size_t>(fast::hash_u64(arr[i + 8].key)) & mask], 1, 1);
                    }
                    uint64_t k = arr[i].key;
                    size_t idx = static_cast<size_t>(fast::hash_u64(k)) & mask;
                    while (ltab[idx].key) {
                        if (ltab[idx].key == k) {
                            out_r.push_back(RefLocal{arr[i].ref, ltab[idx].nid});
                            goto found_local;
                        }
                        idx = (idx + 1) & mask;
                    }
                    {
                        uint32_t nid = static_cast<uint32_t>(out_k.size());
                        out_k.push_back(k);
                        ltab[idx] = LocalSlot{k, nid, 0};
                        out_r.push_back(RefLocal{arr[i].ref, nid});
                    }
                found_local:;
                }
            }
        }
    });

    std::vector<uint32_t> b_node_off(NB + 1, 0);
    for (size_t b = 0; b < NB; ++b) b_node_off[b + 1] = b_node_off[b] + static_cast<uint32_t>(b_keys[b].size());
    size_t num_nodes = b_node_off[NB];

    // Raw uninitialized allocations (eliminates 23 MB of single-threaded zero-fill!).
    uint64_t* keys = static_cast<uint64_t*>(std::malloc(num_nodes * sizeof(uint64_t)));
    uint32_t* parent = static_cast<uint32_t*>(std::malloc(num_nodes * sizeof(uint32_t)));
    uint32_t* edge_ends = static_cast<uint32_t*>(std::malloc(total_edges * 2 * sizeof(uint32_t)));

    // Step C: Parallel scatter of keys[], parent[i]=i, and edge_ends[]!
    next_b.store(0, std::memory_order_relaxed);
    team.run([&](size_t tid) {
        if (tid >= num_threads) return;
        while (true) {
            size_t b = next_b.fetch_add(1, std::memory_order_relaxed);
            if (b >= NB) break;
            uint32_t off = b_node_off[b];
            size_t ksz = b_keys[b].size();
            if (ksz > 0) {
                std::memcpy(keys + off, b_keys[b].data(), ksz * sizeof(uint64_t));
                for (size_t i = 0; i < ksz; ++i) parent[off + i] = off + static_cast<uint32_t>(i);
            }
            for (const auto& rl : b_refs[b]) {
                edge_ends[rl.ref] = off + rl.local_nid;
            }
        }
    });

    // Step D: Pure integer Union-Find with union-by-min_key (ZERO rank_sz, ZERO min_key, ZERO canonical!).
    auto find_root = [&](uint32_t x) -> uint32_t {
        while (parent[x] != x) {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        return x;
    };

    for (size_t e = 0; e < total_edges; ++e) {
        uint32_t u = find_root(edge_ends[e * 2]);
        uint32_t v = find_root(edge_ends[e * 2 + 1]);
        if (u == v) continue;
        if (fb.key_less(keys[v], keys[u])) {
            parent[u] = v;
        } else {
            parent[v] = u;
        }
    }

    size_t mapped = 0;
    for (size_t i = 0; i < num_nodes; ++i) {
        uint32_t r = find_root(static_cast<uint32_t>(i));
        parent[i] = r;
        mapped += static_cast<size_t>(r != static_cast<uint32_t>(i));
    }

    // Step E: Parallel lock-free IdentityMap64 build directly from (keys, parent)!
    id_map.build_from_parent(keys, parent, num_nodes, mapped, team, num_threads);

    std::free(edge_ends);
    std::free(parent);
    std::free(keys);
}

// Hot/Cold split for CampDayAccum:
// - HotCampDay (8 bytes: sessions + assists) = 224 KB per thread -> 100% L2-resident!
//   Credited session + assist hit the EXACT SAME 8-byte word!
// - ColdCampDay (16 bytes: purchasing_sessions + revenue_cents) -> ONLY touched on 3% purchasing sessions!
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

    inline void add_session_dense(uint32_t camp_id, uint32_t day_off, int64_t rev) {
        size_t idx = static_cast<size_t>(camp_id) * static_cast<size_t>(num_days_) + day_off;
        dense_hot_[idx].sessions++;
        if (__builtin_expect(rev > 0, 0)) {
            dense_cold_[idx].purchasing_sessions++;
            dense_cold_[idx].revenue_cents += rev;
        }
    }

    inline void add_assist_dense(uint32_t camp_id, uint32_t day_off) {
        dense_hot_[static_cast<size_t>(camp_id) * static_cast<size_t>(num_days_) + day_off].assists++;
    }

    inline void add_session_and_assist_dense(uint32_t camp_id, uint32_t day_off, int64_t rev) {
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

    inline void add_assist(uint32_t camp_id, int32_t day) {
        if (__builtin_expect(dense_ && day >= base_day_ && day < base_day_ + num_days_, 1)) {
            add_assist_dense(camp_id, static_cast<uint32_t>(day - base_day_));
            return;
        }
        get_sparse(camp_id, day).assisted_sessions++;
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

// Single-pass-histogram, skip-trivial, zero-memcpy LSB radix sort on upper 31 bits of uint64_t!
// Returns pointer to the sorted array (either `a` or `tmp.data()`).
static const uint64_t* radix_sort_ev64(uint64_t* a, size_t n, std::vector<uint64_t>& tmp) {
    if (tmp.size() < n) tmp.resize(n);
    uint64_t* b = tmp.data();
    uint32_t c0[2048], c1[1024], c2[1024];
    std::memset(c0, 0, sizeof(c0));
    std::memset(c1, 0, sizeof(c1));
    std::memset(c2, 0, sizeof(c2));

    for (size_t i = 0; i < n; ++i) {
        uint64_t v = a[i];
        c0[(v >> 32) & 2047U]++;
        c1[(v >> 43) & 1023U]++;
        c2[(v >> 53) & 1023U]++;
    }

    uint64_t* src = a;
    uint64_t* dst = b;
    uint32_t n32 = static_cast<uint32_t>(n);

    if (c0[(src[0] >> 32) & 2047U] != n32) {
        uint32_t sum = 0;
        for (size_t i = 0; i < 2048; ++i) { uint32_t c = c0[i]; c0[i] = sum; sum += c; }
        for (size_t i = 0; i < n; ++i) dst[c0[(src[i] >> 32) & 2047U]++] = src[i];
        std::swap(src, dst);
    }
    if (c1[(src[0] >> 43) & 1023U] != n32) {
        uint32_t sum = 0;
        for (size_t i = 0; i < 1024; ++i) { uint32_t c = c1[i]; c1[i] = sum; sum += c; }
        for (size_t i = 0; i < n; ++i) dst[c1[(src[i] >> 43) & 1023U]++] = src[i];
        std::swap(src, dst);
    }
    if (c2[(src[0] >> 53) & 1023U] != n32) {
        uint32_t sum = 0;
        for (size_t i = 0; i < 1024; ++i) { uint32_t c = c2[i]; c2[i] = sum; sum += c; }
        for (size_t i = 0; i < n; ++i) dst[c2[(src[i] >> 53) & 1023U]++] = src[i];
        std::swap(src, dst);
    }
    return src;
}

// Compact L1-resident 4-pass 8-bit skip-trivial radix sort (4 KB stack histogram) for 64 <= n < 1024!
static const uint64_t* radix_sort_ev64_8bit(uint64_t* a, size_t n, std::vector<uint64_t>& tmp) {
    if (tmp.size() < n) tmp.resize(n);
    uint64_t* b = tmp.data();
    uint32_t c0[256], c1[256], c2[256], c3[256];
    std::memset(c0, 0, sizeof(c0));
    std::memset(c1, 0, sizeof(c1));
    std::memset(c2, 0, sizeof(c2));
    std::memset(c3, 0, sizeof(c3));

    for (size_t i = 0; i < n; ++i) {
        uint64_t v = a[i];
        c0[(v >> 32) & 255U]++;
        c1[(v >> 40) & 255U]++;
        c2[(v >> 48) & 255U]++;
        c3[(v >> 56) & 255U]++;
    }

    uint64_t* src = a;
    uint64_t* dst = b;
    uint32_t n32 = static_cast<uint32_t>(n);
    uint32_t* cs[4] = {c0, c1, c2, c3};
    unsigned shifts[4] = {32, 40, 48, 56};
    for (int p = 0; p < 4; ++p) {
        uint32_t* cnt = cs[p];
        unsigned sh = shifts[p];
        if (cnt[(src[0] >> sh) & 255U] == n32) continue;
        uint32_t sum = 0;
        for (size_t i = 0; i < 256; ++i) { uint32_t c = cnt[i]; cnt[i] = sum; sum += c; }
        for (size_t i = 0; i < n; ++i) dst[cnt[(src[i] >> sh) & 255U]++] = src[i];
        std::swap(src, dst);
    }
    return src;
}

static inline void cswap64(uint64_t& a, uint64_t& b) {
    uint64_t lo = a < b ? a : b;
    uint64_t hi = a < b ? b : a;
    a = lo;
    b = hi;
}

// Hyper-tuned hybrid sort for uint64_t events:
// - n <= 8: optimal branchless sorting networks (ZERO branch mispredicts!)
// - 9 <= n <= 24: tight insertion sort!
// - 25 <= n < 64: std::sort
// - 64 <= n < 1024: L1-resident 8-bit skip-trivial zero-memcpy radix sort!
// - n >= 1024: 3-pass 11+10+10 radix sort!
static inline const uint64_t* sort_ev64(uint64_t* a, size_t n, std::vector<uint64_t>& tmp) {
    if (n <= 1) return a;
    if (n == 2) {
        cswap64(a[0], a[1]);
        return a;
    }
    if (n == 3) {
        cswap64(a[0], a[1]); cswap64(a[1], a[2]); cswap64(a[0], a[1]);
        return a;
    }
    if (n == 4) {
        cswap64(a[0], a[1]); cswap64(a[2], a[3]);
        cswap64(a[0], a[2]); cswap64(a[1], a[3]);
        cswap64(a[1], a[2]);
        return a;
    }
    if (n == 5) {
        cswap64(a[0], a[1]); cswap64(a[3], a[4]);
        cswap64(a[2], a[4]); cswap64(a[2], a[3]); cswap64(a[1], a[4]);
        cswap64(a[0], a[3]); cswap64(a[0], a[2]); cswap64(a[1], a[3]);
        cswap64(a[1], a[2]);
        return a;
    }
    if (n == 6) {
        cswap64(a[1], a[2]); cswap64(a[4], a[5]);
        cswap64(a[0], a[2]); cswap64(a[3], a[5]);
        cswap64(a[0], a[1]); cswap64(a[3], a[4]); cswap64(a[2], a[5]);
        cswap64(a[0], a[3]); cswap64(a[1], a[4]);
        cswap64(a[2], a[4]); cswap64(a[1], a[3]);
        cswap64(a[2], a[3]);
        return a;
    }
    if (n == 7) {
        cswap64(a[1], a[2]); cswap64(a[3], a[4]); cswap64(a[5], a[6]);
        cswap64(a[0], a[2]); cswap64(a[3], a[5]); cswap64(a[4], a[6]);
        cswap64(a[0], a[1]); cswap64(a[4], a[5]); cswap64(a[2], a[6]);
        cswap64(a[0], a[4]); cswap64(a[1], a[5]);
        cswap64(a[0], a[3]); cswap64(a[2], a[5]);
        cswap64(a[1], a[3]); cswap64(a[2], a[4]);
        cswap64(a[2], a[3]);
        return a;
    }
    if (n == 8) {
        cswap64(a[0], a[1]); cswap64(a[2], a[3]); cswap64(a[4], a[5]); cswap64(a[6], a[7]);
        cswap64(a[0], a[2]); cswap64(a[1], a[3]); cswap64(a[4], a[6]); cswap64(a[5], a[7]);
        cswap64(a[1], a[2]); cswap64(a[5], a[6]); cswap64(a[0], a[4]); cswap64(a[3], a[7]);
        cswap64(a[1], a[5]); cswap64(a[2], a[6]);
        cswap64(a[1], a[4]); cswap64(a[3], a[6]);
        cswap64(a[2], a[4]); cswap64(a[3], a[5]);
        cswap64(a[3], a[4]);
        return a;
    }
    if (n <= 24) {
        // Unguarded insertion sort: hoist minimum to a[0] first so inner loop drops `j > 0` branch!
        size_t min_i = 0;
        uint64_t min_v = a[0];
        for (size_t i = 1; i < n; ++i) {
            if (a[i] < min_v) { min_v = a[i]; min_i = i; }
        }
        a[min_i] = a[0];
        a[0] = min_v;
        for (size_t i = 2; i < n; ++i) {
            uint64_t key = a[i];
            size_t j = i;
            while (a[j - 1] > key) {
                a[j] = a[j - 1];
                --j;
            }
            a[j] = key;
        }
        return a;
    }
    if (n >= 1024) {
        return radix_sort_ev64(a, n, tmp);
    }
    if (n >= 40) {
        return radix_sort_ev64_8bit(a, n, tmp);
    }
    std::sort(a, a + n);
    return a;
}

struct ClickItem {
    uint64_t ts_camp;
    uint64_t cid;
};

static inline bool click_less(const ClickItem& a, const ClickItem& b, const FallbackInterner& fb) {
    uint64_t ta = a.ts_camp >> 20;
    uint64_t tb = b.ts_camp >> 20;
    if (__builtin_expect(ta != tb, 1)) return ta < tb;
    return fb.key_less(a.cid, b.cid);
}

static const ClickItem* sort_clicks(ClickItem* a, size_t n, const FallbackInterner& fb,
                                    std::vector<ClickItem>& tmp) {
    if (n <= 1) return a;
    if (n == 2) {
        if (click_less(a[1], a[0], fb)) std::swap(a[0], a[1]);
        return a;
    }
    if (n <= 24) {
        for (size_t i = 1; i < n; ++i) {
            ClickItem key = a[i];
            size_t j = i;
            while (j > 0 && click_less(key, a[j - 1], fb)) {
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
            // Fix any rare equal-timestamp runs by cid!
            for (size_t i = 1; i < n; ++i) {
                if (__builtin_expect((src[i - 1].ts_camp >> 20) == (src[i].ts_camp >> 20), 0)) {
                    size_t st = i - 1;
                    uint64_t t0 = src[st].ts_camp >> 20;
                    while (i < n && (src[i].ts_camp >> 20) == t0) ++i;
                    std::sort(src + st, src + i, [&](const ClickItem& x, const ClickItem& y) {
                        return fb.key_less(x.cid, y.cid);
                    });
                }
            }
            return src;
        }
    }
    std::sort(a, a + n, [&](const ClickItem& x, const ClickItem& y) {
        return click_less(x, y, fb);
    });
    return a;
}

// Raw uninitialized malloc-backed Parquet column builder.
// Uses INT32 for sessions (mathematically <= 48/day!) and INT32 for events when total_event_bytes < 20 GB!
struct UserDailyWriter {
    size_t rg_limit = 250000;
    bool events_i32 = true;
    pq::ParquetFileWriter writer;
    uint8_t* col_uid = nullptr;
    size_t uid_cap = 0, uid_pos = 0;
    int32_t* col_day = nullptr;
    int32_t* col_sessions = nullptr;
    int32_t* col_events32 = nullptr;
    int64_t* col_events64 = nullptr;
    int64_t* col_revenue = nullptr;
    uint8_t* col_top_cat = nullptr;
    size_t cat_cap = 0, cat_pos = 0;
    uint8_t* def_bits = nullptr;
    size_t buffered = 0;

    static inline void append_str_fast(uint8_t*& buf, size_t& cap, size_t& pos, const char* data, uint32_t len) {
        if (__builtin_expect(pos + 24 + len > cap, 0)) {
            cap = std::max(cap * 2, pos + 24 + len + 4096);
            buf = static_cast<uint8_t*>(std::realloc(buf, cap));
        }
        std::memcpy(buf + pos, &len, 4);
        std::memcpy(buf + pos + 4, data, len);
        pos += 4 + len;
    }

    static inline void append_uid_fast(uint8_t*& buf, size_t& cap, size_t& pos,
                                       const char* data, uint32_t len, bool is_inline) {
        if (__builtin_expect(pos + 24 + len > cap, 0)) {
            cap = std::max(cap * 2, pos + 24 + len + 4096);
            buf = static_cast<uint8_t*>(std::realloc(buf, cap));
        }
        std::memcpy(buf + pos, &len, 4);
        if (__builtin_expect(is_inline, 1)) {
            // Fixed 16-byte SIMD store (safe: data points to pid_buf[16] and col_uid has >= 24B slack!)
            std::memcpy(buf + pos + 4, data, 16);
        } else {
            std::memcpy(buf + pos + 4, data, len);
        }
        pos += 4 + len;
    }

    bool open(const std::string& path, size_t est_rows, bool use_events_i32) {
        events_i32 = use_events_i32;
        rg_limit = std::clamp<size_t>(est_rows, 4096, 250000);
        uid_cap = rg_limit * 14;
        col_uid = static_cast<uint8_t*>(std::malloc(uid_cap));
        uid_pos = 0;
        col_day = static_cast<int32_t*>(std::malloc(rg_limit * sizeof(int32_t)));
        col_sessions = static_cast<int32_t*>(std::malloc(rg_limit * sizeof(int32_t)));
        if (events_i32) {
            col_events32 = static_cast<int32_t*>(std::malloc(rg_limit * sizeof(int32_t)));
        } else {
            col_events64 = static_cast<int64_t*>(std::malloc(rg_limit * sizeof(int64_t)));
        }
        col_revenue = static_cast<int64_t*>(std::malloc(rg_limit * sizeof(int64_t)));
        cat_cap = rg_limit * 4 + 4096;
        col_top_cat = static_cast<uint8_t*>(std::malloc(cat_cap));
        cat_pos = 0;
        def_bits = static_cast<uint8_t*>(std::malloc((rg_limit + 7) / 8 + 8));
        buffered = 0;
        return writer.open(path, {
            {"user_id", pq::PT_BYTE_ARRAY, pq::REP_REQUIRED, pq::CT_UTF8},
            {"day", pq::PT_INT32, pq::REP_REQUIRED, pq::CT_DATE},
            {"sessions", pq::PT_INT32, pq::REP_REQUIRED, pq::CT_NONE},
            {"events", events_i32 ? pq::PT_INT32 : pq::PT_INT64, pq::REP_REQUIRED, pq::CT_NONE},
            {"revenue_cents", pq::PT_INT64, pq::REP_REQUIRED, pq::CT_NONE},
            {"top_category", pq::PT_BYTE_ARRAY, pq::REP_OPTIONAL, pq::CT_UTF8},
        });
    }

    inline void add_row(std::string_view uid, bool uid_is_inline, int32_t day, int32_t sessions,
                        int64_t events, int64_t revenue, const std::string* top_cat) {
        size_t i = buffered;
        append_uid_fast(col_uid, uid_cap, uid_pos, uid.data(), static_cast<uint32_t>(uid.size()), uid_is_inline);
        col_day[i] = day;
        col_sessions[i] = sessions;
        if (__builtin_expect(events_i32, 1)) {
            col_events32[i] = static_cast<int32_t>(events);
        } else {
            col_events64[i] = events;
        }
        col_revenue[i] = revenue;

        if ((i & 7) == 0) def_bits[i >> 3] = 0;
        if (top_cat != nullptr) {
            def_bits[i >> 3] |= static_cast<uint8_t>(1U << (i & 7));
            append_str_fast(col_top_cat, cat_cap, cat_pos, top_cat->data(), static_cast<uint32_t>(top_cat->size()));
        }
        buffered = i + 1;
        if (__builtin_expect(buffered == rg_limit, 0)) flush();
    }

    void flush() {
        if (buffered == 0) return;
        std::vector<pq::ParquetFileWriter::ColumnData> cols(6);
        cols[0] = {col_uid, uid_pos, nullptr};
        cols[1] = {col_day, buffered * sizeof(int32_t), nullptr};
        cols[2] = {col_sessions, buffered * sizeof(int32_t), nullptr};
        if (events_i32) {
            cols[3] = {col_events32, buffered * sizeof(int32_t), nullptr};
        } else {
            cols[3] = {col_events64, buffered * sizeof(int64_t), nullptr};
        }
        cols[4] = {col_revenue, buffered * sizeof(int64_t), nullptr};
        cols[5] = {col_top_cat, cat_pos, def_bits};
        writer.write_row_group(static_cast<int64_t>(buffered), cols);
        uid_pos = 0;
        cat_pos = 0;
        buffered = 0;
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
        cols[0] = {col_cid.data(), col_cid.size(), nullptr};
        cols[1] = {col_ch.data(), col_ch.size(), nullptr};
        cols[2] = {col_day.data(), col_day.size() * sizeof(int32_t), nullptr};
        cols[3] = {col_sess.data(), col_sess.size() * sizeof(int64_t), nullptr};
        cols[4] = {col_psess.data(), col_psess.size() * sizeof(int64_t), nullptr};
        cols[5] = {col_rev.data(), col_rev.size() * sizeof(int64_t), nullptr};
        cols[6] = {col_ass.data(), col_ass.size() * sizeof(int64_t), nullptr};
        writer.write_row_group(static_cast<int64_t>(col_day.size()), cols);
    }
    return writer.close();
}

} // namespace

int main() {
    auto t_total = Clock::now();

    const char* env_data = std::getenv("DATA_DIR");
    const char* env_out = std::getenv("OUTPUT_DIR");
    const char* env_ncpu = std::getenv("NCPU");
    const char* env_mmap = std::getenv("USE_MMAP");

    std::string data_dir = env_data ? env_data : "/data";
    std::string out_dir = env_out ? env_out : "/output";
    size_t hw_threads = std::thread::hardware_concurrency();
    if (hw_threads == 0) hw_threads = 1;
    size_t ncpu = env_ncpu ? static_cast<size_t>(std::max(1, std::atoi(env_ncpu))) : std::min<size_t>(8, hw_threads);
    bool use_mmap = env_mmap ? (std::atoi(env_mmap) != 0) : false;
    const char* env_pf = std::getenv("PREFETCH");
    bool use_prefetch = env_pf ? (std::atoi(env_pf) != 0) : true;
    const char* env_huge = std::getenv("HUGE");
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

    // Stage 0a: Load catalogs BEFORE spawning threads so main thread is never CFS-preempted by newborn workers!
    auto t_cat = Clock::now();
    Catalogs catalogs;
    load_catalogs(data_dir, catalogs);
    double ms_cat = elapsed_sec(t_cat) * 1000.0;
    if (catalogs.categories.size() > 255) compact_ev_mode = false;

    auto t_s0_start = Clock::now();
    size_t total_files = n_ev_files + n_cl_files + n_ln_files;
    size_t init_threads = (total_files <= 6) ? 1 : ncpu;
    ThreadTeam team(init_threads);
    double ms_team = elapsed_sec(t_s0_start) * 1000.0;

    // Stage 0b: Parallel open()+fstat() across ALL threads (NO mmap_lock contention — mmap is lazy in Stage 1!).
    double ms_open = 0.0;
    {
        auto to = Clock::now();
        std::atomic<size_t> next_f{0};
        team.run([&](size_t) {
            while (true) {
                size_t i = next_f.fetch_add(1, std::memory_order_relaxed);
                if (i >= total_files) break;
                if (i < n_ev_files) {
                    event_files[i].open(event_paths[i], use_mmap);
                } else if (i < n_ev_files + n_cl_files) {
                    click_files[i - n_ev_files].open(click_paths[i - n_ev_files], use_mmap);
                } else {
                    link_files[i - n_ev_files - n_cl_files].open(link_paths[i - n_ev_files - n_cl_files], use_mmap);
                }
            }
        });
        ms_open = elapsed_sec(to) * 1000.0;
    }

    size_t total_event_bytes = 0;
    for (size_t i = 0; i < n_ev_files; ++i) total_event_bytes += event_files[i].size;
    size_t total_link_bytes = 0;
    for (size_t i = 0; i < n_ln_files; ++i) total_link_bytes += link_files[i].size;

    bool use_events_i32 = (total_event_bytes < 20ULL * 1024 * 1024 * 1024);

    size_t num_threads = team.size();
    if (total_event_bytes < 512 * 1024 && total_link_bytes < 512 * 1024) {
        num_threads = 1;
    }

    size_t num_buckets = (total_event_bytes < 4 * 1024 * 1024)
                             ? 16
                             : ((total_event_bytes >= 1536ULL * 1024 * 1024) ? 512 : 256);
    unsigned bucket_shift = (num_buckets == 16) ? 60U : ((num_buckets == 512) ? 55U : 56U);
    size_t num_parts = (total_event_bytes < 4 * 1024 * 1024)
                           ? 16
                           : ((total_event_bytes >= 2048ULL * 1024 * 1024) ? 256 : 128);
    unsigned part_shift = (num_parts == 16) ? 60U : ((num_parts == 256) ? 56U : 57U);
    size_t slab_bytes = (total_event_bytes < 4 * 1024 * 1024) ? 4096 : 32768;

    FallbackInterner fallback;
    IdentityMap64 identity_map;
    std::atomic<bool> identity_ready{false};
    double dsu_ms = 0.0;

    // Overlap links parse + DSU on Thread 0 during Stage 1 whenever events dominates links (saves ~30ms!).
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
                edges.reserve(total_link_bytes / (24 * num_threads) + 64);
                while (true) {
                    size_t ci = next_lc.fetch_add(1, std::memory_order_relaxed);
                    if (ci >= link_chunks.size()) break;
                    const auto& ch = link_chunks[ci];
                    auto sv = get_aligned_chunk(link_files[ch.file_idx], ch, io_buf);
                    if (sv.empty()) continue;
                    fast::scan_csv_chunk<2>(sv.data(), sv.size(), [&](const fast::Field (&f)[2]) {
                        if (f[0].len == 0 || f[1].len == 0) return;
                        if (f[0].len == f[1].len && std::memcmp(f[0].ptr, f[1].ptr, f[0].len) == 0) return;
                        edges.push_back(LinkEdge64{
                            encode_key(f[0].ptr, f[0].len, fallback),
                            encode_key(f[1].ptr, f[1].len, fallback)
                        });
                    });
                }
            });
            build_identity_from_links(link_bufs, fallback, identity_map, team, num_threads);
        }
        dsu_ms = elapsed_sec(t_dsu) * 1000.0;
        identity_ready.store(true, std::memory_order_release);
    }

    // Async kernel readahead in consumption order (links -> events -> clicks): ~0 cost on a hot page
    // cache, but keeps the block device queue full on a COLD /data so mmap faults hit page cache!
    std::thread prefetcher;
#if defined(__linux__) && defined(POSIX_FADV_WILLNEED)
    if (use_prefetch && use_mmap && total_event_bytes >= 32ULL * 1024 * 1024) {
        prefetcher = std::thread([&]() {
            constexpr size_t STEP = 4ULL * 1024 * 1024;
            auto adv = [&](const InputFile& f) {
                if (f.fd < 0) return;
                for (size_t off = 0; off < f.size; off += STEP) {
                    posix_fadvise(f.fd, static_cast<off_t>(off), static_cast<off_t>(std::min(STEP, f.size - off)),
                                  POSIX_FADV_WILLNEED);
                }
            };
            for (size_t i = 0; i < n_ln_files; ++i) adv(link_files[i]);
            for (size_t i = 0; i < n_ev_files; ++i) adv(event_files[i]);
            for (size_t i = 0; i < n_cl_files; ++i) adv(click_files[i]);
        });
    }
#endif

    // Stage 1: Overlapped links+DSU (Thread 0) + line128 events + line64 batch-64 clicks!
    auto t_s1 = Clock::now();
    auto event_chunks = build_read_chunks(event_files.get(), n_ev_files, READ_CHUNK_SIZE);
    auto click_chunks = build_read_chunks(click_files.get(), n_cl_files, READ_CHUNK_SIZE);

    std::vector<SlabArena> arenas(num_threads);
    std::vector<EvBucketSet> ev_buckets(num_threads);
    std::vector<std::vector<std::vector<uint32_t>>> ev_purchases(num_threads);
    std::vector<std::vector<std::vector<WideEvRow>>> wide_ev_buckets(num_threads);
    std::vector<BucketSet<PersonClick>> person_cl(num_threads);
    for (size_t t = 0; t < num_threads; ++t) {
        arenas[t].init(slab_bytes, (total_event_bytes < 4 * 1024 * 1024) ? 65536 : 4 * 1024 * 1024);
        ev_buckets[t].init(num_buckets, slab_bytes);
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
            auto sv = get_aligned_chunk(click_files[ch.file_idx], ch, io_buf);
            if (sv.empty()) continue;

            fast::scan_csv_chunk<4>(sv.data(), sv.size(), [&](const fast::Field (&f)[4]) {
                if (__builtin_expect(f[0].len == 0 || f[1].len == 0 || f[2].len == 0, 0)) return;
                uint32_t camp_id = catalogs.lookup_campaign(f[2].ptr, f[2].len);
                if (__builtin_expect(camp_id == 0, 0)) return;
                int64_t ts_ms = 0;
                if (__builtin_expect(!fast::parse_timestamp_ms(f[3].ptr, f[3].len, dtab, ts_ms), 0)) return;
                uint64_t ts_off = static_cast<uint64_t>(ts_ms - click_start);
                if (__builtin_expect(ts_ms < click_start || ts_off >= cl_span_u64, 0)) return;

                b_uid[nb] = encode_key(f[1].ptr, f[1].len, fallback);
                b_cid[nb] = encode_key(f[0].ptr, f[0].len, fallback);
                b_tc[nb] = (ts_off << 20) | camp_id;
                if (__builtin_expect(++nb == 64, 0)) flush_batch();
            });
        }
        if (nb > 0) flush_batch();
    };

    auto scan_events_body = [&](size_t tid) {
        auto tev0 = Clock::now();
        std::vector<char> io_buf;
        if (!use_mmap) io_buf.reserve(READ_CHUNK_SIZE + 8192);
        auto& my_ev = ev_buckets[tid];
        auto& my_purch = ev_purchases[tid];
        auto& my_wide = wide_ev_buckets[tid];
        auto& arena = arenas[tid];

        while (true) {
            size_t ci = next_ec.fetch_add(1, std::memory_order_relaxed);
            if (ci >= event_chunks.size()) break;
            const auto& ch = event_chunks[ci];
            auto sv = get_aligned_chunk(event_files[ch.file_idx], ch, io_buf);
            if (sv.empty()) continue;

            fast::scan_csv_chunk<7>(sv.data(), sv.size(), [&](const fast::Field (&f)[7]) {
                if (__builtin_expect(f[0].len == 0, 0)) return;

                int64_t ingest_sec = 0, ingest_day_sec = 0;
                uint64_t ingest_ym = 0;
                uint16_t ingest_dd = 0;
                if (__builtin_expect(!fast::parse_ingest_f1_sec(f[3].ptr, f[3].len, dtab, ingest_sec, ingest_day_sec,
                                                                 ingest_ym, ingest_dd), 0)) {
                    return;
                }
                int64_t ingest_ms = ingest_sec * 1000LL;

                uint64_t eid = encode_eid(f[0].ptr, f[0].len, fallback);
                uint64_t eid_h = fast::hash_u64(eid);
                size_t bucket = static_cast<size_t>(eid_h >> bucket_shift);

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
                    my_ev.push(bucket, eid, 0ULL, static_cast<uint64_t>(ingest_sec), arena);
                    return;
                }

                uint64_t uid = encode_key(f[1].ptr, f[1].len, fallback);
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
                        my_ev.push(bucket, eid, uid, meta, arena);
                        return;
                    }
                }
                my_wide[bucket].push_back(WideEvRow{eid, uid, ingest_sec, ts_ms, price_cents, cat_id});
            });
        }
        my_ev.finish_all();
        auto tcl0 = Clock::now();
        prof_s1_ev_ns.fetch_add(
            static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(tcl0 - tev0).count()),
            std::memory_order_relaxed);
        // Keep early finishers available for clicks. Otherwise the identity worker
        // can end up scanning all the clicks alone while its peers wait for Stage 2.
        identity_ready.wait(false, std::memory_order_acquire);
        auto click_scan_start = Clock::now();
        scan_clicks_for(tid, io_buf);
        person_cl[tid].finish_all();
        prof_s1_cl_ns.fetch_add(
            static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - click_scan_start).count()),
            std::memory_order_relaxed);
    };

    team.run([&](size_t tid) {
        if (tid >= num_threads) return;
        if (overlap_dsu && tid == 0) {
            auto td = Clock::now();
            parse_links_and_build_identity_single(link_files.get(), n_ln_files, total_link_bytes, fallback, identity_map);
            dsu_ms = elapsed_sec(td) * 1000.0;
            identity_ready.store(true, std::memory_order_release);
            identity_ready.notify_all();
        }
        scan_events_body(tid);
    });
    if (prefetcher.joinable()) prefetcher.join();
    std::fprintf(stderr, "Stage 1 in %.3fs (s0=%.1fms [team=%.1f open=%.1f cat=%.1f] dsu=%.1fms%s | avg/th: ev=%.1fms cl=%.1fms)\n",
                 elapsed_sec(t_s1),
                 std::chrono::duration<double, std::milli>(t_s0_end - t_total).count(),
                 ms_team, ms_open, ms_cat,
                 dsu_ms, overlap_dsu ? "[ovl]" : "[par]",
                 prof_s1_ev_ns.load() / (1e6 * num_threads),
                 prof_s1_cl_ns.load() / (1e6 * num_threads));

    // Stage 2: Deduplicate events and partition the surviving payloads by person.
    auto t_s2 = Clock::now();
    std::vector<BucketSet<PersonEv16>> person_ev(num_threads);
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
        std::vector<DedupSlot> table;
        std::vector<uint8_t*> slab_ptrs;
        std::vector<WideEvRow*> wide_ptrs;
        auto& my_ev = person_ev[tid];
        auto& my_wev = wide_person_ev[tid];

        auto get_ev_ingest = [&](const EvPay& r) -> int64_t {
            if (__builtin_expect(r.uid == 0, 0)) return static_cast<int64_t>(r.meta);
            uint32_t ts_off = static_cast<uint32_t>(r.meta >> 33);
            int64_t ts_sec = __builtin_expect(ws_whole_sec, 1)
                                 ? (ws_sec + static_cast<int64_t>(ts_off / 1000U))
                                 : ((window_start + static_cast<int64_t>(ts_off)) / 1000LL);
            int64_t delta = static_cast<int64_t>((r.meta >> 15) & 0x3FFFFULL) - 65536LL;
            return ts_sec + delta;
        };

        auto get_slot_eid = [&](uint32_t loc) -> uint64_t {
            if (__builtin_expect((loc & 0x80000000U) == 0, 1)) {
                return reinterpret_cast<const uint64_t*>(slab_ptrs[(loc >> 12) - 1])[loc & 0xFFFU];
            }
            return wide_ptrs[loc & 0x7FFFFFFFU]->eid;
        };

        auto get_slot_ingest = [&](uint32_t loc) -> int64_t {
            if (__builtin_expect((loc & 0x80000000U) == 0, 1)) {
                const EvPay* pays = reinterpret_cast<const EvPay*>(slab_ptrs[(loc >> 12) - 1] + pay_off);
                return get_ev_ingest(pays[loc & 0xFFFU]);
            }
            return wide_ptrs[loc & 0x7FFFFFFFU]->ingest_sec;
        };

        auto kill_slot = [&](uint32_t loc) {
            if (__builtin_expect((loc & 0x80000000U) == 0, 1)) {
                EvPay* pays = reinterpret_cast<EvPay*>(slab_ptrs[(loc >> 12) - 1] + pay_off);
                pays[loc & 0xFFFU].uid = 0;
            } else {
                wide_ptrs[loc & 0x7FFFFFFFU]->uid = 0;
            }
        };

        uint64_t loc_p1_ns = 0, loc_p2_ns = 0;
        while (true) {
            size_t b = next_db.fetch_add(1, std::memory_order_relaxed);
            if (b >= num_buckets) break;

            size_t total_recs = 0, total_wide = 0;
            for (size_t t = 0; t < num_threads; ++t) {
                total_recs += ev_buckets[t].lists[b].total_count;
                total_wide += wide_ev_buckets[t][b].size();
            }
            if (total_recs + total_wide == 0) continue;
            if (total_wide > 0) has_any_wide.store(true, std::memory_order_relaxed);

            auto tp1 = Clock::now();
            // Lower load factor (~32% on 10M -> 1 MB L2-resident) cuts linear-probe collisions & mispredicts!
            size_t cap = next_pow2(std::max<size_t>(32, (total_recs + total_wide) * 2));
            if (table.size() < cap) table.resize(cap);
            std::memset(table.data(), 0, cap * sizeof(DedupSlot));
            size_t mask = cap - 1;
            slab_ptrs.clear();
            wide_ptrs.clear();

            // Pass 1: SOA streaming ONLY eids[] + 16-deep hash ring buffer (32-slot ring -> zero aliasing!)
            uint64_t h_ring[32];
            for (size_t t = 0; t < num_threads; ++t) {
                auto& lst = ev_buckets[t].lists[b];
                for (size_t si = 0; si < lst.slabs.size(); ++si) {
                    uint8_t* raw = lst.slabs[si];
                    const uint64_t* eids = reinterpret_cast<const uint64_t*>(raw);
                    EvPay* pays = reinterpret_cast<EvPay*>(raw + pay_off);
                    uint32_t cnt = lst.counts[si];
                    uint32_t sidx_p1 = static_cast<uint32_t>(slab_ptrs.size()) + 1;
                    slab_ptrs.push_back(raw);

                    uint32_t pre = std::min<uint32_t>(16, cnt);
                    for (uint32_t p = 0; p < pre; ++p) {
                        uint64_t nh = fast::hash_u64(eids[p]);
                        h_ring[p] = nh;
                        __builtin_prefetch(&table[static_cast<size_t>(nh) & mask], 1, 1);
                    }
                    for (uint32_t i = 0; i < cnt; ++i) {
                        uint64_t h = h_ring[i & 31];
                        if (__builtin_expect(i + 16 < cnt, 1)) {
                            uint64_t nh = fast::hash_u64(eids[i + 16]);
                            h_ring[(i + 16) & 31] = nh;
                            __builtin_prefetch(&table[static_cast<size_t>(nh) & mask], 1, 1);
                        }
                        uint64_t eid = eids[i];
                        uint32_t tag = static_cast<uint32_t>(h >> 23);
                        size_t slot = static_cast<size_t>(h) & mask;
                        while (true) {
                            DedupSlot& ds = table[slot];
                            if (__builtin_expect(ds.loc == 0, 1)) {
                                ds.tag = tag;
                                ds.loc = (sidx_p1 << 12) | i;
                                break;
                            }
                            if (__builtin_expect(ds.tag == tag && get_slot_eid(ds.loc) == eid, 0)) {
                                int64_t cur_ing = get_ev_ingest(pays[i]);
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

            if (__builtin_expect(total_wide > 0, 0)) {
                for (size_t t = 0; t < num_threads; ++t) {
                    auto& wv = wide_ev_buckets[t][b];
                    for (auto& wr : wv) {
                        uint32_t wloc = 0x80000000U | static_cast<uint32_t>(wide_ptrs.size());
                        wide_ptrs.push_back(&wr);
                        uint64_t eid = wr.eid;
                        uint64_t h = fast::hash_u64(eid);
                        uint32_t tag = static_cast<uint32_t>(h >> 23);
                        size_t slot = static_cast<size_t>(h) & mask;
                        while (true) {
                            DedupSlot& ds = table[slot];
                            if (ds.loc == 0) {
                                ds.tag = tag;
                                ds.loc = wloc;
                                break;
                            }
                            if (ds.tag == tag && get_slot_eid(ds.loc) == eid) {
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
            }
            auto tp2 = Clock::now();
            loc_p1_ns += static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(tp2 - tp1).count());

            // Pass 2: SOA streaming ONLY pays[] (16B aligned) + Batch-64 Compacted Bloom -> Slot!
            uint64_t b_uid[64], b_h[64], b_meta[64];
            for (size_t t = 0; t < num_threads; ++t) {
                auto& lst = ev_buckets[t].lists[b];
                const auto& pv = ev_purchases[t][b];
                size_t nb = 0;
                auto flush_ev_batch = [&]() {
                    identity_map.resolve_batch(b_uid, b_h, nb);
                    for (size_t k = 0; k < nb; ++k) {
                        if (__builtin_expect(k + 8 < nb, 1)) {
                            __builtin_prefetch(my_ev.cursors[static_cast<size_t>(b_h[k + 8] >> part_shift)].cur, 1, 1);
                        }
                        uint64_t meta = b_meta[k];
                        uint32_t pidx = static_cast<uint32_t>(meta & 0x7FFFU);
                        uint32_t pc = __builtin_expect(pidx == 0, 1) ? 0U : pv[pidx - 1];
                        uint64_t ts_off = meta >> 33;
                        size_t p_part = static_cast<size_t>(b_h[k] >> part_shift);
                        my_ev.push(p_part, PersonEv16{b_uid[k], (ts_off << 32) | pc}, arena);
                    }
                    nb = 0;
                };
                for (size_t si = 0; si < lst.slabs.size(); ++si) {
                    const EvPay* pays = reinterpret_cast<const EvPay*>(lst.slabs[si] + pay_off);
                    uint32_t cnt = lst.counts[si];
                    for (uint32_t i = 0; i < cnt; ++i) {
                        uint64_t uid = pays[i].uid;
                        if (__builtin_expect(uid == 0, 0)) continue;
                        b_uid[nb] = uid;
                        b_meta[nb] = pays[i].meta;
                        if (__builtin_expect(++nb == 64, 0)) flush_ev_batch();
                    }
                }
                if (nb > 0) flush_ev_batch();
                ev_buckets[t].recycle_bucket(b, arena);

                if (__builtin_expect(!wide_ev_buckets[t][b].empty(), 0)) {
                    for (const auto& wr : wide_ev_buckets[t][b]) {
                        if (wr.uid == 0) continue;
                        uint64_t pid_h = 0;
                        uint64_t pid = identity_map.resolve(wr.uid, pid_h);
                        size_t p_part = static_cast<size_t>(pid_h >> part_shift);
                        my_wev[p_part].push_back(WidePersonEv{pid, wr.ts_ms, wr.price_cents, wr.cat_id});
                    }
                }
            }
            loc_p2_ns += static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - tp2).count());
        }
        my_ev.finish_all();
        prof_p1_ns.fetch_add(loc_p1_ns, std::memory_order_relaxed);
        prof_p2_ns.fetch_add(loc_p2_ns, std::memory_order_relaxed);
    });
    std::fprintf(stderr, "Stage 2 in %.3fs (avg/thread: dedup_p1=%.1fms emit_p2=%.1fms)\n",
                 elapsed_sec(t_s2),
                 prof_p1_ns.load() / (1e6 * num_threads),
                 prof_p2_ns.load() / (1e6 * num_threads));

    // Stage 3: Zero-double-copy CSR + hybrid sort + compact 16B CampDayAccum + fast Parquet write!
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

    team.run([&](size_t tid) {
        if (tid >= num_threads) return;
        UserDailyWriter ud_writer;
        ud_writer.open(out_dir + "/user_daily/part-" + std::to_string(tid) + ".parquet", est_ud_per_thread, use_events_i32);
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
            uint64_t start_cl_off; // start_ms - click_start (eliminates click_start + in inner loops!)
            int64_t rev_cents;
            int32_t day;
            uint32_t day_off;
        };
        struct CampAssistState {
            uint32_t gen;
            uint32_t next_sess;
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

        std::vector<SessionRec> u_sessions;
        u_sessions.reserve(1024);
        std::vector<int64_t> cat_rev(num_cats, 0);
        std::vector<uint32_t> touched_cats;
        touched_cats.reserve(num_cats);

        std::vector<CampAssistState> camp_state(num_camps, CampAssistState{0, 0});
        std::vector<uint64_t> radix_tmp;
        uint32_t cur_gen = 0;
        char pid_buf[16];
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

            // E / 30 people -> ~2600 users/part on 10M -> 8K slots = 128 KB (4x smaller, 100% L2-resident!)
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
                // Direct slab -> ev_arr64 scatter (eliminates tmp_ev64 intermediate copy!)
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
            // Direct slab -> cl_arr scatter (eliminates tmp_cl intermediate copy!)
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

            for (size_t u = 0; u < num_users; ++u) {
                u_sessions.clear();
                touched_cats.clear();
                bool pid_inline = ((u_pids[u] >> 63) == 0);
                std::string_view pid_sv = fallback.key_to_sv(u_pids[u], pid_buf);

                auto flush_day = [&](int32_t cur_day, int32_t day_sessions, int64_t day_events, int64_t day_rev) {
                    const std::string* top_cat = nullptr;
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
                        top_cat = &catalogs.categories[best_cat];
                    }
                    touched_cats.clear();
                    ud_writer.add_row(pid_sv, pid_inline, cur_day, day_sessions, day_events, day_rev, top_cat);
                };

                if (__builtin_expect(!part_wide, 1)) {
                    uint64_t* eb_raw = ev_arr64.data() + ev_off[u];
                    size_t en = static_cast<size_t>(ev_off[u + 1] - ev_off[u]);
                    const uint64_t* eb = sort_ev64(eb_raw, en, radix_tmp);

                    int64_t sess_off = static_cast<int64_t>(eb[0] >> 32);
                    int64_t prev_off = sess_off;
                    int32_t sess_day = min_day;
                    int64_t next_day_off = day0_end_off;
                    while (sess_off >= next_day_off) {
                        sess_day++;
                        next_day_off += DAY_MS;
                    }
                    int64_t sess_rev = 0;
                    int32_t cur_day = sess_day;
                    int32_t day_sessions = 1;
                    int64_t day_events = 0;
                    int64_t day_rev = 0;

                    for (size_t i = 0; i < en; ++i) {
                        uint64_t ev = eb[i];
                        int64_t off = static_cast<int64_t>(ev >> 32);
                        if (i > 0 && off - prev_off > SESSION_GAP_MS) {
                            u_sessions.push_back(SessionRec{static_cast<uint64_t>(sess_off + WEEK_MS), sess_rev, sess_day, static_cast<uint32_t>(sess_day - min_day)});
                            sess_off = off;
                            while (sess_off >= next_day_off) {
                                sess_day++;
                                next_day_off += DAY_MS;
                            }
                            sess_rev = 0;
                            if (sess_day != cur_day) {
                                flush_day(cur_day, day_sessions, day_events, day_rev);
                                cur_day = sess_day;
                                day_sessions = 1;
                                day_events = 0;
                                day_rev = 0;
                            } else {
                                day_sessions++;
                            }
                        }
                        prev_off = off;
                        day_events++;
                        uint32_t price = static_cast<uint32_t>(ev) & 0x00FFFFFFU;
                        if (__builtin_expect(price > 0, 0)) {
                            uint32_t cid = static_cast<uint32_t>(ev) >> 24;
                            sess_rev += price;
                            day_rev += price;
                            if (cat_rev[cid] == 0) touched_cats.push_back(cid);
                            cat_rev[cid] += price;
                        }
                    }
                    u_sessions.push_back(SessionRec{static_cast<uint64_t>(sess_off + WEEK_MS), sess_rev, sess_day, static_cast<uint32_t>(sess_day - min_day)});
                    flush_day(cur_day, day_sessions, day_events, day_rev);
                } else {
                    WideEvItem* eb = wev_arr.data() + ev_off[u];
                    WideEvItem* ee = wev_arr.data() + ev_off[u + 1];
                    size_t en = static_cast<size_t>(ee - eb);
                    if (en > 1) {
                        std::sort(eb, ee, [](const WideEvItem& a, const WideEvItem& b) {
                            return a.ts_ms < b.ts_ms;
                        });
                    }
                    int64_t sess_start = eb[0].ts_ms;
                    int64_t prev_ts = eb[0].ts_ms;
                    int32_t sess_day = fast::floor_div_day(sess_start);
                    int64_t sess_rev = 0;
                    int32_t cur_day = sess_day;
                    int32_t day_sessions = 1;
                    int64_t day_events = 0;
                    int64_t day_rev = 0;

                    for (size_t i = 0; i < en; ++i) {
                        const auto& e = eb[i];
                        if (i > 0 && e.ts_ms - prev_ts > SESSION_GAP_MS) {
                            u_sessions.push_back(SessionRec{static_cast<uint64_t>(sess_start - click_start), sess_rev, sess_day, static_cast<uint32_t>(sess_day - min_day)});
                            sess_start = e.ts_ms;
                            sess_day = fast::floor_div_day(sess_start);
                            sess_rev = 0;
                            if (sess_day != cur_day) {
                                flush_day(cur_day, day_sessions, day_events, day_rev);
                                cur_day = sess_day;
                                day_sessions = 1;
                                day_events = 0;
                                day_rev = 0;
                            } else {
                                day_sessions++;
                            }
                        }
                        prev_ts = e.ts_ms;
                        day_events++;
                        if (e.price_cents > 0) {
                            sess_rev += e.price_cents;
                            day_rev += e.price_cents;
                            if (cat_rev[e.cat_id] == 0) touched_cats.push_back(e.cat_id);
                            cat_rev[e.cat_id] += e.price_cents;
                        }
                    }
                    u_sessions.push_back(SessionRec{static_cast<uint64_t>(sess_start - click_start), sess_rev, sess_day, static_cast<uint32_t>(sess_day - min_day)});
                    flush_day(cur_day, day_sessions, day_events, day_rev);
                }

                size_t S = u_sessions.size();
                ClickItem* cb_raw = cl_arr.data() + cl_off[u];
                size_t C = static_cast<size_t>(cl_off[u + 1] - cl_off[u]);
                const bool fast_dense = cacc_dense && !part_wide;

                if (C == 0) {
                    if (__builtin_expect(fast_dense, 1)) {
                        for (size_t si = 0; si < S; ++si) {
                            cacc.add_session_dense(0, u_sessions[si].day_off, u_sessions[si].rev_cents);
                        }
                    } else {
                        for (size_t si = 0; si < S; ++si) {
                            cacc.add_session(0, u_sessions[si].day, u_sessions[si].rev_cents);
                        }
                    }
                    continue;
                }

                constexpr uint64_t WEEK_U64 = static_cast<uint64_t>(WEEK_MS);
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

                const ClickItem* cb = sort_clicks(cb_raw, C, fallback, cl_radix_tmp);

                // Early exit when earliest click is strictly after user's last session!
                if (__builtin_expect(u_sessions[S - 1].start_cl_off < (cb[0].ts_camp >> 20), 0)) {
                    if (__builtin_expect(fast_dense, 1)) {
                        for (size_t si = 0; si < S; ++si) {
                            cacc.add_session_dense(0, u_sessions[si].day_off, u_sessions[si].rev_cents);
                        }
                    } else {
                        for (size_t si = 0; si < S; ++si) {
                            cacc.add_session(0, u_sessions[si].day, u_sessions[si].rev_cents);
                        }
                    }
                    continue;
                }

                size_t ci = 0;
                uint64_t best_ts_off = UINT64_MAX;
                uint32_t best_camp = 0;
                for (size_t si = 0; si < S; ++si) {
                    uint64_t soff = u_sessions[si].start_cl_off;
                    while (ci < C) {
                        uint64_t cts_off = cb[ci].ts_camp >> 20;
                        if (cts_off > soff) break;
                        if (best_ts_off == UINT64_MAX || cts_off > best_ts_off) {
                            best_ts_off = cts_off;
                            best_camp = static_cast<uint32_t>(cb[ci].ts_camp & 0xFFFFFU);
                        }
                        ci++;
                    }
                    uint32_t credited = (best_ts_off != UINT64_MAX && soff - best_ts_off <= WEEK_U64)
                                            ? best_camp
                                            : 0;
                    if (__builtin_expect(fast_dense, 1)) {
                        cacc.add_session_dense(credited, u_sessions[si].day_off, u_sessions[si].rev_cents);
                    } else {
                        cacc.add_session(credited, u_sessions[si].day, u_sessions[si].rev_cents);
                    }
                }

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
    // GCC writes its training counters on normal exit. Keep the same control
    // flow in both builds so the optimized build can use the recorded profile.
    if (std::getenv("SHOPPING_PGO_TRAIN")) return 0;
    _exit(0);
}
