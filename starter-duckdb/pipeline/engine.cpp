#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <fcntl.h>
#include <limits>
#include <string>
#include <string_view>
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
constexpr size_t MAX_BUCKETS = 256;
constexpr size_t READ_CHUNK_SIZE = 2 * 1024 * 1024; // 2 MB pread chunks (zero page faults!)

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

struct OpenFile {
    int fd = -1;
    size_t size = 0;
    size_t header_end = 0;

    bool open(const std::string& path) {
        fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) return false;
        struct stat st;
        if (fstat(fd, &st) != 0 || st.st_size <= 0) {
            ::close(fd);
            fd = -1;
            return st.st_size == 0;
        }
        size = static_cast<size_t>(st.st_size);
#if defined(POSIX_FADV_SEQUENTIAL)
        posix_fadvise(fd, 0, 0, POSIX_FADV_SEQUENTIAL);
#endif
        // Locate end of header line.
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

    void close() {
        if (fd >= 0) {
#if defined(POSIX_FADV_DONTNEED)
            posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
#endif
            ::close(fd);
            fd = -1;
        }
        size = 0;
        header_end = 0;
    }

    ~OpenFile() { close(); }
};

struct ReadChunk {
    size_t file_idx;
    size_t start; // >= header_end
    size_t end;   // <= file.size
    bool is_first;
};

std::vector<ReadChunk> build_read_chunks(const std::vector<OpenFile>& files, size_t chunk_sz) {
    std::vector<ReadChunk> chunks;
    for (size_t fi = 0; fi < files.size(); ++fi) {
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

// Read exact line-aligned slice for ReadChunk into reusable thread-local buffer (zero page faults!).
// Boundary rule:
// - If is_first: starts at ch.start (right after header '\n').
// - Else: starts right after the first '\n' at index >= ch.start - 1.
// - If ch.end == file.size: ends at file.size.
// - Else: ends right after (inclusive) the first '\n' at index >= ch.end - 1 (or EOF if none).
std::string_view read_aligned_chunk(const OpenFile& f, const ReadChunk& ch, std::vector<char>& buf) {
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
        if (!nl) {
            // Line spans past this chunk's entire read; keep reading until '\n' or EOF.
            size_t file_pos = read_from + got;
            char tmp[4096];
            while (file_pos < f.size) {
                size_t to_read = std::min(sizeof(tmp), f.size - file_pos);
                ssize_t r = ::pread(f.fd, tmp, to_read, static_cast<off_t>(file_pos));
                if (r <= 0) return {};
                const char* n2 = static_cast<const char*>(std::memchr(tmp, '\n', static_cast<size_t>(r)));
                if (n2) {
                    size_t next_start = file_pos + static_cast<size_t>(n2 - tmp) + 1;
                    if (next_start >= ch.end) return {};
                    // Rare: single line > chunk_size
                    return {};
                }
                file_pos += static_cast<size_t>(r);
            }
            return {};
        }
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
            // Line crossing ch.end is longer than 4 KB; read until '\n' or EOF.
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

// Flat string arena + open-addressing map: string_view -> uint32_t value.
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
};

// Identity Map with a tiny L3-resident blocked Bloom filter and compact 16-byte slots.
class IdentityMap {
    struct Slot {
        uint64_t hash = 0; // 0 = empty; |= 1
        uint32_t src_off = 0;
        uint32_t dst_idx = 0; // index into canon_
    };
    struct Canon {
        uint64_t hash;
        uint32_t off;
        uint32_t len;
    };
    std::vector<char> arena_;
    std::vector<uint32_t> src_lens_;
    std::vector<Canon> canon_;
    std::vector<Slot> slots_;
    std::vector<uint64_t> bloom_;
    size_t mask_ = 0;
    size_t bloom_mask_ = 0;
    size_t count_ = 0;

public:
    bool empty() const { return count_ == 0; }

    void build(const std::vector<char>& str_arena,
               const std::vector<uint32_t>& offs,
               const std::vector<uint32_t>& lens,
               const std::vector<uint32_t>& canonical_node) {
        size_t n = offs.size();
        size_t mapped = 0;
        for (size_t i = 0; i < n; ++i) {
            if (canonical_node[i] != i) mapped++;
        }
        if (mapped == 0) return;

        size_t bloom_words = next_pow2(std::max<size_t>(64, (mapped * 12) / 64));
        if (bloom_words > (1U << 19)) bloom_words = (1U << 19); // max 4 MB
        bloom_.assign(bloom_words, 0);
        bloom_mask_ = bloom_words - 1;

        size_t cap = next_pow2(std::max<size_t>(16, mapped * 5 / 2));
        slots_.assign(cap, Slot{});
        src_lens_.assign(cap, 0);
        mask_ = cap - 1;
        arena_ = str_arena;

        std::vector<uint32_t> node_to_canon(n, UINT32_MAX);
        for (size_t i = 0; i < n; ++i) {
            uint32_t c = canonical_node[i];
            if (c == i) continue;
            uint32_t cidx = node_to_canon[c];
            if (cidx == UINT32_MAX) {
                cidx = static_cast<uint32_t>(canon_.size());
                node_to_canon[c] = cidx;
                canon_.push_back(Canon{
                    fast::hash_bytes(arena_.data() + offs[c], lens[c]),
                    offs[c],
                    lens[c]
                });
            }
            uint64_t raw_h = fast::hash_bytes(arena_.data() + offs[i], lens[i]);
            uint64_t h = raw_h | 1ULL;

            uint64_t bmask = (1ULL << (raw_h & 63)) | (1ULL << ((raw_h >> 6) & 63)) |
                             (1ULL << ((raw_h >> 12) & 63));
            bloom_[(raw_h >> 18) & bloom_mask_] |= bmask;

            size_t idx = static_cast<size_t>(h) & mask_;
            while (slots_[idx].hash) idx = (idx + 1) & mask_;
            slots_[idx] = Slot{h, offs[i], cidx};
            src_lens_[idx] = lens[i];
            count_++;
        }
    }

    std::string_view resolve(std::string_view uid, uint64_t& out_hash) const {
        uint64_t raw_h = fast::hash_sv(uid);
        if (__builtin_expect(count_ == 0, 0)) {
            out_hash = raw_h;
            return uid;
        }
        uint64_t bmask = (1ULL << (raw_h & 63)) | (1ULL << ((raw_h >> 6) & 63)) |
                         (1ULL << ((raw_h >> 12) & 63));
        if (__builtin_expect((bloom_[(raw_h >> 18) & bloom_mask_] & bmask) != bmask, 1)) {
            out_hash = raw_h;
            return uid;
        }
        uint64_t h = raw_h | 1ULL;
        size_t idx = static_cast<size_t>(h) & mask_;
        while (slots_[idx].hash) {
            const Slot& s = slots_[idx];
            if (s.hash == h && src_lens_[idx] == uid.size() &&
                std::memcmp(arena_.data() + s.src_off, uid.data(), uid.size()) == 0) {
                const Canon& cn = canon_[s.dst_idx];
                out_hash = cn.hash;
                return std::string_view(arena_.data() + cn.off, cn.len);
            }
            idx = (idx + 1) & mask_;
        }
        out_hash = raw_h;
        return uid;
    }
};

// Zero-realloc chunked byte buffer with small inline staging buffer and slab recycling!
struct SlabPool {
    std::vector<uint8_t*> free_slabs;
    size_t slab_cap = 65536;

    uint8_t* alloc(size_t need) {
        if (need <= slab_cap && !free_slabs.empty()) {
            uint8_t* p = free_slabs.back();
            free_slabs.pop_back();
            return p;
        }
        return static_cast<uint8_t*>(std::malloc(std::max(slab_cap, need)));
    }

    void recycle(uint8_t* p, size_t cap) {
        if (cap == slab_cap && free_slabs.size() < 4096) {
            free_slabs.push_back(p);
        } else {
            std::free(p);
        }
    }

    void clear() {
        for (uint8_t* p : free_slabs) std::free(p);
        free_slabs.clear();
    }

    ~SlabPool() { clear(); }
};

struct ChunkedBuf {
    static constexpr size_t MAX_SLAB_CAP = 1U << 18; // 256 KB
    std::vector<uint8_t*> slabs;
    std::vector<size_t> slab_sizes;
    std::vector<size_t> slab_caps;
    size_t slab_cap = 65536;
    size_t cur_used = 0;
    uint32_t record_count = 0;
    uint8_t stage[256];
    size_t stage_used = 0;

    void init(size_t cap) {
        slab_cap = std::min(cap, MAX_SLAB_CAP);
        cur_used = slab_cap;
        record_count = 0;
        stage_used = 0;
    }

    void flush_stage(SlabPool* pool = nullptr) {
        if (stage_used == 0) return;
        append_direct(stage, stage_used, pool);
        stage_used = 0;
    }

    void append_direct(const void* data, size_t len, SlabPool* pool = nullptr) {
        if (len == 0) return;
        if (slabs.empty() || cur_used + len > slab_caps.back()) {
            size_t alloc_sz = std::max(slab_cap, len);
            if (!slabs.empty()) slab_sizes.back() = cur_used;
            uint8_t* p = pool ? pool->alloc(alloc_sz) : static_cast<uint8_t*>(std::malloc(alloc_sz));
            slabs.push_back(p);
            slab_sizes.push_back(0);
            slab_caps.push_back(alloc_sz);
            cur_used = 0;
        }
        std::memcpy(slabs.back() + cur_used, data, len);
        cur_used += len;
        slab_sizes.back() = cur_used;
    }

    void append_record(const void* data, size_t len, SlabPool* pool = nullptr) {
        if (__builtin_expect(stage_used + len <= sizeof(stage), 1)) {
            std::memcpy(stage + stage_used, data, len);
            stage_used += len;
            return;
        }
        flush_stage(pool);
        if (len <= sizeof(stage)) {
            std::memcpy(stage, data, len);
            stage_used = len;
        } else {
            append_direct(data, len, pool);
        }
    }

    void finish(SlabPool* pool = nullptr) { flush_stage(pool); }

    void release(SlabPool* pool = nullptr) {
        if (pool) {
            for (size_t i = 0; i < slabs.size(); ++i) pool->recycle(slabs[i], slab_caps[i]);
        } else {
            for (uint8_t* p : slabs) std::free(p);
        }
        slabs.clear();
        slab_sizes.clear();
        slab_caps.clear();
        cur_used = slab_cap;
        stage_used = 0;
        record_count = 0;
    }

    ~ChunkedBuf() { release(nullptr); }
};

static inline size_t encode_len(uint8_t* dst, size_t len) {
    if (__builtin_expect(len < 255, 1)) {
        dst[0] = static_cast<uint8_t>(len);
        return 1;
    }
    dst[0] = 0xFF;
    uint32_t l32 = static_cast<uint32_t>(len);
    std::memcpy(dst + 1, &l32, 4);
    return 5;
}

static inline size_t decode_len(const uint8_t*& p) {
    uint8_t b = *p++;
    if (__builtin_expect(b < 255, 1)) return b;
    uint32_t l32;
    std::memcpy(&l32, p, 4);
    p += 4;
    return l32;
}

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
    StringMap product_to_cat;
    std::vector<std::string> categories; // byte-wise sorted!
    uint32_t unknown_cat_id = 0;

    StringMap campaign_to_id;
    std::vector<std::string> campaign_names;    // 0 = "organic"
    std::vector<std::string> campaign_channels; // 0 = "organic"
};

void load_catalogs(const std::string& data_dir, Catalogs& cat) {
    // Products
    {
        std::vector<char> buf;
        std::vector<std::pair<std::string, std::string>> rows;
        std::vector<std::string> cats;
        cats.push_back("unknown");
        if (read_all_file(data_dir + "/products.csv", buf) && !buf.empty()) {
            const char* nl = static_cast<const char*>(std::memchr(buf.data(), '\n', buf.size()));
            if (nl) {
                size_t start = static_cast<size_t>(nl - buf.data()) + 1;
                rows.reserve(60000);
                fast::scan_csv_chunk<2>(buf.data() + start, buf.size() - start, [&](const fast::Field (&f)[2]) {
                    if (f[0].len == 0) return;
                    rows.emplace_back(std::string(f[0].ptr, f[0].len), std::string(f[1].ptr, f[1].len));
                    cats.emplace_back(f[1].ptr, f[1].len);
                });
            }
        }
        std::sort(cats.begin(), cats.end(), [](const std::string& a, const std::string& b) {
            return fast::less_sv(a, b);
        });
        cats.erase(std::unique(cats.begin(), cats.end()), cats.end());
        cat.categories = std::move(cats);
        StringMap cat_idx;
        cat_idx.reserve(cat.categories.size() + 4);
        for (uint32_t i = 0; i < cat.categories.size(); ++i) {
            cat_idx.insert(cat.categories[i], i);
            if (cat.categories[i] == "unknown") cat.unknown_cat_id = i;
        }
        cat.product_to_cat.reserve(rows.size() + 16);
        for (const auto& [pid, cname] : rows) {
            uint32_t cid = cat.unknown_cat_id;
            cat_idx.find(cname, cid);
            cat.product_to_cat.insert(pid, cid);
        }
    }

    // Campaigns
    {
        cat.campaign_names.push_back("organic");
        cat.campaign_channels.push_back("organic");
        std::vector<char> buf;
        std::vector<std::pair<std::string, std::string>> rows;
        if (read_all_file(data_dir + "/campaigns.csv", buf) && !buf.empty()) {
            const char* nl = static_cast<const char*>(std::memchr(buf.data(), '\n', buf.size()));
            if (nl) {
                size_t start = static_cast<size_t>(nl - buf.data()) + 1;
                rows.reserve(4096);
                fast::scan_csv_chunk<2>(buf.data() + start, buf.size() - start, [&](const fast::Field (&f)[2]) {
                    if (f[0].len == 0) return;
                    rows.emplace_back(std::string(f[0].ptr, f[0].len), std::string(f[1].ptr, f[1].len));
                });
            }
        }
        cat.campaign_to_id.reserve(rows.size() + 16);
        for (const auto& [cid, ch] : rows) {
            uint32_t id = static_cast<uint32_t>(cat.campaign_names.size());
            cat.campaign_names.push_back(cid);
            cat.campaign_channels.push_back(ch);
            cat.campaign_to_id.insert(cid, id);
        }
    }
}

struct LinkEdge {
    uint64_t ha;
    uint64_t hb;
    uint32_t off_a;
    uint32_t len_a;
    uint32_t off_b;
    uint32_t len_b;
};

struct LinkThreadBuf {
    std::vector<char> arena;
    std::vector<LinkEdge> edges;
};

void build_identity_from_links(const std::vector<LinkThreadBuf>& bufs, IdentityMap& id_map) {
    size_t total_edges = 0;
    for (const auto& b : bufs) total_edges += b.edges.size();
    if (total_edges == 0) return;

    struct NodeSlot {
        uint64_t hash = 0;
        uint32_t node = 0;
    };
    size_t est_nodes = std::max<size_t>(64, total_edges * 3 / 2);
    size_t cap = next_pow2(est_nodes * 2);
    std::vector<NodeSlot> table(cap);
    size_t mask = cap - 1;

    std::vector<char> str_arena;
    str_arena.reserve(total_edges * 20);
    std::vector<uint32_t> offs, lens, parent, rank_sz, min_node;
    offs.reserve(est_nodes);
    lens.reserve(est_nodes);
    parent.reserve(est_nodes);
    rank_sz.reserve(est_nodes);
    min_node.reserve(est_nodes);

    auto grow_table = [&]() {
        size_t ncap = table.size() * 2;
        std::vector<NodeSlot> nt(ncap);
        size_t nmask = ncap - 1;
        for (const auto& s : table) {
            if (!s.hash) continue;
            size_t idx = static_cast<size_t>(s.hash) & nmask;
            while (nt[idx].hash) idx = (idx + 1) & nmask;
            nt[idx] = s;
        }
        table.swap(nt);
        mask = nmask;
    };

    auto intern = [&](const char* p, uint32_t len, uint64_t raw_h) -> uint32_t {
        if ((offs.size() + 1) * 10 >= table.size() * 6) grow_table();
        uint64_t h = raw_h | 1ULL;
        size_t idx = static_cast<size_t>(h) & mask;
        while (table[idx].hash) {
            if (table[idx].hash == h) {
                uint32_t nid = table[idx].node;
                if (lens[nid] == len && std::memcmp(str_arena.data() + offs[nid], p, len) == 0) {
                    return nid;
                }
            }
            idx = (idx + 1) & mask;
        }
        uint32_t nid = static_cast<uint32_t>(offs.size());
        uint32_t off = static_cast<uint32_t>(str_arena.size());
        str_arena.insert(str_arena.end(), p, p + len);
        offs.push_back(off);
        lens.push_back(len);
        parent.push_back(nid);
        rank_sz.push_back(1);
        min_node.push_back(nid);
        table[idx] = NodeSlot{h, nid};
        return nid;
    };

    auto find_root = [&](uint32_t x) -> uint32_t {
        while (parent[x] != x) {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        return x;
    };

    for (const auto& b : bufs) {
        const char* base = b.arena.data();
        for (const auto& e : b.edges) {
            uint32_t u = find_root(intern(base + e.off_a, e.len_a, e.ha));
            uint32_t v = find_root(intern(base + e.off_b, e.len_b, e.hb));
            if (u == v) continue;
            if (rank_sz[u] < rank_sz[v]) std::swap(u, v);
            parent[v] = u;
            rank_sz[u] += rank_sz[v];
            uint32_t mu = min_node[u], mv = min_node[v];
            if (fast::cmp_bytes(str_arena.data() + offs[mv], lens[mv],
                                str_arena.data() + offs[mu], lens[mu]) < 0) {
                min_node[u] = mv;
            }
        }
    }

    size_t n = offs.size();
    std::vector<uint32_t> canonical(n);
    for (size_t i = 0; i < n; ++i) {
        canonical[i] = min_node[find_root(static_cast<uint32_t>(i))];
    }
    id_map.build(str_arena, offs, lens, canonical);
}

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
    std::vector<CampDayStats> dense_arr_;

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

public:
    void init(uint32_t num_camps, int32_t min_day, int32_t max_day) {
        num_camps_ = num_camps;
        base_day_ = min_day;
        int64_t span = static_cast<int64_t>(max_day) - static_cast<int64_t>(min_day) + 1;
        if (span > 0 && span <= 64 && static_cast<uint64_t>(num_camps) * static_cast<uint64_t>(span) <= 262144ULL) {
            dense_ = true;
            num_days_ = static_cast<int32_t>(span);
            dense_arr_.assign(static_cast<size_t>(num_camps) * static_cast<size_t>(num_days_), CampDayStats{});
        } else {
            dense_ = false;
        }
    }

    CampDayStats& get(uint32_t camp_id, int32_t day) {
        if (__builtin_expect(dense_ && day >= base_day_ && day < base_day_ + num_days_, 1)) {
            return dense_arr_[static_cast<size_t>(camp_id) * static_cast<size_t>(num_days_) +
                              static_cast<size_t>(day - base_day_)];
        }
        if ((map_count_ + 1) * 10 >= map_slots_.size() * 6) grow_map();
        uint64_t k = (static_cast<uint64_t>(camp_id) << 32) | static_cast<uint32_t>(day);
        uint64_t h = fast::wy_mum(k ^ 0x9e3779b97f4a7c15ULL, 0xbf58476d1ce4e5b9ULL) | 1ULL;
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

    template <typename Fn>
    void for_each_nonzero(Fn&& fn) const {
        if (dense_) {
            for (uint32_t c = 0; c < num_camps_; ++c) {
                const CampDayStats* row = dense_arr_.data() + static_cast<size_t>(c) * num_days_;
                for (int32_t d = 0; d < num_days_; ++d) {
                    if (row[d].sessions > 0 || row[d].assisted_sessions > 0) {
                        fn(c, base_day_ + d, row[d]);
                    }
                }
            }
        }
        for (const auto& s : map_slots_) {
            if (s.hash && (s.stats.sessions > 0 || s.stats.assisted_sessions > 0)) {
                fn(s.key.camp_id, s.key.day, s.stats);
            }
        }
    }

    void merge_from(const CampDayAccum& other) {
        if (dense_ && other.dense_ && num_days_ == other.num_days_ && base_day_ == other.base_day_) {
            size_t total = dense_arr_.size();
            for (size_t i = 0; i < total; ++i) {
                dense_arr_[i].sessions += other.dense_arr_[i].sessions;
                dense_arr_[i].purchasing_sessions += other.dense_arr_[i].purchasing_sessions;
                dense_arr_[i].revenue_cents += other.dense_arr_[i].revenue_cents;
                dense_arr_[i].assisted_sessions += other.dense_arr_[i].assisted_sessions;
            }
        } else {
            other.for_each_nonzero([&](uint32_t c, int32_t d, const CampDayStats& st) {
                auto& dst = get(c, d);
                dst.sessions += st.sessions;
                dst.purchasing_sessions += st.purchasing_sessions;
                dst.revenue_cents += st.revenue_cents;
                dst.assisted_sessions += st.assisted_sessions;
            });
            return;
        }
        for (const auto& s : other.map_slots_) {
            if (s.hash && (s.stats.sessions > 0 || s.stats.assisted_sessions > 0)) {
                auto& dst = get(s.key.camp_id, s.key.day);
                dst.sessions += s.stats.sessions;
                dst.purchasing_sessions += s.stats.purchasing_sessions;
                dst.revenue_cents += s.stats.revenue_cents;
                dst.assisted_sessions += s.stats.assisted_sessions;
            }
        }
    }
};

struct UserDailyWriter {
    static constexpr size_t RG_LIMIT = 250000;
    pq::ParquetFileWriter writer;
    std::vector<uint8_t> col_uid;
    std::vector<int32_t> col_day;
    std::vector<int64_t> col_sessions;
    std::vector<int64_t> col_events;
    std::vector<int64_t> col_revenue;
    std::vector<uint8_t> col_top_cat;
    std::vector<uint8_t> def_bits;
    size_t buffered = 0;

    static inline void append_str(std::vector<uint8_t>& buf, const char* data, uint32_t len) {
        const uint8_t* lb = reinterpret_cast<const uint8_t*>(&len);
        buf.insert(buf.end(), lb, lb + 4);
        const uint8_t* db = reinterpret_cast<const uint8_t*>(data);
        buf.insert(buf.end(), db, db + len);
    }

    bool open(const std::string& path) {
        col_uid.reserve(RG_LIMIT * 14);
        col_day.reserve(RG_LIMIT);
        col_sessions.reserve(RG_LIMIT);
        col_events.reserve(RG_LIMIT);
        col_revenue.reserve(RG_LIMIT);
        col_top_cat.reserve(RG_LIMIT * 6);
        def_bits.reserve((RG_LIMIT + 7) / 8 + 8);
        return writer.open(path, {
            {"user_id", pq::PT_BYTE_ARRAY, pq::REP_REQUIRED, pq::CT_UTF8},
            {"day", pq::PT_INT32, pq::REP_REQUIRED, pq::CT_DATE},
            {"sessions", pq::PT_INT64, pq::REP_REQUIRED, pq::CT_NONE},
            {"events", pq::PT_INT64, pq::REP_REQUIRED, pq::CT_NONE},
            {"revenue_cents", pq::PT_INT64, pq::REP_REQUIRED, pq::CT_NONE},
            {"top_category", pq::PT_BYTE_ARRAY, pq::REP_OPTIONAL, pq::CT_UTF8},
        });
    }

    void add_row(std::string_view uid, int32_t day, int64_t sessions, int64_t events,
                 int64_t revenue, const std::string* top_cat) {
        append_str(col_uid, uid.data(), static_cast<uint32_t>(uid.size()));
        col_day.push_back(day);
        col_sessions.push_back(sessions);
        col_events.push_back(events);
        col_revenue.push_back(revenue);

        if ((buffered & 7) == 0) def_bits.push_back(0);
        if (top_cat != nullptr) {
            def_bits.back() |= static_cast<uint8_t>(1U << (buffered & 7));
            append_str(col_top_cat, top_cat->data(), static_cast<uint32_t>(top_cat->size()));
        }
        buffered++;
        if (buffered >= RG_LIMIT) flush();
    }

    void flush() {
        if (buffered == 0) return;
        std::vector<pq::ParquetFileWriter::ColumnData> cols(6);
        cols[0] = {col_uid.data(), col_uid.size(), nullptr};
        cols[1] = {col_day.data(), col_day.size() * sizeof(int32_t), nullptr};
        cols[2] = {col_sessions.data(), col_sessions.size() * sizeof(int64_t), nullptr};
        cols[3] = {col_events.data(), col_events.size() * sizeof(int64_t), nullptr};
        cols[4] = {col_revenue.data(), col_revenue.size() * sizeof(int64_t), nullptr};
        cols[5] = {col_top_cat.data(), col_top_cat.size(), def_bits.data()};
        writer.write_row_group(static_cast<int64_t>(buffered), cols);
        col_uid.clear();
        col_day.clear();
        col_sessions.clear();
        col_events.clear();
        col_revenue.clear();
        col_top_cat.clear();
        def_bits.clear();
        buffered = 0;
    }

    bool close() {
        flush();
        return writer.close();
    }
};

bool write_campaign_daily(const std::string& path, const Catalogs& cat, const CampDayAccum& acc) {
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
        size_t pos = buf.size();
        buf.resize(pos + 4 + l);
        std::memcpy(buf.data() + pos, &l, 4);
        std::memcpy(buf.data() + pos + 4, s.data(), l);
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

    std::string data_dir = env_data ? env_data : "/data";
    std::string out_dir = env_out ? env_out : "/output";
    size_t hw_threads = std::thread::hardware_concurrency();
    if (hw_threads == 0) hw_threads = 1;
    size_t ncpu = env_ncpu ? static_cast<size_t>(std::max(1, std::atoi(env_ncpu))) : std::min<size_t>(8, hw_threads);

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
    int64_t base_sec = window_start / 1000LL;
    bool compact_ts = (window_end >= window_start) &&
                      (static_cast<uint64_t>(window_end - window_start) <= 0xFFFFFFFFULL);

    fast::DateTable dtab;
    dtab.precompute(window_start, window_end);

    Catalogs catalogs;
    load_catalogs(data_dir, catalogs);
    bool compact_cat = catalogs.categories.size() <= 255;

    auto event_paths = list_csv_files(data_dir + "/events");
    auto click_paths = list_csv_files(data_dir + "/clicks");
    auto link_paths = list_csv_files(data_dir + "/links");

    std::vector<OpenFile> event_files(event_paths.size());
    size_t total_event_bytes = 0;
    for (size_t i = 0; i < event_paths.size(); ++i) {
        event_files[i].open(event_paths[i]);
        total_event_bytes += event_files[i].size;
    }

    std::vector<OpenFile> click_files(click_paths.size());
    for (size_t i = 0; i < click_paths.size(); ++i) {
        click_files[i].open(click_paths[i]);
    }

    std::vector<OpenFile> link_files(link_paths.size());
    size_t total_link_bytes = 0;
    for (size_t i = 0; i < link_paths.size(); ++i) {
        link_files[i].open(link_paths[i]);
        total_link_bytes += link_files[i].size;
    }

    size_t num_threads = (total_event_bytes < 512 * 1024 && total_link_bytes < 512 * 1024) ? 1 : ncpu;
    size_t num_buckets = (total_event_bytes < 4 * 1024 * 1024) ? 16 : MAX_BUCKETS;
    unsigned bucket_shift = (num_buckets == 16) ? 60U : 56U;
    size_t slab_cap = (total_event_bytes < 4 * 1024 * 1024) ? 4096 : 65536; // <= 64KB avoids glibc mmap/munmap!

    // Stage 0b: Parse links/*.csv in parallel.
    auto link_chunks = build_read_chunks(link_files, 1 << 20);
    std::vector<LinkThreadBuf> link_bufs(num_threads);
    if (!link_chunks.empty()) {
        std::atomic<size_t> next_lc{0};
        auto parse_links_worker = [&](size_t tid) {
            std::vector<char> io_buf;
            auto& buf = link_bufs[tid];
            buf.arena.reserve(total_link_bytes / num_threads + 1024);
            buf.edges.reserve(total_link_bytes / (24 * num_threads) + 64);
            while (true) {
                size_t ci = next_lc.fetch_add(1, std::memory_order_relaxed);
                if (ci >= link_chunks.size()) break;
                const auto& ch = link_chunks[ci];
                auto sv = read_aligned_chunk(link_files[ch.file_idx], ch, io_buf);
                if (sv.empty()) continue;
                fast::scan_csv_chunk<2>(sv.data(), sv.size(), [&](const fast::Field (&f)[2]) {
                    if (f[0].len == 0 || f[1].len == 0) return;
                    if (f[0].len == f[1].len && std::memcmp(f[0].ptr, f[1].ptr, f[0].len) == 0) return;
                    uint64_t ha = fast::hash_bytes(f[0].ptr, f[0].len);
                    uint64_t hb = fast::hash_bytes(f[1].ptr, f[1].len);
                    uint32_t oa = static_cast<uint32_t>(buf.arena.size());
                    buf.arena.insert(buf.arena.end(), f[0].ptr, f[0].ptr + f[0].len);
                    uint32_t ob = static_cast<uint32_t>(buf.arena.size());
                    buf.arena.insert(buf.arena.end(), f[1].ptr, f[1].ptr + f[1].len);
                    buf.edges.push_back(LinkEdge{ha, hb, oa, static_cast<uint32_t>(f[0].len),
                                                 ob, static_cast<uint32_t>(f[1].len)});
                });
            }
        };
        if (num_threads == 1) {
            parse_links_worker(0);
        } else {
            std::vector<std::thread> pool;
            pool.reserve(num_threads - 1);
            for (size_t t = 1; t < num_threads; ++t) pool.emplace_back(parse_links_worker, t);
            parse_links_worker(0);
            for (auto& th : pool) th.join();
        }
    }
    for (auto& f : link_files) f.close();

    // Stage 1: Overlap DSU + IdentityMap build (Thread 0) with parallel events/*.csv scanning!
    auto t_s1 = Clock::now();
    auto event_chunks = build_read_chunks(event_files, READ_CHUNK_SIZE);
    std::vector<std::vector<ChunkedBuf>> ev_buckets(num_threads);
    for (size_t t = 0; t < num_threads; ++t) {
        ev_buckets[t].resize(num_buckets);
        for (size_t b = 0; b < num_buckets; ++b) ev_buckets[t][b].init(slab_cap);
    }

    IdentityMap identity_map;
    std::atomic<size_t> next_ec{0};

    // Event bucket record binary layout (guaranteed contiguous inside ONE slab!):
    //   [0..3]  uint32_t eid_h32
    //   [4]     uint8_t flags: bit0=valid, bit1=wide_ingest(8B vs 4B)
    //   [5..]   int32_t ingest_off (or int64_t ingest_sec if bit1)
    //           varint eid_len + eid bytes
    //   if valid:
    //           uint32_t ts_off (if compact_ts) else int64_t ts_ms
    //           uint32_t (cat<<24 | price) (if compact_cat) else (uint32_t price + uint16_t cat)
    //           varint uid_len + uid bytes
    auto scan_events_worker = [&](size_t tid) {
        std::vector<char> io_buf;
        io_buf.reserve(READ_CHUNK_SIZE + 8192);
        std::vector<uint8_t> big_rec;
        uint8_t rec_tmp[256];
        auto& my_buckets = ev_buckets[tid];

        while (true) {
            size_t ci = next_ec.fetch_add(1, std::memory_order_relaxed);
            if (ci >= event_chunks.size()) break;
            const auto& ch = event_chunks[ci];
            auto sv = read_aligned_chunk(event_files[ch.file_idx], ch, io_buf);
            if (sv.empty()) continue;

            fast::scan_csv_chunk<7>(sv.data(), sv.size(), [&](const fast::Field (&f)[7]) {
                if (__builtin_expect(f[0].len == 0, 0)) return;

                int64_t ingest_ms = 0;
                if (__builtin_expect(!fast::parse_f1_ms(f[3].ptr, f[3].len, dtab, ingest_ms), 0)) {
                    return;
                }
                int64_t ingest_sec = ingest_ms / 1000LL;
                int64_t ingest_diff = ingest_sec - base_sec;
                bool wide_ingest = (ingest_diff < std::numeric_limits<int32_t>::min() ||
                                    ingest_diff > std::numeric_limits<int32_t>::max());

                uint64_t eid_h = fast::hash_bytes(f[0].ptr, f[0].len);
                size_t bucket = static_cast<size_t>(eid_h >> bucket_shift);
                uint32_t eid_h32 = static_cast<uint32_t>(eid_h);

                bool valid = false;
                int64_t ts_ms = 0;
                uint32_t price_cents = 0;
                uint32_t cat_id = 0;

                if (__builtin_expect(f[1].len > 0 && f[5].len > 0, 1)) {
                    uint8_t etype = 0;
                    if (f[4].len == 4) {
                        uint32_t w;
                        std::memcpy(&w, f[4].ptr, 4);
                        if (w == 0x77656976U || w == 0x74726163U) etype = 1; // "view" | "cart"
                    } else if (f[4].len == 8) {
                        uint64_t w;
                        std::memcpy(&w, f[4].ptr, 8);
                        if (w == 0x6573616863727570ULL) etype = 2; // "purchase"
                    }

                    if (__builtin_expect(etype != 0, 1)) {
                        if (__builtin_expect(fast::parse_timestamp_ms(f[2].ptr, f[2].len, dtab, ts_ms), 1)) {
                            if (__builtin_expect(ts_ms >= window_start && ts_ms < window_end &&
                                                     (ingest_ms - ts_ms) <= DAY_MS, 1)) {
                                if (etype == 1) {
                                    valid = true;
                                } else if (__builtin_expect(fast::parse_price_cents(f[6].ptr, f[6].len, price_cents), 1)) {
                                    valid = true;
                                    cat_id = catalogs.unknown_cat_id;
                                    catalogs.product_to_cat.find(std::string_view(f[5].ptr, f[5].len), cat_id);
                                }
                            }
                        }
                    }
                }

                ChunkedBuf& cb = my_buckets[bucket];
                cb.record_count++;

                size_t need = 4 + 1 + (wide_ingest ? 8 : 4) + 5 + f[0].len +
                              (valid ? (8 + 6 + 5 + f[1].len) : 0);
                uint8_t* dst = rec_tmp;
                if (__builtin_expect(need > sizeof(rec_tmp), 0)) {
                    big_rec.resize(need);
                    dst = big_rec.data();
                }

                uint8_t* w = dst;
                std::memcpy(w, &eid_h32, 4); w += 4;
                // bit0 = survived/valid (mutable during dedup), bit1 = wide_ingest, bit2 = has_payload
                *w++ = static_cast<uint8_t>((valid ? 5 : 0) | (wide_ingest ? 2 : 0));
                if (__builtin_expect(!wide_ingest, 1)) {
                    int32_t id32 = static_cast<int32_t>(ingest_diff);
                    std::memcpy(w, &id32, 4); w += 4;
                } else {
                    std::memcpy(w, &ingest_sec, 8); w += 8;
                }
                w += encode_len(w, f[0].len);
                std::memcpy(w, f[0].ptr, f[0].len); w += f[0].len;

                if (valid) {
                    if (__builtin_expect(compact_ts, 1)) {
                        uint32_t tsoff = static_cast<uint32_t>(ts_ms - window_start);
                        std::memcpy(w, &tsoff, 4); w += 4;
                    } else {
                        std::memcpy(w, &ts_ms, 8); w += 8;
                    }
                    if (__builtin_expect(compact_cat, 1)) {
                        uint32_t pc = (cat_id << 24) | price_cents;
                        std::memcpy(w, &pc, 4); w += 4;
                    } else {
                        uint16_t cat16 = static_cast<uint16_t>(cat_id);
                        std::memcpy(w, &price_cents, 4); w += 4;
                        std::memcpy(w, &cat16, 2); w += 2;
                    }
                    w += encode_len(w, f[1].len);
                    std::memcpy(w, f[1].ptr, f[1].len); w += f[1].len;
                }
                cb.append_record(dst, static_cast<size_t>(w - dst));
            });
        }
        for (size_t b = 0; b < num_buckets; ++b) my_buckets[b].finish();
    };

    if (num_threads == 1) {
        build_identity_from_links(link_bufs, identity_map);
        link_bufs.clear();
        link_bufs.shrink_to_fit();
        scan_events_worker(0);
    } else {
        std::vector<std::thread> pool;
        pool.reserve(num_threads - 1);
        for (size_t t = 1; t < num_threads; ++t) pool.emplace_back(scan_events_worker, t);
        build_identity_from_links(link_bufs, identity_map);
        link_bufs.clear();
        link_bufs.shrink_to_fit();
        scan_events_worker(0);
        for (auto& th : pool) th.join();
    }
    for (auto& f : event_files) f.close();
    std::fprintf(stderr, "Stage 1 (links + events scan) in %.3fs\n", elapsed_sec(t_s1));

    // Stage 2: Parallel clicks/*.csv scan + zero-RecInfo L3-resident event dedup!
    auto t_s2 = Clock::now();
    std::vector<std::vector<ChunkedBuf>> person_ev(num_threads);
    std::vector<std::vector<ChunkedBuf>> person_cl(num_threads);
    for (size_t t = 0; t < num_threads; ++t) {
        person_ev[t].resize(num_buckets);
        person_cl[t].resize(num_buckets);
        for (size_t b = 0; b < num_buckets; ++b) {
            person_ev[t][b].init(slab_cap);
            person_cl[t][b].init(slab_cap / 2 >= 2048 ? slab_cap / 2 : 2048);
        }
    }

    auto click_chunks = build_read_chunks(click_files, READ_CHUNK_SIZE);
    std::atomic<size_t> next_cc{0};
    std::atomic<size_t> next_db{0};

    auto stage2_worker = [&](size_t tid) {
        std::vector<char> io_buf;
        io_buf.reserve(READ_CHUNK_SIZE + 8192);
        std::vector<uint8_t> big_tmp;
        uint8_t tmp[256];
        auto& my_cl = person_cl[tid];

        // 2a: Clicks scan
        while (true) {
            size_t ci = next_cc.fetch_add(1, std::memory_order_relaxed);
            if (ci >= click_chunks.size()) break;
            const auto& ch = click_chunks[ci];
            auto sv = read_aligned_chunk(click_files[ch.file_idx], ch, io_buf);
            if (sv.empty()) continue;

            fast::scan_csv_chunk<4>(sv.data(), sv.size(), [&](const fast::Field (&f)[4]) {
                if (__builtin_expect(f[0].len == 0 || f[1].len == 0 || f[2].len == 0, 0)) return;
                uint32_t camp_id = 0;
                if (__builtin_expect(!catalogs.campaign_to_id.find(std::string_view(f[2].ptr, f[2].len), camp_id), 0)) {
                    return;
                }
                int64_t ts_ms = 0;
                if (__builtin_expect(!fast::parse_timestamp_ms(f[3].ptr, f[3].len, dtab, ts_ms), 0)) {
                    return;
                }
                if (__builtin_expect(ts_ms < click_start || ts_ms >= window_end, 0)) return;

                uint64_t ph = 0;
                std::string_view pid = identity_map.resolve(std::string_view(f[1].ptr, f[1].len), ph);
                size_t p = static_cast<size_t>(ph >> bucket_shift);

                ChunkedBuf& cb = my_cl[p];
                cb.record_count++;
                uint32_t ph32 = static_cast<uint32_t>(ph);
                size_t need = 16 + 10 + pid.size() + f[0].len;
                uint8_t* dst = tmp;
                if (__builtin_expect(need > sizeof(tmp), 0)) {
                    big_tmp.resize(need);
                    dst = big_tmp.data();
                }
                uint8_t* w = dst;
                std::memcpy(w, &ph32, 4); w += 4;
                std::memcpy(w, &ts_ms, 8); w += 8;
                std::memcpy(w, &camp_id, 4); w += 4;
                w += encode_len(w, pid.size());
                w += encode_len(w, f[0].len);
                std::memcpy(w, pid.data(), pid.size()); w += pid.size();
                std::memcpy(w, f[0].ptr, f[0].len); w += f[0].len;
                cb.append_record(dst, static_cast<size_t>(w - dst));
            });
        }
        for (size_t b = 0; b < num_buckets; ++b) my_cl[b].finish();
        io_buf.clear();
        io_buf.shrink_to_fit();

        // 2b: Sequential-emission event dedup with mutable flags + slab recycling!
        // During dedup, losing copies have bit 0 of flags cleared in-place.
        // Afterwards we scan slabs sequentially (hardware prefetcher!) and emit surviving flags&1 records.
        SlabPool local_pool;
        local_pool.slab_cap = slab_cap;

        struct DedupSlot {
            uint32_t h32;
            uint32_t loc; // 0 = empty; else ((sidx + 1) << 18) | off (or bit31 overflow)
        };
        std::vector<DedupSlot> table;
        std::vector<uint8_t*> slab_bases;
        std::vector<uint8_t*> overflow_ptrs;
        auto& my_ev = person_ev[tid];

        auto get_rec_ptr = [&](uint32_t loc) -> uint8_t* {
            if (__builtin_expect((loc & 0x80000000U) == 0, 1)) {
                uint32_t sidx = (loc >> 18) - 1;
                uint32_t off = loc & 0x3FFFFU;
                return slab_bases[sidx] + off;
            }
            return overflow_ptrs[loc & 0x7FFFFFFFU];
        };

        auto decode_rec_header = [&](uint8_t* rec, int64_t& ingest_sec,
                                     const uint8_t*& eid_ptr, size_t& eid_len) {
            const uint8_t* q = rec + 4; // skip h32
            uint8_t flags = *q++;
            if (__builtin_expect((flags & 2) == 0, 1)) {
                int32_t id32;
                std::memcpy(&id32, q, 4); q += 4;
                ingest_sec = base_sec + id32;
            } else {
                std::memcpy(&ingest_sec, q, 8); q += 8;
            }
            eid_len = decode_len(q);
            eid_ptr = q;
        };

        while (true) {
            size_t b = next_db.fetch_add(1, std::memory_order_relaxed);
            if (b >= num_buckets) break;

            size_t total_recs = 0;
            for (size_t t = 0; t < num_threads; ++t) total_recs += ev_buckets[t][b].record_count;
            if (total_recs == 0) {
                for (size_t t = 0; t < num_threads; ++t) ev_buckets[t][b].release(&local_pool);
                continue;
            }

            size_t cap = next_pow2(std::max<size_t>(32, total_recs * 3 / 2));
            if (table.size() < cap) table.resize(cap);
            std::memset(table.data(), 0, cap * sizeof(DedupSlot));
            size_t mask = cap - 1;
            slab_bases.clear();
            overflow_ptrs.clear();

            // Pass 1: Dedup in-place, clearing valid bit on every losing duplicate!
            for (size_t t = 0; t < num_threads; ++t) {
                auto& cb = ev_buckets[t][b];
                for (size_t si = 0; si < cb.slabs.size(); ++si) {
                    uint8_t* base = cb.slabs[si];
                    uint8_t* p = base;
                    uint8_t* end = base + cb.slab_sizes[si];
                    uint32_t sidx = static_cast<uint32_t>(slab_bases.size());
                    slab_bases.push_back(base);

                    while (p < end) {
                        uint8_t* rec_start = p;
                        uint32_t h32;
                        std::memcpy(&h32, p, 4); p += 4;
                        uint8_t* flags_ptr = p++;
                        uint8_t flags = *flags_ptr;
                        uint8_t valid = flags & 1;
                        int64_t ingest_sec;
                        if (__builtin_expect((flags & 2) == 0, 1)) {
                            int32_t id32;
                            std::memcpy(&id32, p, 4); p += 4;
                            ingest_sec = base_sec + id32;
                        } else {
                            std::memcpy(&ingest_sec, p, 8); p += 8;
                        }
                        const uint8_t* cp = p;
                        size_t eid_len = decode_len(cp);
                        const uint8_t* eid_ptr = cp;
                        cp += eid_len;
                        if (valid) {
                            cp += (compact_ts ? 4 : 8) + (compact_cat ? 4 : 6);
                            size_t uid_len = decode_len(cp);
                            cp += uid_len;
                        }
                        p = const_cast<uint8_t*>(cp);

                        size_t off = static_cast<size_t>(rec_start - base);
                        uint32_t loc;
                        if (__builtin_expect(off < (1U << 18) && sidx < 8190, 1)) {
                            loc = ((sidx + 1) << 18) | static_cast<uint32_t>(off);
                        } else {
                            loc = 0x80000000U | static_cast<uint32_t>(overflow_ptrs.size());
                            overflow_ptrs.push_back(rec_start);
                        }

                        size_t slot = h32 & mask;
                        while (true) {
                            DedupSlot& ds = table[slot];
                            if (ds.loc == 0) {
                                ds.h32 = h32;
                                ds.loc = loc;
                                break;
                            }
                            if (__builtin_expect(ds.h32 == h32, 0)) {
                                uint8_t* prev_rec = get_rec_ptr(ds.loc);
                                int64_t prev_ingest;
                                const uint8_t* prev_eid;
                                size_t prev_len;
                                decode_rec_header(prev_rec, prev_ingest, prev_eid, prev_len);
                                if (prev_len == eid_len && std::memcmp(prev_eid, eid_ptr, eid_len) == 0) {
                                    if (ingest_sec < prev_ingest) {
                                        // New copy wins; mark old winner dead!
                                        prev_rec[4] &= static_cast<uint8_t>(~1U);
                                        ds.loc = loc;
                                    } else {
                                        // Existing copy wins; mark new copy dead!
                                        *flags_ptr &= static_cast<uint8_t>(~1U);
                                    }
                                    break;
                                }
                            }
                            slot = (slot + 1) & mask;
                        }
                    }
                }
            }

            // Pass 2: 100% SEQUENTIAL scan over slabs -> emit surviving valid events!
            size_t payload_len = (compact_ts ? 4 : 8) + (compact_cat ? 4 : 6);
            for (size_t t = 0; t < num_threads; ++t) {
                auto& cb = ev_buckets[t][b];
                for (size_t si = 0; si < cb.slabs.size(); ++si) {
                    const uint8_t* q = cb.slabs[si];
                    const uint8_t* end = q + cb.slab_sizes[si];
                    while (q < end) {
                        uint8_t flags = q[4];
                        // Note: original layout had valid payload iff bit 2 (has_payload) was set!
                        // Wait: if a valid record lost dedup, we cleared bit 0, so we MUST still skip its payload!
                        bool had_payload = (flags & 4) != 0;
                        bool survived = (flags & 1) != 0;
                        q += 5 + (((flags & 2) == 0) ? 4 : 8);
                        size_t eid_len = decode_len(q);
                        q += eid_len;
                        if (!had_payload) continue;

                        const uint8_t* payload_ptr = q;
                        q += payload_len;
                        size_t uid_len = decode_len(q);
                        std::string_view raw_uid(reinterpret_cast<const char*>(q), uid_len);
                        q += uid_len;
                        if (!survived) continue;

                        uint64_t ph = 0;
                        std::string_view pid = identity_map.resolve(raw_uid, ph);
                        size_t p_part = static_cast<size_t>(ph >> bucket_shift);

                        ChunkedBuf& out_cb = my_ev[p_part];
                        out_cb.record_count++;
                        uint32_t ph32 = static_cast<uint32_t>(ph);
                        size_t need = 4 + payload_len + 5 + pid.size();
                        uint8_t* dst = tmp;
                        if (__builtin_expect(need > sizeof(tmp), 0)) {
                            big_tmp.resize(need);
                            dst = big_tmp.data();
                        }
                        uint8_t* w = dst;
                        std::memcpy(w, &ph32, 4); w += 4;
                        std::memcpy(w, payload_ptr, payload_len); w += payload_len;
                        w += encode_len(w, pid.size());
                        std::memcpy(w, pid.data(), pid.size()); w += pid.size();
                        out_cb.append_record(dst, static_cast<size_t>(w - dst), &local_pool);
                    }
                }
                cb.release(&local_pool);
            }
        }
        for (size_t b = 0; b < num_buckets; ++b) my_ev[b].finish(&local_pool);
    };

    if (num_threads == 1) {
        stage2_worker(0);
    } else {
        std::vector<std::thread> pool;
        pool.reserve(num_threads - 1);
        for (size_t t = 1; t < num_threads; ++t) pool.emplace_back(stage2_worker, t);
        stage2_worker(0);
        for (auto& th : pool) th.join();
    }
    for (auto& f : click_files) f.close();
    ev_buckets.clear();
    ev_buckets.shrink_to_fit();
    std::fprintf(stderr, "Stage 2 (clicks + event dedup) in %.3fs\n", elapsed_sec(t_s2));

    // Stage 3: Single-pass CSR grouping + sessionize + attribute + parallel Parquet write!
    auto t_s3 = Clock::now();
    std::vector<std::pair<size_t, size_t>> part_order(num_buckets);
    for (size_t p = 0; p < num_buckets; ++p) {
        size_t cnt = 0;
        for (size_t t = 0; t < num_threads; ++t) cnt += person_ev[t][p].record_count;
        part_order[p] = {cnt, p};
    }
    std::sort(part_order.begin(), part_order.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });

    uint32_t num_camps = static_cast<uint32_t>(catalogs.campaign_names.size());
    uint32_t num_cats = static_cast<uint32_t>(catalogs.categories.size());
    std::vector<CampDayAccum> camp_accums(num_threads);
    for (size_t t = 0; t < num_threads; ++t) camp_accums[t].init(num_camps, min_day, max_day);

    std::atomic<size_t> next_part{0};

    auto stage3_worker = [&](size_t tid) {
        UserDailyWriter ud_writer;
        ud_writer.open(out_dir + "/user_daily/part-" + std::to_string(tid) + ".parquet");

        CampDayAccum& cacc = camp_accums[tid];

        struct UserSlot {
            uint32_t h32;
            uint32_t uid_plus1;
        };
        struct EventRec {
            int64_t ts_ms;
            uint32_t price_cents;
            uint16_t cat_id;
            uint16_t pad;
        };
        struct ClickRec {
            int64_t ts_ms;
            uint32_t camp_id;
            uint32_t cid_len;
            const char* cid_ptr;
        };
        struct SessionRec {
            int64_t start_ms;
            int32_t day;
            int64_t rev_cents;
        };

        std::vector<UserSlot> utable;
        std::vector<std::string_view> u_strs;
        std::vector<uint32_t> u_ev_cnt;
        std::vector<uint32_t> u_cl_cnt;
        std::vector<uint32_t> ev_off, cl_off, ev_cur, cl_cur;
        std::vector<uint64_t> tmp_ev64, ev_arr64; // fast path: compact_ts && compact_cat (8B!)
        std::vector<EventRec> tmp_ev, ev_arr;     // general fallback (16B)
        std::vector<uint32_t> tmp_ev_uid;
        std::vector<ClickRec> tmp_cl, cl_arr;
        std::vector<uint32_t> tmp_cl_uid;
        const bool fast_ev64 = compact_ts && compact_cat;

        std::vector<SessionRec> u_sessions;
        std::vector<int64_t> cat_rev(num_cats, 0);
        std::vector<uint32_t> touched_cats;
        touched_cats.reserve(num_cats);

        std::vector<uint32_t> camp_gen(num_camps, 0);
        std::vector<size_t> camp_next_sess(num_camps, 0);
        uint32_t cur_gen = 0;

        while (true) {
            size_t oi = next_part.fetch_add(1, std::memory_order_relaxed);
            if (oi >= num_buckets) break;
            size_t p = part_order[oi].second;
            size_t total_ev = part_order[oi].first;
            if (total_ev == 0) {
                for (size_t t = 0; t < num_threads; ++t) {
                    person_ev[t][p].release();
                    person_cl[t][p].release();
                }
                continue;
            }

            size_t est_users = std::max<size_t>(32, total_ev / 10);
            size_t ucap = next_pow2(est_users * 2);
            utable.assign(ucap, UserSlot{0, 0});
            size_t umask = ucap - 1;
            u_strs.clear();
            u_ev_cnt.clear();

            auto grow_utable = [&]() {
                size_t ncap = utable.size() * 2;
                std::vector<UserSlot> nt(ncap, UserSlot{0, 0});
                size_t nmask = ncap - 1;
                for (const auto& s : utable) {
                    if (s.uid_plus1 == 0) continue;
                    size_t idx = s.h32 & nmask;
                    while (nt[idx].uid_plus1 != 0) idx = (idx + 1) & nmask;
                    nt[idx] = s;
                }
                utable.swap(nt);
                umask = nmask;
            };

            auto get_or_add_user = [&](uint32_t h32, std::string_view pid) -> uint32_t {
                if (__builtin_expect((u_strs.size() + 1) * 10 >= utable.size() * 6, 0)) grow_utable();
                size_t idx = h32 & umask;
                while (utable[idx].uid_plus1 != 0) {
                    if (utable[idx].h32 == h32) {
                        uint32_t uid = utable[idx].uid_plus1 - 1;
                        if (u_strs[uid].size() == pid.size() &&
                            std::memcmp(u_strs[uid].data(), pid.data(), pid.size()) == 0) {
                            return uid;
                        }
                    }
                    idx = (idx + 1) & umask;
                }
                uint32_t uid = static_cast<uint32_t>(u_strs.size());
                u_strs.push_back(pid);
                u_ev_cnt.push_back(0);
                utable[idx] = UserSlot{h32, uid + 1};
                return uid;
            };

            auto find_user = [&](uint32_t h32, std::string_view pid) -> int64_t {
                size_t idx = h32 & umask;
                while (utable[idx].uid_plus1 != 0) {
                    if (utable[idx].h32 == h32) {
                        uint32_t uid = utable[idx].uid_plus1 - 1;
                        if (u_strs[uid].size() == pid.size() &&
                            std::memcmp(u_strs[uid].data(), pid.data(), pid.size()) == 0) {
                            return static_cast<int64_t>(uid);
                        }
                    }
                    idx = (idx + 1) & umask;
                }
                return -1;
            };

            // SINGLE PASS over person_ev[t][p]: resolve uid ONCE and store (uid, packed Event)!
            tmp_ev_uid.clear();
            tmp_ev_uid.reserve(total_ev);
            if (__builtin_expect(fast_ev64, 1)) {
                tmp_ev64.clear();
                tmp_ev64.reserve(total_ev);
                for (size_t t = 0; t < num_threads; ++t) {
                    const auto& cb = person_ev[t][p];
                    for (size_t si = 0; si < cb.slabs.size(); ++si) {
                        const uint8_t* q = cb.slabs[si];
                        const uint8_t* end = q + cb.slab_sizes[si];
                        while (q < end) {
                            uint32_t h32, tsoff, pc;
                            std::memcpy(&h32, q, 4);
                            std::memcpy(&tsoff, q + 4, 4);
                            std::memcpy(&pc, q + 8, 4);
                            q += 12;
                            size_t plen = decode_len(q);
                            std::string_view pid(reinterpret_cast<const char*>(q), plen);
                            q += plen;
                            uint32_t uid = get_or_add_user(h32, pid);
                            u_ev_cnt[uid]++;
                            tmp_ev_uid.push_back(uid);
                            tmp_ev64.push_back((static_cast<uint64_t>(tsoff) << 32) | pc);
                        }
                    }
                }
            } else {
                tmp_ev.clear();
                tmp_ev.reserve(total_ev);
                for (size_t t = 0; t < num_threads; ++t) {
                    const auto& cb = person_ev[t][p];
                    for (size_t si = 0; si < cb.slabs.size(); ++si) {
                        const uint8_t* q = cb.slabs[si];
                        const uint8_t* end = q + cb.slab_sizes[si];
                        while (q < end) {
                            uint32_t h32;
                            std::memcpy(&h32, q, 4); q += 4;
                            int64_t ts_ms;
                            if (compact_ts) {
                                uint32_t tsoff;
                                std::memcpy(&tsoff, q, 4); q += 4;
                                ts_ms = window_start + static_cast<int64_t>(tsoff);
                            } else {
                                std::memcpy(&ts_ms, q, 8); q += 8;
                            }
                            uint32_t price;
                            uint16_t cat16;
                            if (compact_cat) {
                                uint32_t pc;
                                std::memcpy(&pc, q, 4); q += 4;
                                price = pc & 0x00FFFFFFU;
                                cat16 = static_cast<uint16_t>(pc >> 24);
                            } else {
                                std::memcpy(&price, q, 4); q += 4;
                                std::memcpy(&cat16, q, 2); q += 2;
                            }
                            size_t plen = decode_len(q);
                            std::string_view pid(reinterpret_cast<const char*>(q), plen);
                            q += plen;
                            uint32_t uid = get_or_add_user(h32, pid);
                            u_ev_cnt[uid]++;
                            tmp_ev_uid.push_back(uid);
                            tmp_ev.push_back(EventRec{ts_ms, price, cat16, 0});
                        }
                    }
                }
            }

            size_t num_users = u_strs.size();
            ev_off.resize(num_users + 1);
            ev_off[0] = 0;
            for (size_t u = 0; u < num_users; ++u) ev_off[u + 1] = ev_off[u] + u_ev_cnt[u];
            ev_cur.assign(ev_off.begin(), ev_off.begin() + num_users);
            if (__builtin_expect(fast_ev64, 1)) {
                ev_arr64.resize(tmp_ev64.size());
                for (size_t i = 0; i < tmp_ev64.size(); ++i) {
                    ev_arr64[ev_cur[tmp_ev_uid[i]]++] = tmp_ev64[i];
                }
            } else {
                ev_arr.resize(tmp_ev.size());
                for (size_t i = 0; i < tmp_ev.size(); ++i) {
                    ev_arr[ev_cur[tmp_ev_uid[i]]++] = tmp_ev[i];
                }
            }

            // SINGLE PASS over person_cl[t][p]: resolve uid ONCE and store (uid, ClickRec)!
            u_cl_cnt.assign(num_users, 0);
            tmp_cl.clear();
            tmp_cl_uid.clear();
            for (size_t t = 0; t < num_threads; ++t) {
                const auto& cb = person_cl[t][p];
                for (size_t si = 0; si < cb.slabs.size(); ++si) {
                    const uint8_t* q = cb.slabs[si];
                    const uint8_t* end = q + cb.slab_sizes[si];
                    while (q < end) {
                        uint32_t h32;
                        int64_t ts_ms;
                        uint32_t camp_id;
                        std::memcpy(&h32, q, 4); q += 4;
                        std::memcpy(&ts_ms, q, 8); q += 8;
                        std::memcpy(&camp_id, q, 4); q += 4;
                        size_t plen = decode_len(q);
                        size_t clen = decode_len(q);
                        std::string_view pid(reinterpret_cast<const char*>(q), plen);
                        q += plen;
                        const char* cid_ptr = reinterpret_cast<const char*>(q);
                        q += clen;
                        int64_t uid = find_user(h32, pid);
                        if (uid >= 0) {
                            u_cl_cnt[static_cast<size_t>(uid)]++;
                            tmp_cl_uid.push_back(static_cast<uint32_t>(uid));
                            tmp_cl.push_back(ClickRec{ts_ms, camp_id, static_cast<uint32_t>(clen), cid_ptr});
                        }
                    }
                }
            }

            cl_off.resize(num_users + 1);
            cl_off[0] = 0;
            for (size_t u = 0; u < num_users; ++u) cl_off[u + 1] = cl_off[u] + u_cl_cnt[u];
            cl_cur.assign(cl_off.begin(), cl_off.begin() + num_users);
            cl_arr.resize(tmp_cl.size());
            for (size_t i = 0; i < tmp_cl.size(); ++i) {
                cl_arr[cl_cur[tmp_cl_uid[i]]++] = tmp_cl[i];
            }

            // Process every person u in partition p!
            for (size_t u = 0; u < num_users; ++u) {
                u_sessions.clear();
                touched_cats.clear();

                auto flush_day = [&](int32_t cur_day, int64_t day_sessions, int64_t day_events, int64_t day_rev) {
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
                    ud_writer.add_row(u_strs[u], cur_day, day_sessions, day_events, day_rev, top_cat);
                };

                if (__builtin_expect(fast_ev64, 1)) {
                    uint64_t* eb = ev_arr64.data() + ev_off[u];
                    uint64_t* ee = ev_arr64.data() + ev_off[u + 1];
                    size_t en = static_cast<size_t>(ee - eb);
                    if (en > 1) std::sort(eb, ee);

                    int64_t sess_start = window_start + static_cast<int64_t>(eb[0] >> 32);
                    int64_t prev_ts = sess_start;
                    int32_t sess_day = fast::floor_div_day(sess_start);
                    int64_t sess_rev = 0;
                    int32_t cur_day = sess_day;
                    int64_t day_sessions = 1;
                    int64_t day_events = 0;
                    int64_t day_rev = 0;

                    for (size_t i = 0; i < en; ++i) {
                        uint64_t ev = eb[i];
                        int64_t ts = window_start + static_cast<int64_t>(ev >> 32);
                        if (i > 0 && ts - prev_ts > SESSION_GAP_MS) {
                            u_sessions.push_back(SessionRec{sess_start, sess_day, sess_rev});
                            sess_start = ts;
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
                        prev_ts = ts;
                        day_events++;
                        uint32_t price = static_cast<uint32_t>(ev) & 0x00FFFFFFU;
                        if (__builtin_expect(price > 0, 0)) {
                            uint32_t cid = (static_cast<uint32_t>(ev) >> 24);
                            sess_rev += price;
                            day_rev += price;
                            if (cat_rev[cid] == 0) touched_cats.push_back(cid);
                            cat_rev[cid] += price;
                        }
                    }
                    u_sessions.push_back(SessionRec{sess_start, sess_day, sess_rev});
                    flush_day(cur_day, day_sessions, day_events, day_rev);
                } else {
                    EventRec* eb = ev_arr.data() + ev_off[u];
                    EventRec* ee = ev_arr.data() + ev_off[u + 1];
                    size_t en = static_cast<size_t>(ee - eb);
                    if (en > 1) {
                        std::sort(eb, ee, [](const EventRec& a, const EventRec& b) {
                            return a.ts_ms < b.ts_ms;
                        });
                    }

                    int64_t sess_start = eb[0].ts_ms;
                    int64_t prev_ts = eb[0].ts_ms;
                    int32_t sess_day = fast::floor_div_day(sess_start);
                    int64_t sess_rev = 0;
                    int32_t cur_day = sess_day;
                    int64_t day_sessions = 1;
                    int64_t day_events = 0;
                    int64_t day_rev = 0;

                    for (size_t i = 0; i < en; ++i) {
                        const EventRec& e = eb[i];
                        if (i > 0 && e.ts_ms - prev_ts > SESSION_GAP_MS) {
                            u_sessions.push_back(SessionRec{sess_start, sess_day, sess_rev});
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
                    u_sessions.push_back(SessionRec{sess_start, sess_day, sess_rev});
                    flush_day(cur_day, day_sessions, day_events, day_rev);
                }

                size_t S = u_sessions.size();
                ClickRec* cb = cl_arr.data() + cl_off[u];
                ClickRec* ce = cl_arr.data() + cl_off[u + 1];
                size_t C = static_cast<size_t>(ce - cb);

                if (C == 0) {
                    for (size_t si = 0; si < S; ++si) {
                        auto& st = cacc.get(0, u_sessions[si].day);
                        st.sessions++;
                        if (u_sessions[si].rev_cents > 0) {
                            st.purchasing_sessions++;
                            st.revenue_cents += u_sessions[si].rev_cents;
                        }
                    }
                    continue;
                }

                if (C > 1) {
                    std::sort(cb, ce, [](const ClickRec& a, const ClickRec& b) {
                        if (a.ts_ms != b.ts_ms) return a.ts_ms < b.ts_ms;
                        return fast::cmp_bytes(a.cid_ptr, a.cid_len, b.cid_ptr, b.cid_len) < 0;
                    });
                }

                size_t ci = 0;
                int64_t best_ts = std::numeric_limits<int64_t>::min();
                uint32_t best_camp = 0;
                for (size_t si = 0; si < S; ++si) {
                    int64_t sstart = u_sessions[si].start_ms;
                    while (ci < C && cb[ci].ts_ms <= sstart) {
                        if (cb[ci].ts_ms > best_ts) {
                            best_ts = cb[ci].ts_ms;
                            best_camp = cb[ci].camp_id;
                        }
                        ci++;
                    }
                    uint32_t credited = (best_ts != std::numeric_limits<int64_t>::min() &&
                                         sstart - best_ts <= WEEK_MS)
                                            ? best_camp
                                            : 0;
                    auto& st = cacc.get(credited, u_sessions[si].day);
                    st.sessions++;
                    if (u_sessions[si].rev_cents > 0) {
                        st.purchasing_sessions++;
                        st.revenue_cents += u_sessions[si].rev_cents;
                    }
                }

                if (++cur_gen == 0) {
                    std::fill(camp_gen.begin(), camp_gen.end(), 0);
                    cur_gen = 1;
                }
                size_t si_lo = 0;
                for (size_t i = 0; i < C; ++i) {
                    int64_t cts = cb[i].ts_ms;
                    while (si_lo < S && u_sessions[si_lo].start_ms < cts) si_lo++;
                    if (si_lo >= S) break;

                    uint32_t camp = cb[i].camp_id;
                    size_t k = (camp_gen[camp] == cur_gen) ? std::max(si_lo, camp_next_sess[camp]) : si_lo;
                    int64_t hi = cts + WEEK_MS;
                    while (k < S && u_sessions[k].start_ms <= hi) {
                        cacc.get(camp, u_sessions[k].day).assisted_sessions++;
                        k++;
                    }
                    camp_gen[camp] = cur_gen;
                    camp_next_sess[camp] = k;
                }
            }

            for (size_t t = 0; t < num_threads; ++t) {
                person_ev[t][p].release();
                person_cl[t][p].release();
            }
        }

        ud_writer.close();
    };

    if (num_threads == 1) {
        stage3_worker(0);
    } else {
        std::vector<std::thread> pool;
        pool.reserve(num_threads - 1);
        for (size_t t = 1; t < num_threads; ++t) pool.emplace_back(stage3_worker, t);
        stage3_worker(0);
        for (auto& th : pool) th.join();
    }

    for (size_t t = 1; t < num_threads; ++t) {
        camp_accums[0].merge_from(camp_accums[t]);
    }
    if (!write_campaign_daily(out_dir + "/campaign_daily/part-0.parquet", catalogs, camp_accums[0])) {
        std::fprintf(stderr, "Failed to write campaign_daily\n");
        return 1;
    }

    std::fprintf(stderr, "Stage 3 (sessionize + attribute + parquet) in %.3fs\n", elapsed_sec(t_s3));
    std::fprintf(stderr, "Completed pipeline in %.3fs\n", elapsed_sec(t_total));
    return 0;
}
