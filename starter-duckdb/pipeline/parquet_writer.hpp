#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <string>
#include <string_view>
#include <sys/uio.h>
#include <unistd.h>
#include <vector>

namespace pq {

// Ultra-compact, zero-dependency Apache Parquet v1 writer:
// Supports PLAIN, RLE_DICTIONARY, DELTA_BINARY_PACKED, and DELTA_BYTE_ARRAY!
// Shrinks user_daily output ~5x (cutting dirty-page writeback at container exit!).

enum PhysicalType : int32_t {
    PT_BOOLEAN = 0,
    PT_INT32 = 1,
    PT_INT64 = 2,
    PT_BYTE_ARRAY = 6,
};

enum RepetitionType : int32_t {
    REP_REQUIRED = 0,
    REP_OPTIONAL = 1,
};

enum ConvertedType : int32_t {
    CT_NONE = -1,
    CT_UTF8 = 0,
    CT_DATE = 6,
};

enum Encoding : int32_t {
    ENC_PLAIN = 0,
    ENC_RLE = 3,
    ENC_DELTA_BINARY_PACKED = 5,
    ENC_DELTA_BYTE_ARRAY = 7,
    ENC_RLE_DICTIONARY = 8,
};

enum PageType : int32_t {
    PAGE_DATA = 0,
    PAGE_DICTIONARY = 2,
};

static inline void append_uleb128(std::vector<uint8_t>& out, uint64_t v) {
    while (v >= 0x80ULL) {
        out.push_back(static_cast<uint8_t>((v & 0x7FULL) | 0x80ULL));
        v >>= 7;
    }
    out.push_back(static_cast<uint8_t>(v));
}

static inline uint64_t zigzag64(int64_t v) {
    return (static_cast<uint64_t>(v) << 1) ^ static_cast<uint64_t>(v >> 63);
}

static inline uint32_t zigzag32(int32_t v) {
    return (static_cast<uint32_t>(v) << 1) ^ static_cast<uint32_t>(v >> 31);
}

// Parquet RLE/bit-packing hybrid encoder for uint32_t indices at bit_width w (1..32).
static inline void encode_rle_hybrid(const uint32_t* vals, size_t n, uint32_t w, std::vector<uint8_t>& out) {
    const size_t nbytes = (w + 7U) >> 3;
    auto emit_packed = [&](size_t a, size_t z) {
        size_t groups = (z - a + 7) >> 3;
        append_uleb128(out, (static_cast<uint64_t>(groups) << 1) | 1ULL);
        const size_t out_bytes = groups * static_cast<size_t>(w);
        const size_t old_sz = out.size();
        if (__builtin_expect(w <= 8, 1)) {
            out.resize(old_sz + out_bytes + 8);
            uint8_t* dst = out.data() + old_sz;
            size_t pos = a;
            for (size_t g = 0; g < groups; ++g, pos += 8) {
                uint64_t acc = 0;
                if (__builtin_expect(pos + 8 <= z, 1)) {
                    const uint32_t* p = vals + pos;
                    acc = static_cast<uint64_t>(p[0]) |
                          (static_cast<uint64_t>(p[1]) << w) |
                          (static_cast<uint64_t>(p[2]) << (2U * w)) |
                          (static_cast<uint64_t>(p[3]) << (3U * w)) |
                          (static_cast<uint64_t>(p[4]) << (4U * w)) |
                          (static_cast<uint64_t>(p[5]) << (5U * w)) |
                          (static_cast<uint64_t>(p[6]) << (6U * w)) |
                          (static_cast<uint64_t>(p[7]) << (7U * w));
                } else {
                    for (size_t k = 0; k < 8 && pos + k < z; ++k) {
                        acc |= (static_cast<uint64_t>(vals[pos + k]) << (k * w));
                    }
                }
                std::memcpy(dst, &acc, 8);
                dst += w;
            }
            out.resize(old_sz + out_bytes);
        } else {
            out.resize(old_sz + out_bytes);
            uint8_t* dst = out.data() + old_sz;
            uint64_t acc = 0;
            uint32_t nb = 0;
            const size_t total = groups << 3;
            for (size_t k = 0; k < total; ++k) {
                uint64_t v = (a + k < z) ? vals[a + k] : 0ULL;
                acc |= (v << nb);
                nb += w;
                while (nb >= 8) {
                    *dst++ = static_cast<uint8_t>(acc);
                    acc >>= 8;
                    nb -= 8;
                }
            }
        }
    };

    size_t i = 0;
    size_t pending = SIZE_MAX;
    while (i < n) {
        size_t r = 1;
        uint32_t v0 = vals[i];
        while (i + r < n && vals[i + r] == v0) ++r;
        if (r >= 8) {
            if (pending != SIZE_MAX) {
                size_t pad = (8 - ((i - pending) & 7)) & 7;
                emit_packed(pending, i + pad);
                pending = SIZE_MAX;
                i += pad;
                r -= pad;
            }
            if (r > 0) {
                append_uleb128(out, static_cast<uint64_t>(r) << 1);
                for (size_t b = 0; b < nbytes; ++b) {
                    out.push_back(static_cast<uint8_t>(v0 >> (8 * b)));
                }
                i += r;
            }
        } else {
            if (pending == SIZE_MAX) pending = i;
            i += r;
        }
    }
    if (pending != SIZE_MAX) emit_packed(pending, n);
}

// Parquet DELTA_BINARY_PACKED encoder (blocks of 128, 4 miniblocks of 32).
template <typename IntT>
static inline void encode_delta_binary_packed(const IntT* vals, size_t n, std::vector<uint8_t>& out) {
    out.reserve(out.size() + n + 64);
    append_uleb128(out, 128);
    append_uleb128(out, 4);
    append_uleb128(out, n);
    append_uleb128(out, zigzag64(n > 0 ? static_cast<int64_t>(vals[0]) : 0LL));
    size_t i = 1;
    int64_t raw[128];
    uint64_t deltas[128];
    while (i < n) {
        size_t cnt = std::min<size_t>(128, n - i);
        int64_t min_d = INT64_MAX;
        for (size_t k = 0; k < cnt; ++k) {
            int64_t d = static_cast<int64_t>(static_cast<uint64_t>(vals[i + k]) -
                                             static_cast<uint64_t>(vals[i + k - 1]));
            raw[k] = d;
            if (d < min_d) min_d = d;
        }
        append_uleb128(out, zigzag64(min_d));
        for (size_t k = 0; k < cnt; ++k) {
            deltas[k] = static_cast<uint64_t>(raw[k]) - static_cast<uint64_t>(min_d);
        }
        for (size_t k = cnt; k < 128; ++k) deltas[k] = 0;
        uint8_t widths[4];
        size_t total_mb_bytes = 0;
        size_t used = (cnt + 31) >> 5;
        for (int m = 0; m < 4; ++m) {
            uint64_t mx = 0;
            for (int k = 0; k < 32; ++k) mx |= deltas[m * 32 + k];
            widths[m] = mx ? static_cast<uint8_t>(64 - __builtin_clzll(mx)) : 0;
            if (static_cast<size_t>(m) < used) total_mb_bytes += 4U * widths[m];
        }
        size_t old_sz = out.size();
        out.resize(old_sz + 4 + total_mb_bytes);
        uint8_t* dst = out.data() + old_sz;
        std::memcpy(dst, widths, 4);
        dst += 4;
        for (size_t m = 0; m < used; ++m) {
            uint32_t w = widths[m];
            if (w == 0) continue;
            __uint128_t acc = 0;
            uint32_t nb = 0;
            const uint64_t* dm = deltas + m * 32;
            for (int k = 0; k < 32; ++k) {
                acc |= (static_cast<__uint128_t>(dm[k]) << nb);
                nb += w;
                if (nb >= 64) {
                    uint64_t lo = static_cast<uint64_t>(acc);
                    std::memcpy(dst, &lo, 8);
                    dst += 8;
                    acc >>= 64;
                    nb -= 64;
                }
            }
            while (nb >= 8) {
                *dst++ = static_cast<uint8_t>(acc);
                acc >>= 8;
                nb -= 8;
            }
        }
        i += cnt;
    }
}

class ThriftWriter {
    std::vector<uint8_t> buf_;
    std::vector<int16_t> last_field_stack_;
    int16_t last_field_ = 0;

    void write_field_header(int16_t field_id, uint8_t type_id) {
        int16_t delta = field_id - last_field_;
        if (delta > 0 && delta <= 15) {
            buf_.push_back(static_cast<uint8_t>((delta << 4) | type_id));
        } else {
            buf_.push_back(type_id);
            append_uleb128(buf_, zigzag32(field_id));
        }
        last_field_ = field_id;
    }

public:
    void clear() {
        buf_.clear();
        last_field_stack_.clear();
        last_field_ = 0;
    }

    const uint8_t* data() const { return buf_.data(); }
    size_t size() const { return buf_.size(); }

    void struct_begin() {
        last_field_stack_.push_back(last_field_);
        last_field_ = 0;
    }

    void struct_end() {
        buf_.push_back(0x00);
        last_field_ = last_field_stack_.back();
        last_field_stack_.pop_back();
    }

    void field_i32(int16_t id, int32_t v) {
        write_field_header(id, 5);
        append_uleb128(buf_, zigzag32(v));
    }

    void field_i64(int16_t id, int64_t v) {
        write_field_header(id, 6);
        append_uleb128(buf_, zigzag64(v));
    }

    void field_string(int16_t id, std::string_view s) {
        write_field_header(id, 8);
        append_uleb128(buf_, s.size());
        buf_.insert(buf_.end(), s.begin(), s.end());
    }

    void field_struct_begin(int16_t id) {
        write_field_header(id, 12);
        struct_begin();
    }

    void field_list_begin(int16_t id, uint8_t elem_type, size_t count) {
        write_field_header(id, 9);
        if (count < 15) {
            buf_.push_back(static_cast<uint8_t>((count << 4) | elem_type));
        } else {
            buf_.push_back(static_cast<uint8_t>(0xF0 | elem_type));
            append_uleb128(buf_, count);
        }
    }

    void list_elem_i32(int32_t v) { append_uleb128(buf_, zigzag32(v)); }

    void list_elem_string(std::string_view s) {
        append_uleb128(buf_, s.size());
        buf_.insert(buf_.end(), s.begin(), s.end());
    }
};

struct ColumnSpec {
    std::string name;
    PhysicalType type;
    RepetitionType rep;
    ConvertedType converted;
};

struct ColumnChunkMeta {
    PhysicalType type;
    std::string name;
    Encoding data_enc;
    bool has_rle;
    bool has_dict;
    int64_t num_values;
    int64_t total_size;
    int64_t data_page_offset;
    int64_t dict_page_offset;
};

struct RowGroupMeta {
    int64_t num_rows;
    int64_t total_byte_size;
    std::vector<ColumnChunkMeta> columns;
};

class ParquetFileWriter {
    int fd_ = -1;
    int64_t pos_ = 0;
    std::vector<ColumnSpec> schema_;
    std::vector<RowGroupMeta> row_groups_;
    int64_t total_rows_ = 0;
    uint8_t io_buf_[256 * 1024];
    size_t io_pos_ = 0;
    std::vector<uint8_t> def_buf_;
    ThriftWriter tw_;

    bool write_all_fd(const uint8_t* p, size_t len) {
        while (len > 0) {
            ssize_t w = ::write(fd_, p, len);
            if (w <= 0) return false;
            p += static_cast<size_t>(w);
            len -= static_cast<size_t>(w);
        }
        return true;
    }

    bool flush_io() {
        if (io_pos_ == 0) return true;
        bool ok = write_all_fd(io_buf_, io_pos_);
        io_pos_ = 0;
        return ok;
    }

    bool write_raw(const void* data, size_t len) {
        if (len == 0) return true;
        pos_ += static_cast<int64_t>(len);
        const uint8_t* p = static_cast<const uint8_t*>(data);
        if (len >= sizeof(io_buf_) / 2) {
            if (!flush_io()) return false;
            return write_all_fd(p, len);
        }
        if (io_pos_ + len > sizeof(io_buf_)) {
            if (!flush_io()) return false;
        }
        std::memcpy(io_buf_ + io_pos_, p, len);
        io_pos_ += len;
        return true;
    }

    void write_data_page_header(int32_t page_size, int32_t num_values, Encoding enc) {
        tw_.clear();
        tw_.struct_begin();
        tw_.field_i32(1, PAGE_DATA);
        tw_.field_i32(2, page_size);
        tw_.field_i32(3, page_size);
        tw_.field_struct_begin(5);
        tw_.field_i32(1, num_values);
        tw_.field_i32(2, enc);
        tw_.field_i32(3, ENC_RLE);
        tw_.field_i32(4, ENC_RLE);
        tw_.struct_end();
        tw_.struct_end();
        write_raw(tw_.data(), tw_.size());
    }

    void write_dict_page_header(int32_t page_size, int32_t num_values) {
        tw_.clear();
        tw_.struct_begin();
        tw_.field_i32(1, PAGE_DICTIONARY);
        tw_.field_i32(2, page_size);
        tw_.field_i32(3, page_size);
        tw_.field_struct_begin(7);
        tw_.field_i32(1, num_values);
        tw_.field_i32(2, ENC_PLAIN);
        tw_.struct_end();
        tw_.struct_end();
        write_raw(tw_.data(), tw_.size());
    }

    void build_def_levels(int64_t num_rows, const uint8_t* def_bits) {
        def_buf_.clear();
        def_buf_.resize(4, 0);
        size_t total_groups = static_cast<size_t>((num_rows + 7) / 8);
        size_t g = 0;
        while (g < total_groups) {
            size_t run = total_groups - g;
            if (run > 1024) run = 1024;
            append_uleb128(def_buf_, static_cast<uint64_t>((run << 1) | 1));
            def_buf_.insert(def_buf_.end(), def_bits + g, def_bits + g + run);
            g += run;
        }
        uint32_t rle_len = static_cast<uint32_t>(def_buf_.size() - 4);
        std::memcpy(def_buf_.data(), &rle_len, 4);
    }

public:
    ParquetFileWriter() = default;
    ~ParquetFileWriter() {
        if (fd_ >= 0) ::close(fd_);
    }

    bool open(const std::string& path, std::vector<ColumnSpec> schema) {
        schema_ = std::move(schema);
        fd_ = ::open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0666);
        if (fd_ < 0) return false;
        pos_ = 0;
        io_pos_ = 0;
        total_rows_ = 0;
        row_groups_.clear();
        return write_raw("PAR1", 4);
    }

    struct ColumnData {
        Encoding enc = ENC_PLAIN;
        const void* plain_data = nullptr;
        size_t plain_bytes = 0;
        const uint8_t* def_bits = nullptr;
        // For ENC_RLE_DICTIONARY:
        const void* dict_data = nullptr;
        size_t dict_bytes = 0;
        int32_t dict_count = 0;
        // For split payload (e.g. DELTA_BYTE_ARRAY = hdr_data + plain_data):
        const void* extra_prefix = nullptr;
        size_t extra_prefix_bytes = 0;
    };

    bool write_row_group(int64_t num_rows, const std::vector<ColumnData>& cols) {
        if (num_rows == 0) return true;
        RowGroupMeta rg;
        rg.num_rows = num_rows;
        rg.total_byte_size = 0;
        rg.columns.reserve(schema_.size());

        for (size_t i = 0; i < schema_.size(); ++i) {
            const auto& spec = schema_[i];
            const auto& col = cols[i];
            int64_t start_off = pos_;
            bool is_opt = (spec.rep == REP_OPTIONAL);
            bool has_dict = (col.enc == ENC_RLE_DICTIONARY);
            int64_t dict_off = 0;
            int64_t data_off = start_off;

            if (has_dict) {
                dict_off = start_off;
                write_dict_page_header(static_cast<int32_t>(col.dict_bytes), col.dict_count);
                write_raw(col.dict_data, col.dict_bytes);
                data_off = pos_;
            }

            if (is_opt) {
                build_def_levels(num_rows, col.def_bits);
                int32_t page_size = static_cast<int32_t>(def_buf_.size() + col.extra_prefix_bytes + col.plain_bytes);
                write_data_page_header(page_size, static_cast<int32_t>(num_rows), col.enc);
                write_raw(def_buf_.data(), def_buf_.size());
                if (col.extra_prefix_bytes) write_raw(col.extra_prefix, col.extra_prefix_bytes);
                write_raw(col.plain_data, col.plain_bytes);
            } else {
                int32_t page_size = static_cast<int32_t>(col.extra_prefix_bytes + col.plain_bytes);
                write_data_page_header(page_size, static_cast<int32_t>(num_rows), col.enc);
                if (col.extra_prefix_bytes) write_raw(col.extra_prefix, col.extra_prefix_bytes);
                write_raw(col.plain_data, col.plain_bytes);
            }

            int64_t chunk_size = pos_ - start_off;
            rg.total_byte_size += chunk_size;
            rg.columns.push_back(ColumnChunkMeta{
                spec.type, spec.name, col.enc, is_opt, has_dict, num_rows, chunk_size, data_off, dict_off
            });
        }
        row_groups_.push_back(std::move(rg));
        total_rows_ += num_rows;
        return true;
    }

    bool close() {
        if (fd_ < 0) return false;
        tw_.clear();
        tw_.struct_begin();
        tw_.field_i32(1, 1);

        tw_.field_list_begin(2, 12, schema_.size() + 1);
        tw_.struct_begin();
        tw_.field_string(4, "schema");
        tw_.field_i32(5, static_cast<int32_t>(schema_.size()));
        tw_.struct_end();
        for (const auto& spec : schema_) {
            tw_.struct_begin();
            tw_.field_i32(1, spec.type);
            tw_.field_i32(3, spec.rep);
            tw_.field_string(4, spec.name);
            if (spec.converted != CT_NONE) {
                tw_.field_i32(6, spec.converted);
                tw_.field_struct_begin(10);
                if (spec.converted == CT_UTF8) {
                    tw_.field_struct_begin(1);
                    tw_.struct_end();
                } else if (spec.converted == CT_DATE) {
                    tw_.field_struct_begin(6);
                    tw_.struct_end();
                }
                tw_.struct_end();
            }
            tw_.struct_end();
        }

        tw_.field_i64(3, total_rows_);

        tw_.field_list_begin(4, 12, row_groups_.size());
        for (const auto& rg : row_groups_) {
            tw_.struct_begin();
            tw_.field_list_begin(1, 12, rg.columns.size());
            for (const auto& col : rg.columns) {
                tw_.struct_begin();
                tw_.field_i64(2, 0);
                tw_.field_struct_begin(3);
                tw_.field_i32(1, col.type);
                if (col.has_dict) {
                    tw_.field_list_begin(2, 5, 3);
                    tw_.list_elem_i32(ENC_PLAIN);
                    tw_.list_elem_i32(ENC_RLE);
                    tw_.list_elem_i32(ENC_RLE_DICTIONARY);
                } else if (col.has_rle || col.data_enc != ENC_PLAIN) {
                    tw_.field_list_begin(2, 5, 2);
                    tw_.list_elem_i32(col.data_enc);
                    tw_.list_elem_i32(ENC_RLE);
                } else {
                    tw_.field_list_begin(2, 5, 1);
                    tw_.list_elem_i32(ENC_PLAIN);
                }
                tw_.field_list_begin(3, 8, 1);
                tw_.list_elem_string(col.name);
                tw_.field_i32(4, 0);
                tw_.field_i64(5, col.num_values);
                tw_.field_i64(6, col.total_size);
                tw_.field_i64(7, col.total_size);
                tw_.field_i64(9, col.data_page_offset);
                if (col.has_dict) {
                    tw_.field_i64(11, col.dict_page_offset);
                }
                tw_.struct_end();
                tw_.struct_end();
            }
            tw_.field_i64(2, rg.total_byte_size);
            tw_.field_i64(3, rg.num_rows);
            tw_.struct_end();
        }
        tw_.struct_end();

        uint32_t meta_len = static_cast<uint32_t>(tw_.size());
        bool ok = write_raw(tw_.data(), tw_.size()) &&
                  write_raw(&meta_len, 4) &&
                  write_raw("PAR1", 4) &&
                  flush_io();
        if (::close(fd_) != 0) ok = false;
        fd_ = -1;
        return ok;
    }
};

} // namespace pq
