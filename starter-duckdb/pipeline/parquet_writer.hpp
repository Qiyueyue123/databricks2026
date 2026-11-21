#pragma once

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <string_view>
#include <vector>

namespace pq {

// Minimal, zero-dependency, Apache Parquet v1 writer (UNCOMPRESSED, PLAIN + RLE def levels).
// Encodes metadata with Thrift Compact Protocol per parquet.thrift.

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
};

class ThriftWriter {
    std::vector<uint8_t> buf_;
    std::vector<int16_t> last_field_stack_;
    int16_t last_field_ = 0;

    void write_varint(uint64_t v) {
        while (v >= 0x80) {
            buf_.push_back(static_cast<uint8_t>((v & 0x7F) | 0x80));
            v >>= 7;
        }
        buf_.push_back(static_cast<uint8_t>(v));
    }

    static uint64_t zigzag64(int64_t v) {
        return (static_cast<uint64_t>(v) << 1) ^ static_cast<uint64_t>(v >> 63);
    }

    static uint32_t zigzag32(int32_t v) {
        return (static_cast<uint32_t>(v) << 1) ^ static_cast<uint32_t>(v >> 31);
    }

    void write_field_header(int16_t field_id, uint8_t type_id) {
        int16_t delta = field_id - last_field_;
        if (delta > 0 && delta <= 15) {
            buf_.push_back(static_cast<uint8_t>((delta << 4) | type_id));
        } else {
            buf_.push_back(type_id);
            write_varint(zigzag32(field_id));
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
        write_varint(zigzag32(v));
    }

    void field_i64(int16_t id, int64_t v) {
        write_field_header(id, 6);
        write_varint(zigzag64(v));
    }

    void field_string(int16_t id, std::string_view s) {
        write_field_header(id, 8);
        write_varint(s.size());
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
            write_varint(count);
        }
    }

    void list_elem_i32(int32_t v) { write_varint(zigzag32(v)); }

    void list_elem_string(std::string_view s) {
        write_varint(s.size());
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
    bool has_rle;
    int64_t num_values;
    int64_t total_size;
    int64_t data_page_offset;
};

struct RowGroupMeta {
    int64_t num_rows;
    int64_t total_byte_size;
    std::vector<ColumnChunkMeta> columns;
};

class ParquetFileWriter {
    FILE* fp_ = nullptr;
    int64_t pos_ = 0;
    std::vector<ColumnSpec> schema_;
    std::vector<RowGroupMeta> row_groups_;
    int64_t total_rows_ = 0;
    std::vector<uint8_t> io_buf_;
    std::vector<uint8_t> def_buf_;
    ThriftWriter tw_;

    bool write_raw(const void* data, size_t len) {
        if (len == 0) return true;
        if (fwrite(data, 1, len, fp_) != len) return false;
        pos_ += static_cast<int64_t>(len);
        return true;
    }

    static void append_uleb128(std::vector<uint8_t>& out, uint32_t v) {
        while (v >= 0x80) {
            out.push_back(static_cast<uint8_t>((v & 0x7F) | 0x80));
            v >>= 7;
        }
        out.push_back(static_cast<uint8_t>(v));
    }

    void write_page_header(int32_t page_size, int32_t num_values) {
        tw_.clear();
        tw_.struct_begin();
        tw_.field_i32(1, 0);          // DATA_PAGE
        tw_.field_i32(2, page_size);  // uncompressed_page_size
        tw_.field_i32(3, page_size);  // compressed_page_size
        tw_.field_struct_begin(5);    // data_page_header
        tw_.field_i32(1, num_values); // num_values
        tw_.field_i32(2, ENC_PLAIN);  // encoding
        tw_.field_i32(3, ENC_RLE);    // definition_level_encoding
        tw_.field_i32(4, ENC_RLE);    // repetition_level_encoding
        tw_.struct_end();
        tw_.struct_end();
        write_raw(tw_.data(), tw_.size());
    }

public:
    ParquetFileWriter() : io_buf_(1 << 20) {}
    ~ParquetFileWriter() {
        if (fp_) fclose(fp_);
    }

    bool open(const std::string& path, std::vector<ColumnSpec> schema) {
        schema_ = std::move(schema);
        fp_ = fopen(path.c_str(), "wb");
        if (!fp_) return false;
        setvbuf(fp_, reinterpret_cast<char*>(io_buf_.data()), _IOFBF, io_buf_.size());
        pos_ = 0;
        total_rows_ = 0;
        row_groups_.clear();
        return write_raw("PAR1", 4);
    }

    struct ColumnData {
        // For INT32 / INT64 / required BYTE_ARRAY: raw PLAIN bytes.
        const void* plain_data = nullptr;
        size_t plain_bytes = 0;
        // For optional BYTE_ARRAY (def_bits != nullptr): 1 bit per row (LSB first).
        const uint8_t* def_bits = nullptr;
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

            if (!is_opt) {
                int32_t page_size = static_cast<int32_t>(col.plain_bytes);
                write_page_header(page_size, static_cast<int32_t>(num_rows));
                write_raw(col.plain_data, col.plain_bytes);
            } else {
                // Encode 1-bit definition levels in bit-packed runs of <= 1024 groups (8192 values).
                def_buf_.clear();
                def_buf_.resize(4, 0); // placeholder for 4-byte LE byte length
                size_t total_groups = static_cast<size_t>((num_rows + 7) / 8);
                size_t g = 0;
                while (g < total_groups) {
                    size_t run = total_groups - g;
                    if (run > 1024) run = 1024;
                    append_uleb128(def_buf_, static_cast<uint32_t>((run << 1) | 1));
                    def_buf_.insert(def_buf_.end(), col.def_bits + g, col.def_bits + g + run);
                    g += run;
                }
                uint32_t rle_len = static_cast<uint32_t>(def_buf_.size() - 4);
                std::memcpy(def_buf_.data(), &rle_len, 4);

                int32_t page_size = static_cast<int32_t>(def_buf_.size() + col.plain_bytes);
                write_page_header(page_size, static_cast<int32_t>(num_rows));
                write_raw(def_buf_.data(), def_buf_.size());
                write_raw(col.plain_data, col.plain_bytes);
            }

            int64_t chunk_size = pos_ - start_off;
            rg.total_byte_size += chunk_size;
            rg.columns.push_back(ColumnChunkMeta{
                spec.type, spec.name, is_opt, num_rows, chunk_size, start_off
            });
        }
        row_groups_.push_back(std::move(rg));
        total_rows_ += num_rows;
        return true;
    }

    bool close() {
        if (!fp_) return false;
        tw_.clear();
        tw_.struct_begin();
        tw_.field_i32(1, 1); // version = 1

        // field 2: schema
        tw_.field_list_begin(2, 12, schema_.size() + 1);
        // Root element
        tw_.struct_begin();
        tw_.field_string(4, "schema");
        tw_.field_i32(5, static_cast<int32_t>(schema_.size()));
        tw_.struct_end();
        // Leaf elements
        for (const auto& spec : schema_) {
            tw_.struct_begin();
            tw_.field_i32(1, spec.type);
            tw_.field_i32(3, spec.rep);
            tw_.field_string(4, spec.name);
            if (spec.converted != CT_NONE) {
                tw_.field_i32(6, spec.converted);
                tw_.field_struct_begin(10); // logicalType union
                if (spec.converted == CT_UTF8) {
                    tw_.field_struct_begin(1); // StringType
                    tw_.struct_end();
                } else if (spec.converted == CT_DATE) {
                    tw_.field_struct_begin(6); // DateType
                    tw_.struct_end();
                }
                tw_.struct_end();
            }
            tw_.struct_end();
        }

        // field 3: num_rows
        tw_.field_i64(3, total_rows_);

        // field 4: row_groups
        tw_.field_list_begin(4, 12, row_groups_.size());
        for (const auto& rg : row_groups_) {
            tw_.struct_begin();
            tw_.field_list_begin(1, 12, rg.columns.size());
            for (const auto& col : rg.columns) {
                tw_.struct_begin();
                tw_.field_i64(2, 0); // file_offset
                tw_.field_struct_begin(3); // meta_data
                tw_.field_i32(1, col.type);
                if (col.has_rle) {
                    tw_.field_list_begin(2, 5, 2);
                    tw_.list_elem_i32(ENC_RLE);
                    tw_.list_elem_i32(ENC_PLAIN);
                } else {
                    tw_.field_list_begin(2, 5, 1);
                    tw_.list_elem_i32(ENC_PLAIN);
                }
                tw_.field_list_begin(3, 8, 1);
                tw_.list_elem_string(col.name);
                tw_.field_i32(4, 0); // UNCOMPRESSED
                tw_.field_i64(5, col.num_values);
                tw_.field_i64(6, col.total_size);
                tw_.field_i64(7, col.total_size);
                tw_.field_i64(9, col.data_page_offset);
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
                  write_raw("PAR1", 4);
        if (fclose(fp_) != 0) ok = false;
        fp_ = nullptr;
        return ok;
    }
};

} // namespace pq
