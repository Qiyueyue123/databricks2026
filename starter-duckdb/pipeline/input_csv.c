#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include <fcntl.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

struct field {
    const char *start;
    const char *end;
    bool has_cr;
};

static const char *find_byte(const char *start, const char *end, char byte)
{
    return memchr(start, byte, (size_t)(end - start));
}

static bool has_width(const char *start, const char *end, size_t width)
{
    for (size_t i = 1; i < width; i++) {
        const char *comma = find_byte(start, end, ',');
        if (!comma)
            return false;
        start = comma + 1;
    }
    return !find_byte(start, end, ',');
}

static bool parse_fields(const char *start, const char *end, size_t width,
                         struct field *fields)
{
    for (size_t i = 0; i < width; i++) {
        const char *comma = find_byte(start, end, ',');
        if ((comma != NULL) != (i + 1 < width))
            return false;

        const char *field_end = comma ? comma : end;
        const char *field_start = start;
        if (field_end - field_start >= 2 && *field_start == '"' && field_end[-1] == '"') {
            field_start++;
            field_end--;
        }
        if ((i == 0 && field_end == field_start) || find_byte(field_start, field_end, '"'))
            return false;

        if (fields) {
            fields[i] = (struct field){
                field_start, field_end, find_byte(field_start, field_end, '\r') != NULL
            };
        }
        if (comma)
            start = comma + 1;
    }
    return true;
}

static bool write_bytes(FILE *target, const char *data, size_t size)
{
    return fwrite(data, 1, size, target) == size;
}

static bool write_fields(FILE *target, const struct field *fields, size_t width)
{
    for (size_t i = 0; i < width; i++) {
        const struct field *field = &fields[i];
        // DuckDB needs carriage returns quoted to preserve them as field content.
        if (field->has_cr && fputc('"', target) == EOF)
            return false;
        if (!write_bytes(target, field->start, (size_t)(field->end - field->start)))
            return false;
        if (field->has_cr && fputc('"', target) == EOF)
            return false;
        if (fputc(i + 1 == width ? '\n' : ',', target) == EOF)
            return false;
    }
    return true;
}

static bool normalize_bytes(FILE *target, const char *start, const char *end, size_t width)
{
    const char *header_end = find_byte(start, end, '\n');
    if (!header_end || width == 0)
        return true;

    struct field fields[width];
    start = header_end + 1;
    const char *run_start = start;

    while (start < end) {
        const char *newline = find_byte(start, end, '\n');
        const char *line_end = newline ? newline : end;
        const char *next = newline ? newline + 1 : end;
        bool nonempty_id = start != line_end && *start != ',';
        bool has_cr = nonempty_id && find_byte(start, line_end, '\r');
        bool has_quotes = nonempty_id && find_byte(start, line_end, '"');
        bool valid = nonempty_id && (has_cr || has_quotes
            ? parse_fields(start, line_end, width, has_cr ? fields : NULL)
            : has_width(start, line_end, width));

        // DuckDB handles whole-field quotes and quoted nulls. Only rewrite CR rows.
        if (valid && !has_cr) {
            if (!newline) {
                return write_bytes(target, run_start, (size_t)(end - run_start)) &&
                       fputc('\n', target) != EOF;
            }
        } else {
            if (!write_bytes(target, run_start, (size_t)(start - run_start)))
                return false;
            if (valid && !write_fields(target, fields, width))
                return false;
            run_start = next;
        }
        start = next;
    }
    return write_bytes(target, run_start, (size_t)(end - run_start));
}

static int normalize_file(FILE *target, const char *path, size_t width)
{
    int source = open(path, O_RDONLY);
    if (source == -1)
        return errno;

    struct stat info;
    const char *data = MAP_FAILED;
    size_t size = 0;
    int error = 0;
    if (fstat(source, &info) == -1) {
        error = errno;
    } else if (info.st_size < 0 || (uintmax_t)info.st_size > PTRDIFF_MAX) {
        error = EOVERFLOW;
    } else if (info.st_size != 0) {
        size = (size_t)info.st_size;
        data = mmap(NULL, size, PROT_READ, MAP_PRIVATE, source, 0);
        if (data == MAP_FAILED)
            error = errno;
    }
    close(source);

    if (data != MAP_FAILED) {
        if (!normalize_bytes(target, data, data + size, width))
            error = errno ? errno : EIO;
        munmap((void *)data, size);
    }
    return error;
}

int normalize_csv(const char *target_path, const char *header, size_t header_size,
                  size_t width, const char *const *source_paths, size_t source_count)
{
    FILE *target = fopen(target_path, "wb");
    if (!target)
        return errno;

    char output_buffer[64 * 1024];
    setvbuf(target, output_buffer, _IOFBF, sizeof(output_buffer));
    int error = 0;
    if (!write_bytes(target, header, header_size))
        error = errno ? errno : EIO;
    for (size_t i = 0; i < source_count && !error; i++)
        error = normalize_file(target, source_paths[i], width);
    if (fclose(target) == EOF && !error)
        error = errno ? errno : EIO;
    return error;
}
