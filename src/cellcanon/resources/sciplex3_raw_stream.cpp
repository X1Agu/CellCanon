#include <zlib.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

struct Options {
    std::string mode;
    std::string matrix;
    std::string feature_map;
    std::string cell_valid;
    std::string out_dir;
    std::string library_size;
    std::string clip_value;
    std::string selection_map;
    std::uint64_t n_cells = 0;
    std::uint64_t n_features = 0;
    std::uint64_t n_human = 0;
    std::uint64_t n_selected = 0;
};

std::string require_arg(
    const std::unordered_map<std::string, std::string>& values,
    const std::string& key
) {
    const auto it = values.find(key);
    if (it == values.end() || it->second.empty()) {
        throw std::runtime_error("Missing required argument " + key);
    }
    return it->second;
}

Options parse_args(int argc, char** argv) {
    std::unordered_map<std::string, std::string> values;
    for (int i = 1; i < argc; i += 2) {
        if (i + 1 >= argc) {
            throw std::runtime_error("Every option must have a value");
        }
        values.emplace(argv[i], argv[i + 1]);
    }
    Options options;
    options.mode = require_arg(values, "--mode");
    options.matrix = require_arg(values, "--matrix");
    options.feature_map = require_arg(values, "--feature-map");
    options.cell_valid = require_arg(values, "--cell-valid");
    options.out_dir = require_arg(values, "--out-dir");
    options.n_cells = std::stoull(require_arg(values, "--n-cells"));
    options.n_features = std::stoull(require_arg(values, "--n-features"));
    options.n_human = std::stoull(require_arg(values, "--n-human"));
    if (values.count("--library-size")) options.library_size = values.at("--library-size");
    if (values.count("--clip-value")) options.clip_value = values.at("--clip-value");
    if (values.count("--selection-map")) options.selection_map = values.at("--selection-map");
    if (values.count("--n-selected")) options.n_selected = std::stoull(values.at("--n-selected"));
    if (options.mode != "pass1" && options.mode != "pass2" && options.mode != "extract") {
        throw std::runtime_error("--mode must be pass1, pass2, or extract");
    }
    return options;
}

template <typename T>
std::vector<T> read_vector(const std::string& path, std::uint64_t expected) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) throw std::runtime_error("Cannot open " + path);
    const auto bytes = static_cast<std::uint64_t>(stream.tellg());
    if (bytes != expected * sizeof(T)) {
        throw std::runtime_error("Unexpected byte size for " + path);
    }
    stream.seekg(0);
    std::vector<T> result(expected);
    if (!stream.read(reinterpret_cast<char*>(result.data()), bytes)) {
        throw std::runtime_error("Cannot read " + path);
    }
    return result;
}

template <typename T>
void write_vector(const std::string& path, const std::vector<T>& values) {
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    if (!stream) throw std::runtime_error("Cannot create " + path);
    const auto bytes = static_cast<std::streamsize>(values.size() * sizeof(T));
    if (!stream.write(reinterpret_cast<const char*>(values.data()), bytes)) {
        throw std::runtime_error("Cannot write " + path);
    }
}

struct Triplet {
    std::uint32_t feature;
    std::uint32_t cell;
    std::uint32_t count;
};

class GzipTriplets {
public:
    explicit GzipTriplets(const std::string& path) : file_(gzopen(path.c_str(), "rb")) {
        if (!file_) throw std::runtime_error("Cannot open gzip matrix " + path);
        gzbuffer(file_, 1 << 20);
        buffer_.resize(1 << 20);
    }

    ~GzipTriplets() {
        if (file_) gzclose(file_);
    }

    bool next(Triplet& out) {
        std::uint64_t values[3] = {0, 0, 0};
        int field = 0;
        bool have_digit = false;
        while (true) {
            if (position_ == available_) {
                const int got = gzread(file_, buffer_.data(), static_cast<unsigned int>(buffer_.size()));
                if (got < 0) {
                    int error = 0;
                    throw std::runtime_error(std::string("gzip read failed: ") + gzerror(file_, &error));
                }
                if (got == 0) {
                    if (field == 0 && !have_digit) return false;
                    throw std::runtime_error("Truncated coordinate triplet at EOF");
                }
                position_ = 0;
                available_ = static_cast<std::size_t>(got);
            }
            const unsigned char ch = static_cast<unsigned char>(buffer_[position_++]);
            if (ch >= '0' && ch <= '9') {
                have_digit = true;
                values[field] = values[field] * 10 + static_cast<std::uint64_t>(ch - '0');
                if (values[field] > std::numeric_limits<std::uint32_t>::max()) {
                    throw std::runtime_error("Coordinate value exceeds uint32");
                }
            } else if (ch == ' ' || ch == '\t') {
                if (have_digit) {
                    ++field;
                    if (field > 2) throw std::runtime_error("Too many coordinate fields");
                    have_digit = false;
                }
            } else if (ch == '\n' || ch == '\r') {
                if (have_digit) ++field;
                if (field == 0) continue;
                if (field != 3) throw std::runtime_error("Coordinate row does not contain three integers");
                out.feature = static_cast<std::uint32_t>(values[0]);
                out.cell = static_cast<std::uint32_t>(values[1]);
                out.count = static_cast<std::uint32_t>(values[2]);
                return true;
            } else {
                throw std::runtime_error("Unexpected byte in coordinate matrix");
            }
        }
    }

private:
    gzFile file_ = nullptr;
    std::vector<char> buffer_;
    std::size_t position_ = 0;
    std::size_t available_ = 0;
};

struct Audit {
    std::uint64_t rows = 0;
    std::uint64_t human_rows = 0;
    std::uint64_t valid_human_rows = 0;
    std::uint64_t invalid_cell_rows = 0;
    std::uint64_t human_umis = 0;
    std::uint64_t valid_human_umis = 0;
    std::uint64_t duplicate_coordinates = 0;
    std::uint32_t max_feature = 0;
    std::uint32_t max_cell = 0;
};

void validate_triplet(
    const Options& options,
    const Triplet& value,
    std::uint32_t previous_cell,
    std::uint32_t previous_feature
) {
    if (value.feature == 0 || value.feature > options.n_features) {
        throw std::runtime_error("Feature index outside annotation range");
    }
    if (value.cell == 0 || value.cell > options.n_cells) {
        throw std::runtime_error("Cell index outside annotation range");
    }
    if (value.count == 0) throw std::runtime_error("Zero UMI coordinate encountered");
    if (value.cell < previous_cell ||
        (value.cell == previous_cell && value.feature < previous_feature)) {
        throw std::runtime_error("Coordinate matrix is not sorted by cell then feature");
    }
}

void write_audit(const Options& options, const Audit& audit) {
    std::ofstream stream(options.out_dir + "/" + options.mode + "_audit.json", std::ios::trunc);
    if (!stream) throw std::runtime_error("Cannot write audit JSON");
    stream << "{\n"
           << "  \"mode\": \"" << options.mode << "\",\n"
           << "  \"coordinate_rows\": " << audit.rows << ",\n"
           << "  \"human_coordinate_rows\": " << audit.human_rows << ",\n"
           << "  \"valid_human_coordinate_rows\": " << audit.valid_human_rows << ",\n"
           << "  \"invalid_cell_coordinate_rows\": " << audit.invalid_cell_rows << ",\n"
           << "  \"human_umis\": " << audit.human_umis << ",\n"
           << "  \"valid_human_umis\": " << audit.valid_human_umis << ",\n"
           << "  \"duplicate_coordinates\": " << audit.duplicate_coordinates << ",\n"
           << "  \"max_feature_index\": " << audit.max_feature << ",\n"
           << "  \"max_cell_index\": " << audit.max_cell << "\n"
           << "}\n";
}

template <typename Callback>
Audit scan(
    const Options& options,
    const std::vector<std::int32_t>& feature_map,
    const std::vector<std::uint8_t>& cell_valid,
    Callback callback
) {
    GzipTriplets input(options.matrix);
    Triplet value{};
    Audit audit;
    std::uint32_t previous_cell = 0;
    std::uint32_t previous_feature = 0;
    while (input.next(value)) {
        validate_triplet(options, value, previous_cell, previous_feature);
        ++audit.rows;
        audit.max_feature = std::max(audit.max_feature, value.feature);
        audit.max_cell = std::max(audit.max_cell, value.cell);
        if (value.cell == previous_cell && value.feature == previous_feature) {
            ++audit.duplicate_coordinates;
        }
        const auto human = feature_map[value.feature - 1];
        if (human >= 0) {
            ++audit.human_rows;
            audit.human_umis += value.count;
            if (cell_valid[value.cell - 1]) {
                ++audit.valid_human_rows;
                audit.valid_human_umis += value.count;
                callback(value, static_cast<std::uint32_t>(human));
            } else {
                ++audit.invalid_cell_rows;
            }
        }
        previous_cell = value.cell;
        previous_feature = value.feature;
        if (audit.rows % 100000000ULL == 0) {
            std::cerr << options.mode << ": " << audit.rows << " coordinate rows\n";
        }
    }
    if (audit.duplicate_coordinates != 0) {
        throw std::runtime_error("Duplicate feature/cell coordinates are not supported");
    }
    return audit;
}

void pass1(
    const Options& options,
    const std::vector<std::int32_t>& feature_map,
    const std::vector<std::uint8_t>& cell_valid
) {
    std::vector<std::uint64_t> library(options.n_cells, 0);
    std::vector<double> sums(options.n_human, 0.0);
    std::vector<double> sums_sq(options.n_human, 0.0);
    std::vector<std::uint64_t> nnz(options.n_human, 0);
    const auto audit = scan(options, feature_map, cell_valid, [&](const Triplet& value, std::uint32_t human) {
        library[value.cell - 1] += value.count;
        const double count = static_cast<double>(value.count);
        sums[human] += count;
        sums_sq[human] += count * count;
        ++nnz[human];
    });
    write_vector(options.out_dir + "/library_size.u64", library);
    write_vector(options.out_dir + "/raw_sum.f64", sums);
    write_vector(options.out_dir + "/raw_sumsq.f64", sums_sq);
    write_vector(options.out_dir + "/raw_nnz.u64", nnz);
    write_audit(options, audit);
}

void pass2(
    const Options& options,
    const std::vector<std::int32_t>& feature_map,
    const std::vector<std::uint8_t>& cell_valid
) {
    if (options.library_size.empty() || options.clip_value.empty()) {
        throw std::runtime_error("pass2 requires --library-size and --clip-value");
    }
    const auto library = read_vector<std::uint64_t>(options.library_size, options.n_cells);
    const auto clip = read_vector<double>(options.clip_value, options.n_human);
    std::vector<double> log_sum(options.n_human, 0.0);
    std::vector<double> log_sumsq(options.n_human, 0.0);
    std::vector<double> clip_sum(options.n_human, 0.0);
    std::vector<double> clip_sumsq(options.n_human, 0.0);
    const auto audit = scan(options, feature_map, cell_valid, [&](const Triplet& value, std::uint32_t human) {
        const auto depth = library[value.cell - 1];
        if (depth == 0) throw std::runtime_error("Valid cell has zero human library size");
        const double count = static_cast<double>(value.count);
        const double logged = std::log1p(count * 10000.0 / static_cast<double>(depth));
        log_sum[human] += logged;
        log_sumsq[human] += logged * logged;
        const double clipped = std::min(count, clip[human]);
        clip_sum[human] += clipped;
        clip_sumsq[human] += clipped * clipped;
    });
    write_vector(options.out_dir + "/log_sum.f64", log_sum);
    write_vector(options.out_dir + "/log_sumsq.f64", log_sumsq);
    write_vector(options.out_dir + "/clip_sum.f64", clip_sum);
    write_vector(options.out_dir + "/clip_sumsq.f64", clip_sumsq);
    write_audit(options, audit);
}

void extract(
    const Options& options,
    const std::vector<std::int32_t>& feature_map,
    const std::vector<std::uint8_t>& cell_valid
) {
    if (options.library_size.empty() || options.selection_map.empty() || options.n_selected == 0) {
        throw std::runtime_error("extract requires --library-size, --selection-map, and --n-selected");
    }
    const auto library = read_vector<std::uint64_t>(options.library_size, options.n_cells);
    const auto selected = read_vector<std::int32_t>(options.selection_map, options.n_features);
    std::vector<std::int32_t> cell_to_valid(options.n_cells, -1);
    std::uint64_t valid_cells = 0;
    for (std::uint64_t cell = 0; cell < options.n_cells; ++cell) {
        if (cell_valid[cell]) cell_to_valid[cell] = static_cast<std::int32_t>(valid_cells++);
    }
    std::vector<std::uint64_t> row_counts(valid_cells, 0);
    std::ofstream data_stream(options.out_dir + "/selected_log_data.f32", std::ios::binary | std::ios::trunc);
    std::ofstream count_stream(options.out_dir + "/selected_counts.u32", std::ios::binary | std::ios::trunc);
    std::ofstream index_stream(options.out_dir + "/selected_indices.u16", std::ios::binary | std::ios::trunc);
    if (!data_stream || !count_stream || !index_stream) {
        throw std::runtime_error("Cannot create selected sparse arrays");
    }
    std::uint64_t selected_nnz = 0;
    const auto audit = scan(options, feature_map, cell_valid, [&](const Triplet& value, std::uint32_t) {
        const auto column = selected[value.feature - 1];
        if (column < 0) return;
        if (static_cast<std::uint64_t>(column) >= options.n_selected ||
            options.n_selected > std::numeric_limits<std::uint16_t>::max()) {
            throw std::runtime_error("Selected column outside uint16 range");
        }
        const auto row = cell_to_valid[value.cell - 1];
        if (row < 0) throw std::runtime_error("Valid-cell map changed during extraction");
        const auto depth = library[value.cell - 1];
        if (depth == 0) throw std::runtime_error("Valid cell has zero human library size");
        const float logged = static_cast<float>(std::log1p(
            static_cast<double>(value.count) * 10000.0 / static_cast<double>(depth)
        ));
        const auto col16 = static_cast<std::uint16_t>(column);
        data_stream.write(reinterpret_cast<const char*>(&logged), sizeof(logged));
        count_stream.write(reinterpret_cast<const char*>(&value.count), sizeof(value.count));
        index_stream.write(reinterpret_cast<const char*>(&col16), sizeof(col16));
        if (!data_stream || !count_stream || !index_stream) {
            throw std::runtime_error("Failed while writing selected sparse arrays");
        }
        ++row_counts[static_cast<std::uint64_t>(row)];
        ++selected_nnz;
    });
    std::vector<std::uint64_t> indptr(valid_cells + 1, 0);
    for (std::uint64_t row = 0; row < valid_cells; ++row) {
        indptr[row + 1] = indptr[row] + row_counts[row];
    }
    if (indptr.back() != selected_nnz) throw std::runtime_error("CSR row counts do not conserve nnz");
    write_vector(options.out_dir + "/selected_indptr.u64", indptr);
    std::ofstream summary(options.out_dir + "/extract_summary.json", std::ios::trunc);
    summary << "{\n"
            << "  \"valid_cells\": " << valid_cells << ",\n"
            << "  \"selected_genes\": " << options.n_selected << ",\n"
            << "  \"selected_nnz\": " << selected_nnz << "\n"
            << "}\n";
    write_audit(options, audit);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto options = parse_args(argc, argv);
        const auto feature_map = read_vector<std::int32_t>(options.feature_map, options.n_features);
        const auto cell_valid = read_vector<std::uint8_t>(options.cell_valid, options.n_cells);
        for (const auto value : feature_map) {
            if (value < -1 || (value >= 0 && static_cast<std::uint64_t>(value) >= options.n_human)) {
                throw std::runtime_error("Feature-to-human map contains an invalid index");
            }
        }
        if (options.mode == "pass1") pass1(options, feature_map, cell_valid);
        if (options.mode == "pass2") pass2(options, feature_map, cell_valid);
        if (options.mode == "extract") extract(options, feature_map, cell_valid);
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "sciplex3_raw_stream: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
