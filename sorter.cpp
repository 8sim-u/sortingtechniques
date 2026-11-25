#include <algorithm>
#include <arpa/inet.h>
#include <cctype>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <netinet/in.h>
#include <random>
#include <sstream>
#include <string>
#include <sys/socket.h>
#include <unistd.h>
#include <vector>

#if __has_include("stb_image.h") && __has_include("stb_image_write.h")
#define HAS_STB_IMAGE 1
#if defined(__clang__) || defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"
#pragma GCC diagnostic ignored "-Wextra"
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#endif
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#if defined(__clang__) || defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
#else
#define HAS_STB_IMAGE 0
#endif

// =============================================================================
// 1. Data Structures
// =============================================================================

struct Pixel {
    int r;
    int g;
    int b;
    int a;
    int originalIndex;
};

struct Stats {
    long long comparisons = 0;
    long long swaps = 0;
    double elapsedMs = 0.0;
};

using Clock = std::chrono::high_resolution_clock;

// =============================================================================
// 2. Test Row Generation & Scrambling Helpers
// =============================================================================

// Creates a sorted row of `size` pixels with originalIndex values 0..size-1.
std::vector<Pixel> generateTestRow(int size) {
    std::vector<Pixel> row;
    row.reserve(size);
    for (int i = 0; i < size; ++i) {
        int colorVal = (size > 1) ? (255 * i) / (size - 1) : 0;
        row.push_back({colorVal, 128, 255 - colorVal, 255, i});
    }
    return row;
}

// Randomly permutes `row` in-place using the Fisher-Yates shuffle: O(n).
void shuffleRow(std::vector<Pixel>& row, std::mt19937& rng) {
    for (int i = static_cast<int>(row.size()) - 1; i > 0; --i) {
        std::uniform_int_distribution<int> dist(0, i);
        int j = dist(rng);
        std::swap(row[i], row[j]);
    }
}

// Reverses `row` in-place (useful for worst-case testing): O(n).
void reverseRow(std::vector<Pixel>& row) {
    std::reverse(row.begin(), row.end());
}

// Restores `row` from the unscrambled `originalRow` backup: O(n).
void resetRow(std::vector<Pixel>& row, const std::vector<Pixel>& originalRow) {
    row = originalRow;
}

// =============================================================================
// 3. Sorting Algorithms (all sort by Pixel::originalIndex ascending)
// =============================================================================

// 1) Bubble Sort: O(n^2) worst/average, O(n) best (early exit when no swaps).
void bubbleSort(std::vector<Pixel>& row, Stats& stats) {
    stats = Stats{};
    auto start = Clock::now();

    int n = static_cast<int>(row.size());
    for (int i = 0; i < n - 1; ++i) {
        bool swapped = false;
        for (int j = 0; j < n - 1 - i; ++j) {
            stats.comparisons++;
            if (row[j].originalIndex > row[j + 1].originalIndex) {
                std::swap(row[j], row[j + 1]);
                stats.swaps++;
                swapped = true;
            }
        }
        if (!swapped) break;
    }

    auto end = Clock::now();
    stats.elapsedMs = std::chrono::duration<double, std::milli>(end - start).count();
}

// 2) Selection Sort: O(n^2) comparisons, at most O(n) swaps.
void selectionSort(std::vector<Pixel>& row, Stats& stats) {
    stats = Stats{};
    auto start = Clock::now();

    int n = static_cast<int>(row.size());
    for (int i = 0; i < n - 1; ++i) {
        int minIdx = i;
        for (int j = i + 1; j < n; ++j) {
            stats.comparisons++;
            if (row[j].originalIndex < row[minIdx].originalIndex) {
                minIdx = j;
            }
        }
        if (minIdx != i) {
            std::swap(row[i], row[minIdx]);
            stats.swaps++;
        }
    }

    auto end = Clock::now();
    stats.elapsedMs = std::chrono::duration<double, std::milli>(end - start).count();
}

// 3) Insertion Sort: O(n^2) worst, O(n) best (nearly sorted).
void insertionSort(std::vector<Pixel>& row, Stats& stats) {
    stats = Stats{};
    auto start = Clock::now();

    int n = static_cast<int>(row.size());
    for (int i = 1; i < n; ++i) {
        Pixel key = row[i];
        int j = i - 1;
        while (j >= 0) {
            stats.comparisons++;
            if (row[j].originalIndex > key.originalIndex) {
                row[j + 1] = row[j];
                stats.swaps++;
                --j;
            } else {
                break;
            }
        }
        row[j + 1] = key;
    }

    auto end = Clock::now();
    stats.elapsedMs = std::chrono::duration<double, std::milli>(end - start).count();
}

// 4) Merge Sort: O(n log n) guaranteed, Divide & Conquer.
namespace {
void mergeRanges(std::vector<Pixel>& row, int left, int mid, int right, Stats& stats) {
    std::vector<Pixel> leftPart(row.begin() + left, row.begin() + mid + 1);
    std::vector<Pixel> rightPart(row.begin() + mid + 1, row.begin() + right + 1);

    int i = 0;
    int j = 0;
    int k = left;

    while (i < static_cast<int>(leftPart.size()) && j < static_cast<int>(rightPart.size())) {
        stats.comparisons++;
        if (leftPart[i].originalIndex <= rightPart[j].originalIndex) {
            row[k++] = leftPart[i++];
        } else {
            row[k++] = rightPart[j++];
        }
        stats.swaps++;
    }

    while (i < static_cast<int>(leftPart.size())) {
        row[k++] = leftPart[i++];
        stats.swaps++;
    }

    while (j < static_cast<int>(rightPart.size())) {
        row[k++] = rightPart[j++];
        stats.swaps++;
    }
}

void mergeSortRange(std::vector<Pixel>& row, int left, int right, Stats& stats) {
    if (left >= right) return;
    int mid = left + (right - left) / 2;
    mergeSortRange(row, left, mid, stats);
    mergeSortRange(row, mid + 1, right, stats);
    mergeRanges(row, left, mid, right, stats);
}
}  // namespace

void mergeSort(std::vector<Pixel>& row, Stats& stats) {
    stats = Stats{};
    auto start = Clock::now();

    if (!row.empty()) {
        mergeSortRange(row, 0, static_cast<int>(row.size()) - 1, stats);
    }

    auto end = Clock::now();
    stats.elapsedMs = std::chrono::duration<double, std::milli>(end - start).count();
}

// 5) Quicksort: O(n log n) average, O(n^2) worst-case (Lomuto partition).
namespace {
int partitionLomuto(std::vector<Pixel>& row, int low, int high, Stats& stats) {
    const Pixel& pivot = row[high];
    int i = low;

    for (int j = low; j < high; ++j) {
        stats.comparisons++;
        if (row[j].originalIndex <= pivot.originalIndex) {
            if (i != j) {
                std::swap(row[i], row[j]);
                stats.swaps++;
            }
            ++i;
        }
    }

    if (i != high) {
        std::swap(row[i], row[high]);
        stats.swaps++;
    }
    return i;
}

void quickSortRange(std::vector<Pixel>& row, int low, int high, Stats& stats) {
    if (low >= high) return;
    int pivotIdx = partitionLomuto(row, low, high, stats);
    quickSortRange(row, low, pivotIdx - 1, stats);
    quickSortRange(row, pivotIdx + 1, high, stats);
}
}  // namespace

void quickSort(std::vector<Pixel>& row, Stats& stats) {
    stats = Stats{};
    auto start = Clock::now();

    if (!row.empty()) {
        quickSortRange(row, 0, static_cast<int>(row.size()) - 1, stats);
    }

    auto end = Clock::now();
    stats.elapsedMs = std::chrono::duration<double, std::milli>(end - start).count();
}

// 6) Heap Sort: O(n log n) guaranteed, in-place using a max-heap.
namespace {
void heapify(std::vector<Pixel>& row, int n, int root, Stats& stats) {
    int largest = root;
    int left = 2 * root + 1;
    int right = 2 * root + 2;

    if (left < n) {
        stats.comparisons++;
        if (row[left].originalIndex > row[largest].originalIndex) {
            largest = left;
        }
    }

    if (right < n) {
        stats.comparisons++;
        if (row[right].originalIndex > row[largest].originalIndex) {
            largest = right;
        }
    }

    if (largest != root) {
        std::swap(row[root], row[largest]);
        stats.swaps++;
        heapify(row, n, largest, stats);
    }
}
}  // namespace

void heapSort(std::vector<Pixel>& row, Stats& stats) {
    stats = Stats{};
    auto start = Clock::now();

    int n = static_cast<int>(row.size());
    // Build max-heap bottom-up
    for (int i = n / 2 - 1; i >= 0; --i) {
        heapify(row, n, i, stats);
    }

    // Extract elements from heap one by one
    for (int i = n - 1; i > 0; --i) {
        std::swap(row[0], row[i]);
        stats.swaps++;
        heapify(row, i, 0, stats);
    }

    auto end = Clock::now();
    stats.elapsedMs = std::chrono::duration<double, std::milli>(end - start).count();
}

// 7) Bucket Sort: O(n + k) average when keys are uniformly distributed.
//    Scatters pixels into sqrt(n) buckets, sorts each bucket with Insertion Sort,
//    then concatenates the buckets back into `row`.
void bucketSort(std::vector<Pixel>& row, Stats& stats) {
    stats = Stats{};
    auto start = Clock::now();

    int n = static_cast<int>(row.size());
    if (n > 1) {
        int minIdx = row[0].originalIndex;
        int maxIdx = row[0].originalIndex;
        for (int i = 1; i < n; ++i) {
            if (row[i].originalIndex < minIdx) minIdx = row[i].originalIndex;
            if (row[i].originalIndex > maxIdx) maxIdx = row[i].originalIndex;
        }

        if (minIdx != maxIdx) {
            int numBuckets = std::max(1, static_cast<int>(std::sqrt(n)));
            std::vector<std::vector<Pixel>> buckets(numBuckets);
            double range = static_cast<double>(maxIdx - minIdx + 1);

            // Scatter pixels into buckets
            for (const Pixel& p : row) {
                int b = static_cast<int>(((p.originalIndex - minIdx) / range) * numBuckets);
                if (b >= numBuckets) b = numBuckets - 1;
                buckets[b].push_back(p);
                stats.swaps++;
            }

            // Sort each bucket individually with Insertion Sort and gather back
            int k = 0;
            for (auto& bucket : buckets) {
                for (int i = 1; i < static_cast<int>(bucket.size()); ++i) {
                    Pixel key = bucket[i];
                    int j = i - 1;
                    while (j >= 0) {
                        stats.comparisons++;
                        if (bucket[j].originalIndex > key.originalIndex) {
                            bucket[j + 1] = bucket[j];
                            stats.swaps++;
                            --j;
                        } else {
                            break;
                        }
                    }
                    bucket[j + 1] = key;
                }
                for (const Pixel& p : bucket) {
                    row[k++] = p;
                    stats.swaps++;
                }
            }
        }
    }

    auto end = Clock::now();
    stats.elapsedMs = std::chrono::duration<double, std::milli>(end - start).count();
}

// =============================================================================
// 4. Grid Operations (2D Image Model: size x size)
// =============================================================================

using Grid = std::vector<std::vector<Pixel>>;

// Creates a `size x size` grid where each row y has pixels 0..size-1 with distinct RGB values.
Grid makeGrid(int size) {
    Grid grid(size);
    int denom = std::max(1, size - 1);
    for (int y = 0; y < size; ++y) {
        grid[y].reserve(size);
        for (int x = 0; x < size; ++x) {
            int r = (255 * x) / denom;
            int g = (255 * y) / denom;
            int b = (255 * (x + y)) / (2 * denom);
            grid[y].push_back({r, g, b, 255, x});
        }
    }
    return grid;
}

void shuffleGrid(Grid& grid, std::mt19937& rng) {
    for (auto& row : grid) {
        shuffleRow(row, rng);
    }
}

void reverseGrid(Grid& grid) {
    for (auto& row : grid) {
        reverseRow(row);
    }
}

// Restores every row to originalIndex 0..size-1 and its original pixel data.
void resetGrid(Grid& grid) {
    grid = makeGrid(static_cast<int>(grid.size()));
}

bool isGridSorted(const Grid& grid) {
    for (const auto& row : grid) {
        bool rowSorted = std::is_sorted(
            row.begin(), row.end(),
            [](const Pixel& a, const Pixel& b) {
                return a.originalIndex < b.originalIndex;
            }
        );
        if (!rowSorted) return false;
    }
    return true;
}

// Runs the selected algorithm on every row of `grid`, accumulating comparisons,
// swaps, and total wall-clock elapsedMs into `totalStats`.
void sortGrid(Grid& grid, const std::string& algo, Stats& totalStats) {
    using SortFn = void (*)(std::vector<Pixel>&, Stats&);
    SortFn sortFn = nullptr;

    if (algo == "bubble") sortFn = bubbleSort;
    else if (algo == "selection") sortFn = selectionSort;
    else if (algo == "insertion") sortFn = insertionSort;
    else if (algo == "merge") sortFn = mergeSort;
    else if (algo == "quick") sortFn = quickSort;
    else if (algo == "heap") sortFn = heapSort;
    else if (algo == "bucket") sortFn = bucketSort;

    totalStats = Stats{};
    if (!sortFn) {
        std::cerr << "Unknown algorithm: " << algo << "\n";
        return;
    }

    auto start = Clock::now();
    for (auto& row : grid) {
        Stats rowStats;
        sortFn(row, rowStats);
        totalStats.comparisons += rowStats.comparisons;
        totalStats.swaps += rowStats.swaps;
    }
    auto end = Clock::now();
    totalStats.elapsedMs = std::chrono::duration<double, std::milli>(end - start).count();
}

// =============================================================================
// 5. Image I/O (via stb_image.h & stb_image_write.h)
// =============================================================================

// Loads an image file into `grid` with originalIndex = 0..outWidth-1 per row.
// If `targetSize > 0`, resizes to `targetSize x targetSize` using nearest-neighbor.
bool loadImageToGrid(const std::string& path, Grid& grid, int targetSize = 0) {
#if !HAS_STB_IMAGE
    (void)path;
    (void)grid;
    (void)targetSize;
    std::cerr << "Error: stb_image.h / stb_image_write.h not found. Cannot load image.\n";
    return false;
#else
    int width = 0;
    int height = 0;
    int channels = 0;
    unsigned char* data = stbi_load(path.c_str(), &width, &height, &channels, 4);
    if (!data) {
        std::cerr << "Error loading image '" << path << "': "
                  << (stbi_failure_reason() ? stbi_failure_reason() : "unknown error") << "\n";
        return false;
    }

    int outW = (targetSize > 0) ? targetSize : width;
    int outH = (targetSize > 0) ? targetSize : height;

    grid.assign(outH, std::vector<Pixel>(outW));
    for (int y = 0; y < outH; ++y) {
        int srcY = (targetSize > 0) ? (y * height) / outH : y;
        for (int x = 0; x < outW; ++x) {
            int srcX = (targetSize > 0) ? (x * width) / outW : x;
            int offset = (srcY * width + srcX) * 4;
            grid[y][x] = {
                data[offset],
                data[offset + 1],
                data[offset + 2],
                data[offset + 3],
                x
            };
        }
    }

    stbi_image_free(data);
    return true;
#endif
}

// Writes `grid` back out as an image file (PNG by default, or JPG/BMP/TGA by extension).
bool saveGridToImage(const std::string& path, const Grid& grid) {
#if !HAS_STB_IMAGE
    (void)path;
    (void)grid;
    std::cerr << "Error: stb_image.h / stb_image_write.h not found. Cannot save image.\n";
    return false;
#else
    if (grid.empty() || grid[0].empty()) {
        std::cerr << "Error: Cannot save an empty grid.\n";
        return false;
    }

    int height = static_cast<int>(grid.size());
    int width = static_cast<int>(grid[0].size());
    std::vector<unsigned char> rgba(static_cast<size_t>(width) * height * 4);

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const Pixel& p = grid[y][x];
            size_t offset = (static_cast<size_t>(y) * width + x) * 4;
            rgba[offset] = static_cast<unsigned char>(std::clamp(p.r, 0, 255));
            rgba[offset + 1] = static_cast<unsigned char>(std::clamp(p.g, 0, 255));
            rgba[offset + 2] = static_cast<unsigned char>(std::clamp(p.b, 0, 255));
            rgba[offset + 3] = static_cast<unsigned char>(std::clamp(p.a, 0, 255));
        }
    }

    auto endsWith = [&](const std::string& ext) {
        return path.size() >= ext.size() &&
               path.compare(path.size() - ext.size(), ext.size(), ext) == 0;
    };

    int ok = 0;
    if (endsWith(".jpg") || endsWith(".jpeg")) {
        ok = stbi_write_jpg(path.c_str(), width, height, 4, rgba.data(), 95);
    } else if (endsWith(".bmp")) {
        ok = stbi_write_bmp(path.c_str(), width, height, 4, rgba.data());
    } else if (endsWith(".tga")) {
        ok = stbi_write_tga(path.c_str(), width, height, 4, rgba.data());
    } else {
        ok = stbi_write_png(path.c_str(), width, height, 4, rgba.data(), width * 4);
    }

    if (!ok) {
        std::cerr << "Error saving image to '" << path << "'.\n";
        return false;
    }
    return true;
#endif
}

// =============================================================================
// 6. Local HTTP Server (--serve on http://127.0.0.1:8080)
// =============================================================================

namespace {

using SortFn = void (*)(std::vector<Pixel>&, Stats&);

struct AlgoComplexity {
    std::string best;
    std::string average;
    std::string worst;
    std::string space;
    bool stable;
};

AlgoComplexity getAlgoComplexity(const std::string& algo) {
    if (algo == "bubble")    return {"O(n)",       "O(n^2)",     "O(n^2)",     "O(1)",     true};
    if (algo == "selection") return {"O(n^2)",     "O(n^2)",     "O(n^2)",     "O(1)",     false};
    if (algo == "insertion") return {"O(n)",       "O(n^2)",     "O(n^2)",     "O(1)",     true};
    if (algo == "merge")     return {"O(n log n)", "O(n log n)", "O(n log n)", "O(n)",     true};
    if (algo == "quick")     return {"O(n log n)", "O(n log n)", "O(n^2)",     "O(log n)", false};
    if (algo == "heap")      return {"O(n log n)", "O(n log n)", "O(n log n)", "O(1)",     false};
    if (algo == "bucket")    return {"O(n + k)",   "O(n + k)",   "O(n^2)",     "O(n + k)", true};
    return {"?", "?", "?", "?", false};
}

SortFn getSortFunction(const std::string& algo) {
    if (algo == "bubble") return bubbleSort;
    if (algo == "selection") return selectionSort;
    if (algo == "insertion") return insertionSort;
    if (algo == "merge") return mergeSort;
    if (algo == "quick") return quickSort;
    if (algo == "heap") return heapSort;
    if (algo == "bucket") return bucketSort;
    return nullptr;
}

std::vector<unsigned char> gridToRgba(const Grid& grid) {
    int height = static_cast<int>(grid.size());
    int width = height > 0 ? static_cast<int>(grid[0].size()) : 0;
    std::vector<unsigned char> rgba(static_cast<size_t>(width) * height * 4);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const Pixel& p = grid[y][x];
            size_t off = (static_cast<size_t>(y) * width + x) * 4;
            rgba[off] = static_cast<unsigned char>(std::clamp(p.r, 0, 255));
            rgba[off + 1] = static_cast<unsigned char>(std::clamp(p.g, 0, 255));
            rgba[off + 2] = static_cast<unsigned char>(std::clamp(p.b, 0, 255));
            rgba[off + 3] = static_cast<unsigned char>(std::clamp(p.a, 0, 255));
        }
    }
    return rgba;
}

std::string base64Encode(const std::vector<unsigned char>& data) {
    static constexpr char kTable[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    out.reserve(((data.size() + 2) / 3) * 4);

    for (size_t i = 0; i < data.size(); i += 3) {
        uint32_t b0 = data[i];
        uint32_t b1 = (i + 1 < data.size()) ? data[i + 1] : 0;
        uint32_t b2 = (i + 2 < data.size()) ? data[i + 2] : 0;
        uint32_t triple = (b0 << 16) | (b1 << 8) | b2;

        out.push_back(kTable[(triple >> 18) & 0x3F]);
        out.push_back(kTable[(triple >> 12) & 0x3F]);
        out.push_back((i + 1 < data.size()) ? kTable[(triple >> 6) & 0x3F] : '=');
        out.push_back((i + 2 < data.size()) ? kTable[triple & 0x3F] : '=');
    }
    return out;
}

std::string toLower(std::string s) {
    for (char& c : s) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return s;
}

std::string urlDecode(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (size_t i = 0; i < s.size(); ++i) {
        if (s[i] == '+') {
            out.push_back(' ');
        } else if (s[i] == '%' && i + 2 < s.size()) {
            auto hexVal = [](char c) -> int {
                if (c >= '0' && c <= '9') return c - '0';
                if (c >= 'a' && c <= 'f') return c - 'a' + 10;
                if (c >= 'A' && c <= 'F') return c - 'A' + 10;
                return -1;
            };
            int hi = hexVal(s[i + 1]);
            int lo = hexVal(s[i + 2]);
            if (hi >= 0 && lo >= 0) {
                out.push_back(static_cast<char>((hi << 4) | lo));
                i += 2;
                continue;
            }
            out.push_back(s[i]);
        } else {
            out.push_back(s[i]);
        }
    }
    return out;
}

std::string getQueryParam(const std::string& query, const std::string& key, const std::string& fallback = "") {
    std::istringstream ss(query);
    std::string pair;
    while (std::getline(ss, pair, '&')) {
        size_t eq = pair.find('=');
        if (eq != std::string::npos && pair.substr(0, eq) == key) {
            return urlDecode(pair.substr(eq + 1));
        }
    }
    return fallback;
}

bool sendAll(int fd, const char* data, size_t len) {
    size_t sent = 0;
    while (sent < len) {
        ssize_t n = send(fd, data + sent, len - sent, 0);
        if (n <= 0) return false;
        sent += static_cast<size_t>(n);
    }
    return true;
}

void sendHttpResponse(int clientFd, int statusCode, const std::string& statusText,
                      const std::string& contentType, const std::string& body) {
    std::ostringstream hdr;
    hdr << "HTTP/1.1 " << statusCode << " " << statusText << "\r\n"
        << "Content-Type: " << contentType << "\r\n"
        << "Content-Length: " << body.size() << "\r\n"
        << "Access-Control-Allow-Origin: *\r\n"
        << "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
        << "Access-Control-Allow-Headers: Content-Type, X-Width, X-Height, X-Algo, X-Algos, X-Scramble, X-Mode, X-Max-Frames\r\n"
        << "Connection: close\r\n\r\n";
    std::string headerStr = hdr.str();
    sendAll(clientFd, headerStr.data(), headerStr.size());
    if (!body.empty()) {
        sendAll(clientFd, body.data(), body.size());
    }
}

// Records evenly spaced RGBA snapshots up to `maxFrames` during per-row sorting.
struct FrameRecorder {
    std::vector<std::vector<unsigned char>>& frames;
    int maxFrames;
    int stepInterval;
    int currentStep = 0;

    FrameRecorder(std::vector<std::vector<unsigned char>>& f, int maxF, int totalSteps)
        : frames(f),
          maxFrames(std::max(1, maxF)),
          stepInterval(std::max(1, (std::max(1, totalSteps) + maxFrames - 1) / maxFrames)) {}

    void tick(const Grid& grid) {
        currentStep++;
        if (currentStep % stepInterval == 0 && static_cast<int>(frames.size()) < maxFrames) {
            frames.push_back(gridToRgba(grid));
        }
    }
};

// Sorts all rows together phase-by-phase so each algorithm's visual signature
// (right-to-left bubble, left-to-right selection, merging blocks, pivot splits,
// heap extraction, bucket bands) is clearly visible across the image.
void sortGridPerRowWithFrames(Grid& grid, const std::string& algo, int maxFrames,
                              Stats& totalStats, std::vector<std::vector<unsigned char>>& frames) {
    totalStats = Stats{};
    int height = static_cast<int>(grid.size());
    int width = height > 0 ? static_cast<int>(grid[0].size()) : 0;
    if (height == 0 || width <= 1) {
        frames.push_back(gridToRgba(grid));
        return;
    }

    auto start = Clock::now();

    if (algo == "bubble") {
        FrameRecorder rec(frames, maxFrames, width - 1);
        for (int i = 0; i < width - 1; ++i) {
            bool anySwapped = false;
            for (int y = 0; y < height; ++y) {
                for (int j = 0; j < width - 1 - i; ++j) {
                    totalStats.comparisons++;
                    if (grid[y][j].originalIndex > grid[y][j + 1].originalIndex) {
                        std::swap(grid[y][j], grid[y][j + 1]);
                        totalStats.swaps++;
                        anySwapped = true;
                    }
                }
            }
            rec.tick(grid);
            if (!anySwapped) break;
        }
    } else if (algo == "selection") {
        FrameRecorder rec(frames, maxFrames, width - 1);
        for (int i = 0; i < width - 1; ++i) {
            for (int y = 0; y < height; ++y) {
                int minIdx = i;
                for (int j = i + 1; j < width; ++j) {
                    totalStats.comparisons++;
                    if (grid[y][j].originalIndex < grid[y][minIdx].originalIndex) {
                        minIdx = j;
                    }
                }
                if (minIdx != i) {
                    std::swap(grid[y][i], grid[y][minIdx]);
                    totalStats.swaps++;
                }
            }
            rec.tick(grid);
        }
    } else if (algo == "insertion") {
        FrameRecorder rec(frames, maxFrames, width - 1);
        for (int i = 1; i < width; ++i) {
            for (int y = 0; y < height; ++y) {
                Pixel key = grid[y][i];
                int j = i - 1;
                while (j >= 0) {
                    totalStats.comparisons++;
                    if (grid[y][j].originalIndex > key.originalIndex) {
                        grid[y][j + 1] = grid[y][j];
                        totalStats.swaps++;
                        --j;
                    } else {
                        break;
                    }
                }
                grid[y][j + 1] = key;
            }
            rec.tick(grid);
        }
    } else if (algo == "merge") {
        // Bottom-up iterative Merge Sort: merges 2-pixel, 4-pixel, 8-pixel... bands across all rows.
        int totalMergeOps = 0;
        for (int b = 1; b < width; b *= 2) {
            for (int left = 0; left < width - b; left += 2 * b) {
                totalMergeOps++;
            }
        }
        FrameRecorder rec(frames, maxFrames, totalMergeOps);
        for (int b = 1; b < width; b *= 2) {
            for (int left = 0; left < width - b; left += 2 * b) {
                int mid = left + b - 1;
                int right = std::min(left + 2 * b - 1, width - 1);
                for (int y = 0; y < height; ++y) {
                    mergeRanges(grid[y], left, mid, right, totalStats);
                }
                rec.tick(grid);
            }
        }
    } else if (algo == "quick") {
        // Level-by-level iterative Quicksort: partitions all active sub-ranges across rows.
        std::vector<std::vector<std::pair<int, int>>> rowQueues(height);
        std::vector<size_t> rowHeads(height, 0);
        for (int y = 0; y < height; ++y) {
            rowQueues[y].push_back({0, width - 1});
        }
        FrameRecorder rec(frames, maxFrames, width);

        bool active = true;
        while (active) {
            active = false;
            for (int y = 0; y < height; ++y) {
                if (rowHeads[y] < rowQueues[y].size()) {
                    auto [low, high] = rowQueues[y][rowHeads[y]++];
                    if (low < high) {
                        int p = partitionLomuto(grid[y], low, high, totalStats);
                        if (low < p - 1) rowQueues[y].push_back({low, p - 1});
                        if (p + 1 < high) rowQueues[y].push_back({p + 1, high});
                        active = true;
                    }
                }
            }
            if (active) rec.tick(grid);
        }
    } else if (algo == "heap") {
        // Phase 1: build max-heap across all rows; Phase 2: extract max from right to left.
        FrameRecorder rec(frames, maxFrames, (width / 2) + (width - 1));
        for (int i = width / 2 - 1; i >= 0; --i) {
            for (int y = 0; y < height; ++y) {
                heapify(grid[y], width, i, totalStats);
            }
            rec.tick(grid);
        }
        for (int i = width - 1; i > 0; --i) {
            for (int y = 0; y < height; ++y) {
                std::swap(grid[y][0], grid[y][i]);
                totalStats.swaps++;
                heapify(grid[y], i, 0, totalStats);
            }
            rec.tick(grid);
        }
    } else if (algo == "bucket") {
        // Phase 1: scatter into sqrt(width) buckets; Phase 2: insertion-sort each bucket.
        int numBuckets = std::max(1, static_cast<int>(std::sqrt(width)));
        std::vector<std::vector<std::vector<Pixel>>> rowBuckets(height, std::vector<std::vector<Pixel>>(numBuckets));
        std::vector<std::vector<int>> bucketOffsets(height, std::vector<int>(numBuckets, 0));

        for (int y = 0; y < height; ++y) {
            int minIdx = grid[y][0].originalIndex;
            int maxIdx = grid[y][0].originalIndex;
            for (int x = 1; x < width; ++x) {
                if (grid[y][x].originalIndex < minIdx) minIdx = grid[y][x].originalIndex;
                if (grid[y][x].originalIndex > maxIdx) maxIdx = grid[y][x].originalIndex;
            }
            double range = static_cast<double>(std::max(1, maxIdx - minIdx + 1));
            for (const Pixel& p : grid[y]) {
                int b = static_cast<int>(((p.originalIndex - minIdx) / range) * numBuckets);
                if (b >= numBuckets) b = numBuckets - 1;
                rowBuckets[y][b].push_back(p);
                totalStats.swaps++;
            }
            int k = 0;
            for (int b = 0; b < numBuckets; ++b) {
                bucketOffsets[y][b] = k;
                for (const Pixel& p : rowBuckets[y][b]) {
                    grid[y][k++] = p;
                    totalStats.swaps++;
                }
            }
        }
        frames.push_back(gridToRgba(grid));  // Show coarse bucket bands

        FrameRecorder rec(frames, maxFrames - 1, numBuckets);
        for (int b = 0; b < numBuckets; ++b) {
            for (int y = 0; y < height; ++y) {
                int off = bucketOffsets[y][b];
                int len = static_cast<int>(rowBuckets[y][b].size());
                for (int i = 1; i < len; ++i) {
                    Pixel key = grid[y][off + i];
                    int j = i - 1;
                    while (j >= 0) {
                        totalStats.comparisons++;
                        if (grid[y][off + j].originalIndex > key.originalIndex) {
                            grid[y][off + j + 1] = grid[y][off + j];
                            totalStats.swaps++;
                            --j;
                        } else {
                            break;
                        }
                    }
                    grid[y][off + j + 1] = key;
                }
            }
            rec.tick(grid);
        }
    }

    auto end = Clock::now();
    totalStats.elapsedMs = std::chrono::duration<double, std::milli>(end - start).count();
    frames.push_back(gridToRgba(grid));  // Always include final sorted frame
}

// Sorts `grid` either concurrently across all rows (`per_row`) or one row at a time (`sequential_rows`).
void sortGridWithFrames(Grid& grid, const std::string& algo, const std::string& mode, int maxFrames,
                        Stats& totalStats, std::vector<std::vector<unsigned char>>& frames) {
    if (mode != "sequential_rows") {
        sortGridPerRowWithFrames(grid, algo, maxFrames, totalStats, frames);
        return;
    }

    SortFn sortFn = getSortFunction(algo);
    totalStats = Stats{};
    int height = static_cast<int>(grid.size());
    int step = std::max(1, (height + maxFrames - 1) / std::max(1, maxFrames));

    auto start = Clock::now();
    for (int y = 0; y < height; ++y) {
        Stats rowStats;
        sortFn(grid[y], rowStats);
        totalStats.comparisons += rowStats.comparisons;
        totalStats.swaps += rowStats.swaps;

        if ((y + 1) % step == 0 || y == height - 1) {
            frames.push_back(gridToRgba(grid));
        }
    }
    auto end = Clock::now();
    totalStats.elapsedMs = std::chrono::duration<double, std::milli>(end - start).count();
}

void handleClient(int clientFd) {
    std::string raw;
    char buf[8192];
    size_t headerEnd = std::string::npos;

    while ((headerEnd = raw.find("\r\n\r\n")) == std::string::npos) {
        ssize_t n = recv(clientFd, buf, sizeof(buf), 0);
        if (n <= 0) return;
        raw.append(buf, static_cast<size_t>(n));
        if (raw.size() > 65536) break;
    }
    if (headerEnd == std::string::npos) {
        sendHttpResponse(clientFd, 400, "Bad Request", "text/plain", "Malformed HTTP headers");
        return;
    }

    std::string headerBlock = raw.substr(0, headerEnd);
    std::string body = raw.substr(headerEnd + 4);

    std::istringstream headerStream(headerBlock);
    std::string requestLine;
    std::getline(headerStream, requestLine);
    if (!requestLine.empty() && requestLine.back() == '\r') requestLine.pop_back();

    std::istringstream reqLineStream(requestLine);
    std::string method, target;
    reqLineStream >> method >> target;

    std::string path = target;
    std::string query;
    size_t qPos = target.find('?');
    if (qPos != std::string::npos) {
        path = target.substr(0, qPos);
        query = target.substr(qPos + 1);
    }

    size_t contentLength = 0;
    std::string hdrWidth, hdrHeight, hdrAlgo, hdrAlgos, hdrScramble, hdrMode, hdrMaxFrames;
    std::string line;
    while (std::getline(headerStream, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        size_t colon = line.find(':');
        if (colon == std::string::npos) continue;
        std::string key = toLower(line.substr(0, colon));
        std::string val = line.substr(colon + 1);
        size_t firstNonSpace = val.find_first_not_of(" \t");
        val = (firstNonSpace == std::string::npos) ? "" : val.substr(firstNonSpace);

        if (key == "content-length") contentLength = static_cast<size_t>(std::stoul(val));
        else if (key == "x-width") hdrWidth = val;
        else if (key == "x-height") hdrHeight = val;
        else if (key == "x-algo") hdrAlgo = val;
        else if (key == "x-algos") hdrAlgos = val;
        else if (key == "x-scramble") hdrScramble = val;
        else if (key == "x-mode") hdrMode = val;
        else if (key == "x-max-frames") hdrMaxFrames = val;
    }

    while (body.size() < contentLength) {
        ssize_t n = recv(clientFd, buf, sizeof(buf), 0);
        if (n <= 0) break;
        body.append(buf, static_cast<size_t>(n));
    }
    if (body.size() > contentLength) {
        body.resize(contentLength);
    }

    if (method == "OPTIONS") {
        sendHttpResponse(clientFd, 204, "No Content", "text/plain", "");
        return;
    }

    if (method == "GET" && path == "/") {
        std::ifstream file("index.html", std::ios::binary);
        if (!file) {
            sendHttpResponse(clientFd, 404, "Not Found", "text/plain", "index.html not found");
            return;
        }
        std::ostringstream ss;
        ss << file.rdbuf();
        sendHttpResponse(clientFd, 200, "OK", "text/html; charset=utf-8", ss.str());
        return;
    }

    auto parseIntOr = [](const std::string& s, int def) {
        if (s.empty()) return def;
        try { return std::stoi(s); } catch (...) { return def; }
    };

    auto decodeAndScramble = [&](int& width, int& height, const std::string& scramble, Grid& grid,
                                 std::string& errJson) -> bool {
        if (width > 0 && height > 0 && body.size() == static_cast<size_t>(width) * height * 4) {
            grid.assign(height, std::vector<Pixel>(width));
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    size_t off = (static_cast<size_t>(y) * width + x) * 4;
                    grid[y][x] = {
                        static_cast<unsigned char>(body[off]),
                        static_cast<unsigned char>(body[off + 1]),
                        static_cast<unsigned char>(body[off + 2]),
                        static_cast<unsigned char>(body[off + 3]),
                        x
                    };
                }
            }
        } else {
#if HAS_STB_IMAGE
            int imgW = 0, imgH = 0, ch = 0;
            unsigned char* decoded = stbi_load_from_memory(
                reinterpret_cast<const unsigned char*>(body.data()),
                static_cast<int>(body.size()), &imgW, &imgH, &ch, 4);
            if (!decoded) {
                errJson = "{\"error\":\"Invalid pixel or image payload\"}";
                return false;
            }
            int outW = (width > 0) ? width : imgW;
            int outH = (height > 0) ? height : imgH;
            grid.assign(outH, std::vector<Pixel>(outW));
            for (int y = 0; y < outH; ++y) {
                int srcY = (y * imgH) / outH;
                for (int x = 0; x < outW; ++x) {
                    int srcX = (x * imgW) / outW;
                    size_t off = (static_cast<size_t>(srcY) * imgW + srcX) * 4;
                    grid[y][x] = {decoded[off], decoded[off + 1], decoded[off + 2], decoded[off + 3], x};
                }
            }
            stbi_image_free(decoded);
            width = outW;
            height = outH;
#else
            errJson = "{\"error\":\"Invalid raw RGBA payload size\"}";
            return false;
#endif
        }

        if (scramble == "shuffle") {
            std::random_device rd;
            std::mt19937 rng(rd());
            shuffleGrid(grid, rng);
        } else if (scramble == "reverse") {
            reverseGrid(grid);
        }
        return true;
    };

    if (method == "POST" && path == "/api/sort") {
        int width = parseIntOr(getQueryParam(query, "width", hdrWidth), 0);
        int height = parseIntOr(getQueryParam(query, "height", hdrHeight), 0);
        std::string algo = getQueryParam(query, "algo", hdrAlgo.empty() ? "bubble" : hdrAlgo);
        std::string scramble = getQueryParam(query, "scramble", hdrScramble.empty() ? "shuffle" : hdrScramble);
        std::string mode = getQueryParam(query, "mode", hdrMode.empty() ? "per_row" : hdrMode);
        int maxFrames = std::clamp(parseIntOr(getQueryParam(query, "maxFrames", hdrMaxFrames), 60), 1, 240);

        if (!getSortFunction(algo)) {
            sendHttpResponse(clientFd, 400, "Bad Request", "application/json",
                             "{\"error\":\"Unknown algorithm\"}");
            return;
        }

        Grid grid;
        std::string errJson;
        if (!decodeAndScramble(width, height, scramble, grid, errJson)) {
            sendHttpResponse(clientFd, 400, "Bad Request", "application/json", errJson);
            return;
        }

        std::vector<std::vector<unsigned char>> frames;
        frames.push_back(gridToRgba(grid));  // Frame 0: scrambled initial state

        Stats totalStats;
        sortGridWithFrames(grid, algo, mode, maxFrames, totalStats, frames);
        AlgoComplexity comp = getAlgoComplexity(algo);

        std::ostringstream json;
        json << "{\"width\":" << width
             << ",\"height\":" << height
             << ",\"comparisons\":" << totalStats.comparisons
             << ",\"swaps\":" << totalStats.swaps
             << ",\"elapsedMs\":" << std::fixed << std::setprecision(4) << totalStats.elapsedMs
             << ",\"bestTime\":\"" << comp.best << "\""
             << ",\"avgTime\":\"" << comp.average << "\""
             << ",\"worstTime\":\"" << comp.worst << "\""
             << ",\"space\":\"" << comp.space << "\""
             << ",\"stable\":" << (comp.stable ? "true" : "false")
             << ",\"frameCount\":" << frames.size()
             << ",\"frames\":[";
        for (size_t i = 0; i < frames.size(); ++i) {
            if (i > 0) json << ",";
            json << "\"" << base64Encode(frames[i]) << "\"";
        }
        json << "]}";

        sendHttpResponse(clientFd, 200, "OK", "application/json", json.str());
        return;
    }

    if (method == "POST" && path == "/api/race") {
        int width = parseIntOr(getQueryParam(query, "width", hdrWidth), 0);
        int height = parseIntOr(getQueryParam(query, "height", hdrHeight), 0);
        std::string algosParam = getQueryParam(query, "algos", hdrAlgos);
        std::string scramble = getQueryParam(query, "scramble", hdrScramble.empty() ? "shuffle" : hdrScramble);
        std::string mode = getQueryParam(query, "mode", hdrMode.empty() ? "per_row" : hdrMode);
        int maxFrames = std::clamp(parseIntOr(getQueryParam(query, "maxFrames", hdrMaxFrames), 60), 1, 240);

        std::vector<std::string> algos;
        std::istringstream algoStream(algosParam);
        std::string item;
        while (std::getline(algoStream, item, ',')) {
            if (!item.empty()) {
                if (!getSortFunction(item)) {
                    sendHttpResponse(clientFd, 400, "Bad Request", "application/json",
                                     "{\"error\":\"Unknown algorithm in algos list\"}");
                    return;
                }
                algos.push_back(item);
            }
        }
        if (algos.size() < 2 || algos.size() > 3) {
            sendHttpResponse(clientFd, 400, "Bad Request", "application/json",
                             "{\"error\":\"Race mode requires 2 or 3 algorithms\"}");
            return;
        }

        Grid baseGrid;
        std::string errJson;
        if (!decodeAndScramble(width, height, scramble, baseGrid, errJson)) {
            sendHttpResponse(clientFd, 400, "Bad Request", "application/json", errJson);
            return;
        }

        std::vector<unsigned char> scrambledFrame = gridToRgba(baseGrid);

        std::ostringstream json;
        json << "{\"width\":" << width
             << ",\"height\":" << height
             << ",\"results\":[";

        for (size_t idx = 0; idx < algos.size(); ++idx) {
            Grid testGrid = baseGrid;
            std::vector<std::vector<unsigned char>> frames;
            frames.push_back(scrambledFrame);  // Frame 0: identical scrambled start

            Stats stats;
            sortGridWithFrames(testGrid, algos[idx], mode, maxFrames, stats, frames);
            AlgoComplexity comp = getAlgoComplexity(algos[idx]);

            if (idx > 0) json << ",";
            json << "{\"algo\":\"" << algos[idx] << "\""
                 << ",\"comparisons\":" << stats.comparisons
                 << ",\"swaps\":" << stats.swaps
                 << ",\"elapsedMs\":" << std::fixed << std::setprecision(4) << stats.elapsedMs
                 << ",\"bestTime\":\"" << comp.best << "\""
                 << ",\"avgTime\":\"" << comp.average << "\""
                 << ",\"worstTime\":\"" << comp.worst << "\""
                 << ",\"space\":\"" << comp.space << "\""
                 << ",\"stable\":" << (comp.stable ? "true" : "false")
                 << ",\"frameCount\":" << frames.size()
                 << ",\"frames\":[";
            for (size_t i = 0; i < frames.size(); ++i) {
                if (i > 0) json << ",";
                json << "\"" << base64Encode(frames[i]) << "\"";
            }
            json << "]}";
        }
        json << "]}";

        sendHttpResponse(clientFd, 200, "OK", "application/json", json.str());
        return;
    }

    sendHttpResponse(clientFd, 404, "Not Found", "text/plain", "Not Found");
}

}  // namespace

int runServer(int port = 8080) {
    int serverFd = socket(AF_INET, SOCK_STREAM, 0);
    if (serverFd < 0) {
        std::cerr << "Error: Failed to create socket.\n";
        return 1;
    }

    int opt = 1;
    setsockopt(serverFd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = inet_addr("127.0.0.1");
    addr.sin_port = htons(static_cast<uint16_t>(port));

    if (bind(serverFd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
        std::cerr << "Error: Failed to bind to 127.0.0.1:" << port << "\n";
        close(serverFd);
        return 1;
    }

    if (listen(serverFd, 16) < 0) {
        std::cerr << "Error: Failed to listen on port " << port << "\n";
        close(serverFd);
        return 1;
    }

    std::cout << "DAA Image Row Sorter server running at http://127.0.0.1:" << port << "\n";
    std::cout << "Press Ctrl+C to stop.\n";

    while (true) {
        int clientFd = accept(serverFd, nullptr, nullptr);
        if (clientFd < 0) continue;
        handleClient(clientFd);
        close(clientFd);
    }

    close(serverFd);
    return 0;
}

// =============================================================================
// 7. CLI Test Runner
// =============================================================================

struct GridAlgorithmEntry {
    std::string key;
    std::string label;
};

int runBenchmark() {
    constexpr int kGridSize = 50;
    std::mt19937 rng(42);

    Grid baseGrid = makeGrid(kGridSize);
    shuffleGrid(baseGrid, rng);

    std::vector<GridAlgorithmEntry> algorithms = {
        {"bubble", "Bubble Sort"},
        {"selection", "Selection Sort"},
        {"insertion", "Insertion Sort"},
        {"merge", "Merge Sort"},
        {"quick", "Quicksort"},
        {"heap", "Heap Sort"},
        {"bucket", "Bucket Sort"},
    };

    std::cout << "DAA Grid Sorter Benchmark (" << kGridSize << " x " << kGridSize << ", shuffled)\n";
    std::cout << std::string(118, '-') << "\n";
    std::cout << std::left << std::setw(16) << "Algorithm"
              << std::right << std::setw(12) << "Comparisons"
              << std::setw(10) << "Swaps"
              << std::setw(11) << "Time (ms)"
              << std::setw(8) << "Status"
              << "  " << std::left
              << std::setw(12) << "Best"
              << std::setw(12) << "Average"
              << std::setw(12) << "Worst"
              << std::setw(10) << "Space"
              << std::setw(8) << "Stable" << "\n";
    std::cout << std::string(118, '-') << "\n";

    bool allPassed = true;
    for (const auto& algo : algorithms) {
        Grid testGrid = baseGrid;
        Stats totalStats;
        sortGrid(testGrid, algo.key, totalStats);
        AlgoComplexity comp = getAlgoComplexity(algo.key);

        bool sorted = isGridSorted(testGrid);
        if (!sorted) allPassed = false;

        std::cout << std::left << std::setw(16) << algo.label
                  << std::right << std::setw(12) << totalStats.comparisons
                  << std::setw(10) << totalStats.swaps
                  << std::setw(11) << std::fixed << std::setprecision(4) << totalStats.elapsedMs
                  << std::setw(8) << (sorted ? "PASS" : "FAIL")
                  << "  " << std::left
                  << std::setw(12) << comp.best
                  << std::setw(12) << comp.average
                  << std::setw(12) << comp.worst
                  << std::setw(10) << comp.space
                  << std::setw(8) << (comp.stable ? "Yes" : "No") << "\n";
    }

    std::cout << std::string(118, '-') << "\n";
    return allPassed ? 0 : 1;
}

int main(int argc, char* argv[]) {
    if (argc == 1) {
        return runBenchmark();
    }

    if (argc == 2 && std::string(argv[1]) == "--serve") {
        return runServer(8080);
    }

    if (argc < 4 || argc > 5) {
        std::cerr << "Usage:\n"
                  << "  " << argv[0] << "                                        (run 50x50 benchmark)\n"
                  << "  " << argv[0] << " --serve                                (start HTTP server on :8080)\n"
                  << "  " << argv[0] << " <input_img> <output_img> <algo> [shuffle|reverse]\n";
        return 1;
    }

    const std::string inputPath = argv[1];
    const std::string outputPath = argv[2];
    const std::string algo = argv[3];
    const std::string scrambleMode = (argc == 5) ? argv[4] : "none";

    Grid grid;
    if (!loadImageToGrid(inputPath, grid)) {
        return 1;
    }

    if (scrambleMode == "shuffle") {
        std::mt19937 rng(42);
        shuffleGrid(grid, rng);
    } else if (scrambleMode == "reverse") {
        reverseGrid(grid);
    } else if (scrambleMode != "none") {
        std::cerr << "Unknown scramble mode '" << scrambleMode << "'. Use 'shuffle' or 'reverse'.\n";
        return 1;
    }

    Stats totalStats;
    sortGrid(grid, algo, totalStats);
    bool sorted = isGridSorted(grid);

    if (!saveGridToImage(outputPath, grid)) {
        return 1;
    }

    std::cout << "Image: " << inputPath << " (" << grid[0].size() << " x " << grid.size() << ")\n"
              << "Algorithm:   " << algo << "\n"
              << "Scramble:    " << scrambleMode << "\n"
              << "Comparisons: " << totalStats.comparisons << "\n"
              << "Swaps:       " << totalStats.swaps << "\n"
              << "Time (ms):   " << std::fixed << std::setprecision(4) << totalStats.elapsedMs << "\n"
              << "Status:      " << (sorted ? "PASS" : "FAIL") << "\n"
              << "Saved to:    " << outputPath << "\n";

    return sorted ? 0 : 1;
}
