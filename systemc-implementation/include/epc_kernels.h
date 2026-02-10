#ifndef EPC_KERNELS_H
#define EPC_KERNELS_H

#include "epc_defs.h"
#include <cmath>
#include <algorithm>
#include <random>
#include <unordered_set>
#include <vector>
#include <string>
#include <sstream>
#include <iomanip>
#include <numeric>
#include <fstream>

// ============================================================
// Computational Kernels (pure math, no SystemC dependency)
// Faithfully translated from epc_hw_sw_codesign reference
// ============================================================

// --- Population Initialization ---
// Translated from: asic_population.cpp::asic_initialize_population
inline void kernel_init_population(
    double population[][CFG_D], int N, int D,
    const double* LB, const double* UB, unsigned int seed
) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    for (int i = 0; i < N; ++i) {
        for (int d = 0; d < D; ++d) {
            double rand_val = dist(rng);
            population[i][d] = LB[d] + rand_val * (UB[d] - LB[d]);
        }
    }
}

// --- Sphere Function ---
// Translated from: asic_objectives.cpp::asic_sphere
inline double kernel_sphere(const double* x, int D) {
    double sum = 0.0;
    for (int i = 0; i < D; ++i) {
        sum += x[i] * x[i];
    }
    return sum;
}

// --- Rosenbrock Function ---
// f(x) = sum_{i=1..D-1} [100*(x_{i+1} - x_i^2)^2 + (x_i - 1)^2]
inline double kernel_rosenbrock(const double* x, int D) {
    double sum = 0.0;
    for (int i = 0; i < D - 1; ++i) {
        double xi = x[i];
        double xnext = x[i + 1];
        sum += 100.0 * (xnext - xi * xi) * (xnext - xi * xi)
             + (xi - 1.0) * (xi - 1.0);
    }
    return sum;
}

// --- Batch Fitness Evaluation ---
// Dispatches to the correct fitness function based on func_name
inline void kernel_evaluate_population(
    double* fitness, const double population[][CFG_D], int N, int D,
    const char* func_name
) {
    for (int i = 0; i < N; ++i) {
        if (std::string(func_name) == "Rosenbrock") {
            fitness[i] = kernel_rosenbrock(population[i], D);
        } else {
            fitness[i] = kernel_sphere(population[i], D);
        }
    }
}

// --- Pair Sampling Helpers ---
// Translated from: asic_sampling.cpp

struct PairHash {
    std::size_t operator()(const DimPair& p) const {
        return std::hash<int>()(p.p) ^ (std::hash<int>()(p.q) << 1);
    }
};

struct PairEqual {
    bool operator()(const DimPair& a, const DimPair& b) const {
        return a.p == b.p && a.q == b.q;
    }
};

inline DimPair kernel_sample_one_pair(int D, unsigned int& seed) {
    std::mt19937 rng(seed);

    int p = rng() % D;
    int q = rng() % (D - 1);

    if (q >= p) {
        q += 1;
    }

    if (p > q) {
        int temp = p;
        p = q;
        q = temp;
    }

    seed = rng();
    return DimPair(p, q);
}

inline void kernel_sample_k_unique_pairs(
    std::vector<DimPair>& pairs,
    int D, int K, unsigned int seed
) {
    pairs.clear();

    if (D < 2 || K <= 0) {
        return;
    }

    int total_pairs = D * (D - 1) / 2;

    if (K >= total_pairs) {
        pairs.reserve(total_pairs);
        for (int p = 0; p < D - 1; ++p) {
            for (int q = p + 1; q < D; ++q) {
                pairs.push_back(DimPair(p, q));
            }
        }
        return;
    }

    std::unordered_set<DimPair, PairHash, PairEqual> pairs_set;
    pairs.reserve(K);

    unsigned int current_seed = seed;

    while (pairs_set.size() < static_cast<size_t>(K)) {
        DimPair pair = kernel_sample_one_pair(D, current_seed);
        pairs_set.insert(pair);
    }

    for (const auto& pair : pairs_set) {
        pairs.push_back(pair);
    }
}

// --- Spiral Update on a Single Pair ---
// Translated from: asic_spiral.cpp::asic_spiral_update_on_pair_with_radius
inline void kernel_spiral_update(
    double* x_new, const double* x_best,
    int p, int q, double Q, double a, double b, double tiny
) {
    (void)a; // unused in the formula but kept for interface compatibility

    double theta_i = std::atan2(x_new[q], x_new[p]);
    double theta_b = std::atan2(x_best[q], x_best[p]);

    double r_i = std::sqrt(x_new[p] * x_new[p] + x_new[q] * x_new[q]);
    double r_b = std::sqrt(x_best[p] * x_best[p] + x_best[q] * x_best[q]);

    double exp_term_b = std::exp(b * theta_b);
    double exp_term_i = std::exp(b * theta_i);

    double term = (1.0 - Q) * exp_term_b + Q * exp_term_i;
    term = std::max(term, tiny);

    double theta_k = (1.0 / b) * std::log(term);

    double r_k = (1.0 - Q) * r_b + Q * r_i;

    x_new[p] = r_k * std::cos(theta_k);
    x_new[q] = r_k * std::sin(theta_k);
}

// --- Clip to Bounds ---
// Translated from: epc_utils.cpp::clip_to_bounds
inline void kernel_clip_to_bounds(double* x, const double* LB, const double* UB, int D) {
    for (int i = 0; i < D; ++i) {
        x[i] = std::max(LB[i], std::min(x[i], UB[i]));
    }
}

// --- Full Penguin Update (Method B_R) ---
// Translated from: epc_cpu.cpp::update_penguin_method_B_R
// This is the complete ASIC update kernel
inline void kernel_update_penguin(
    double* x_new,
    const double* x_i,
    const double* x_best,
    double mu, double m, double a, double b,
    const double* LB, const double* UB,
    int D, double tiny, int k_pairs,
    unsigned int& seed
) {
    // (1) Copy current position
    for (int d = 0; d < D; ++d) {
        x_new[d] = x_i[d];
    }

    // (2) Compute distance to best (CPU in reference, ASIC here)
    double dist = 0.0;
    for (int d = 0; d < D; ++d) {
        double diff = x_best[d] - x_new[d];
        dist += diff * diff;
    }
    dist = std::sqrt(dist);

    // (3) Compute Q
    double Q = std::exp(-mu * dist);
    Q = std::max(tiny, std::min(Q, 1.0));

    // (4) Sample K unique random pairs
    std::vector<DimPair> pairs;
    kernel_sample_k_unique_pairs(pairs, D, k_pairs, seed);
    seed += k_pairs;

    // (5) Spiral update on each pair (sequential - order matters!)
    for (const auto& pair : pairs) {
        kernel_spiral_update(x_new, x_best, pair.p, pair.q, Q, a, b, tiny);
    }

    // (6) Add mutation noise
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> dist_uniform(-1.0, 1.0);
    for (int d = 0; d < D; ++d) {
        double noise = dist_uniform(rng);
        x_new[d] += m * noise;
    }
    seed = rng();

    // (7) Clip to bounds
    kernel_clip_to_bounds(x_new, LB, UB, D);
}

// ============================================================
// Log Formatting Functions
// Translated from: epc_utils.cpp
// ============================================================

inline std::string epc_banner_line(const std::string& title = "EPC", int width = 91) {
    std::string mid = " " + title + " ";
    int side = (width - (int)mid.length()) / 2;
    int remainder = width - side - (int)mid.length();

    std::string result;
    result.append(side, '=');
    result.append(mid);
    result.append(remainder, '=');
    return result;
}

inline std::string epc_header_block(int N, int D, int Tmax, double init_f) {
    int total_pairs = D * (D - 1) / 2;

    std::ostringstream oss;
    oss << epc_banner_line("EPC", 91) << "\n";
    oss << "Number of Population=" << N
        << " | D=" << D
        << ", Max Iter=" << Tmax
        << " | Total pairs=" << total_pairs
        << " | Initial Fitness=" << std::fixed << std::setprecision(3) << init_f << "\n";
    oss << std::string(91, '=');
    return oss.str();
}

inline std::string epc_iter_line(int it, int Tmax, double best, double mu, double mutation) {
    std::ostringstream oss;
    oss << "Iter " << std::setw(4) << it << "/" << Tmax
        << " | best=" << std::fixed << std::setprecision(4) << best
        << " | mu=" << std::fixed << std::setprecision(3) << mu
        << " | mutation=" << std::fixed << std::setprecision(3) << mutation;
    return oss.str();
}

inline std::string epc_summary_banner() {
    return std::string(16, '=') + " EPC Benchmark Summary " + std::string(16, '=');
}

inline std::string summary_line(
    const std::string& name, int D,
    const std::vector<double>& best_fs, double total_time
) {
    int runs = (int)best_fs.size();

    double mean = std::accumulate(best_fs.begin(), best_fs.end(), 0.0) / runs;

    double variance = 0.0;
    for (double f : best_fs) {
        variance += (f - mean) * (f - mean);
    }
    double std_dev = std::sqrt(variance / runs);

    double best = *std::min_element(best_fs.begin(), best_fs.end());
    double worst = *std::max_element(best_fs.begin(), best_fs.end());

    std::ostringstream oss;
    oss << std::left << std::setw(10) << name
        << " | D=" << std::setw(4) << D
        << " | runs=" << std::setw(2) << runs
        << " | mean=" << std::scientific << std::setprecision(6) << mean
        << " \u00b1 " << std::scientific << std::setprecision(3) << std_dev
        << " | best=" << std::scientific << std::setprecision(6) << best
        << " | worst=" << std::scientific << std::setprecision(6) << worst
        << " | time=" << std::fixed << std::setprecision(3) << total_time << "s";
    return oss.str();
}

// ============================================================
// Simple Logger (file + console)
// ============================================================

class SimpleLogger {
private:
    std::ofstream file_;
    bool file_enabled_;
public:
    SimpleLogger(const std::string& filepath = "")
        : file_enabled_(false) {
        if (!filepath.empty()) {
            file_.open(filepath, std::ios::app);
            if (file_.is_open()) file_enabled_ = true;
        }
    }
    ~SimpleLogger() { close(); }

    void log(const std::string& msg = "") {
        std::cout << msg << std::endl;
        if (file_enabled_ && file_.is_open()) {
            file_ << msg << std::endl;
            file_.flush();
        }
    }

    void close() {
        if (file_.is_open()) file_.close();
        file_enabled_ = false;
    }
};

#endif // EPC_KERNELS_H
