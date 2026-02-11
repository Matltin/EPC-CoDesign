#ifndef EPC_DEFS_H
#define EPC_DEFS_H

#include <systemc.h>
#include <iostream>

// ============================================================
// Configurable ASIC Parallelism
// ============================================================
// Change this to 1, 2, 4, or 8 to scale parallel update units
constexpr int NUM_ASIC = 8;

// ============================================================
// Algorithm Constants
// ============================================================
constexpr int    CFG_N        = 20;
constexpr int    CFG_D        = 10;
constexpr int    CFG_TMAX     = 100;
constexpr int    CFG_K        = 15;      // pairs_per_penguin
constexpr double CFG_LB       = -5.0;
constexpr double CFG_UB       =  5.0;
constexpr double CFG_MU0      =  0.5;
constexpr double CFG_M0       =  0.5;
constexpr double CFG_MU_DECAY =  0.99;
constexpr double CFG_M_DECAY  =  0.99;
constexpr double CFG_A        =  1.0;
constexpr double CFG_B        =  0.5;
constexpr double CFG_TINY     =  1e-300;
constexpr double CFG_EPSILON  =  1e-12;
constexpr int    FIFO_DEPTH   = CFG_N;

// ============================================================
// DimPair: pair of dimension indices (p < q)
// ============================================================
struct DimPair {
    int p, q;
    DimPair() : p(0), q(0) {}
    DimPair(int p_, int q_) : p(p_), q(q_) {}
};

// ============================================================
// Request / Response Structs for sc_fifo Communication
// ============================================================
// All use fixed-size arrays (CFG_N, CFG_D known at compile time)
// Each struct needs operator<< for sc_fifo tracing

// --- Population Initialization ---
struct InitReq {
    int N, D;
    double LB[CFG_D];
    double UB[CFG_D];
    unsigned int seed;

    InitReq() : N(0), D(0), seed(0) {
        for (int i = 0; i < CFG_D; ++i) { LB[i] = 0; UB[i] = 0; }
    }
};

inline std::ostream& operator<<(std::ostream& os, const InitReq&) {
    return os << "InitReq";
}

struct InitRes {
    double population[CFG_N][CFG_D];

    InitRes() {
        for (int i = 0; i < CFG_N; ++i)
            for (int j = 0; j < CFG_D; ++j)
                population[i][j] = 0.0;
    }
};

inline std::ostream& operator<<(std::ostream& os, const InitRes&) {
    return os << "InitRes";
}

// --- Fitness Evaluation (per-penguin) ---
struct FitnessReq {
    int penguin_idx;
    int D;
    double x[CFG_D];
    char func_name[32];

    FitnessReq() : penguin_idx(0), D(0) {
        for (int i = 0; i < CFG_D; ++i) x[i] = 0.0;
        func_name[0] = '\0';
    }
};

inline std::ostream& operator<<(std::ostream& os, const FitnessReq& r) {
    return os << "FitnessReq[" << r.penguin_idx << "]";
}

struct FitnessRes {
    int penguin_idx;
    double fitness;

    FitnessRes() : penguin_idx(0), fitness(0.0) {}
};

inline std::ostream& operator<<(std::ostream& os, const FitnessRes& r) {
    return os << "FitnessRes[" << r.penguin_idx << "]";
}

// --- Penguin Update ---
struct UpdateReq {
    int penguin_idx;
    int D;
    double x_i[CFG_D];
    double x_best[CFG_D];
    double LB[CFG_D];
    double UB[CFG_D];
    double mu, m, a, b, tiny;
    int k_pairs;
    unsigned int seed;

    UpdateReq() : penguin_idx(0), D(0), mu(0), m(0), a(0), b(0),
                  tiny(0), k_pairs(0), seed(0) {
        for (int i = 0; i < CFG_D; ++i) {
            x_i[i] = 0; x_best[i] = 0; LB[i] = 0; UB[i] = 0;
        }
    }
};

inline std::ostream& operator<<(std::ostream& os, const UpdateReq& r) {
    return os << "UpdateReq[" << r.penguin_idx << "]";
}

struct UpdateRes {
    int penguin_idx;
    double x_new[CFG_D];
    unsigned int seed_out;

    UpdateRes() : penguin_idx(0), seed_out(0) {
        for (int i = 0; i < CFG_D; ++i) x_new[i] = 0.0;
    }
};

inline std::ostream& operator<<(std::ostream& os, const UpdateRes& r) {
    return os << "UpdateRes[" << r.penguin_idx << "]";
}

#endif // EPC_DEFS_H
