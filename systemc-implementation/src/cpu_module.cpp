#include "cpu_module.h"
#include "epc_kernels.h"

#include <chrono>
#include <cstring>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <algorithm>

// ============================================================
// CPU_Module::run() - Main EPC Optimization Thread
// ============================================================
// This SC_THREAD contains the full benchmark:
//   - 3 runs with seeds {0, 1, 2}
//   - Each run: init -> eval -> 100 iterations -> result
//   - All ASIC work via sc_fifo blocking communication
//   - Produces output matching the reference out.txt
// ============================================================

void CPU_Module::run()
{
    const std::string LOG_PATH = "out.txt";

    // Clear log file at start
    {
        std::ofstream ofs(LOG_PATH, std::ios::trunc);
    }

    // Problem configuration
    const std::string problem_name = "Rosenbrock"; // Sphere && Rosenbrock
    const int D = CFG_D;
    const int N = CFG_N;
    const double lb = CFG_LB;
    const double ub = CFG_UB;

    // Seeds for 3 runs
    unsigned int seeds[] = {0, 1, 2};
    int num_runs = 3;

    // Bounds arrays
    double LB[CFG_D], UB[CFG_D];
    for (int d = 0; d < D; ++d)
    {
        LB[d] = lb;
        UB[d] = ub;
    }

    // Collect results across runs
    std::vector<double> all_best_fs;
    double total_time = 0.0;

    auto t_total_start = std::chrono::high_resolution_clock::now();

    // ========================================================
    // Run Loop
    // ========================================================
    for (int run_idx = 0; run_idx < num_runs; ++run_idx)
    {
        unsigned int seed = seeds[run_idx];

        SimpleLogger logger(LOG_PATH);

        // ====================================================
        // STEP 1: Initialize population via ASIC_Init
        // ====================================================
        InitReq ireq;
        ireq.N = N;
        ireq.D = D;
        ireq.seed = seed;
        std::memcpy(ireq.LB, LB, sizeof(double) * D);
        std::memcpy(ireq.UB, UB, sizeof(double) * D);

        init_req_out.write(ireq);          // CPU -> ASIC_Init
        InitRes ires = init_res_in.read(); // ASIC_Init -> CPU

        // Local population copy
        double population[CFG_N][CFG_D];
        std::memcpy(population, ires.population, sizeof(population));

        // ====================================================
        // STEP 2: Initial fitness evaluation via ASIC_Fitness
        // ====================================================
        // Distribute penguins round-robin to NUM_ASIC fitness units
        double fitness[CFG_N];
        {
            int fit_count[NUM_ASIC];
            for (int a = 0; a < NUM_ASIC; ++a)
                fit_count[a] = 0;

            for (int i = 0; i < N; ++i)
            {
                int target = i % NUM_ASIC;

                FitnessReq freq;
                freq.penguin_idx = i;
                freq.D = D;
                std::memcpy(freq.x, population[i], sizeof(double) * D);
                std::strncpy(freq.func_name, problem_name.c_str(), sizeof(freq.func_name) - 1);
                freq.func_name[sizeof(freq.func_name) - 1] = '\0';

                fitness_req_out[target].write(freq);
                fit_count[target]++;
            }

            for (int a = 0; a < NUM_ASIC; ++a)
            {
                for (int c = 0; c < fit_count[a]; ++c)
                {
                    FitnessRes fres = fitness_res_in[a].read();
                    fitness[fres.penguin_idx] = fres.fitness;
                }
            }
        }

        // ====================================================
        // STEP 3: Find initial best
        // ====================================================
        int best_idx = 0;
        double best_f = fitness[0];
        for (int i = 1; i < N; ++i)
        {
            if (fitness[i] < best_f)
            {
                best_f = fitness[i];
                best_idx = i;
            }
        }
        double best_x[CFG_D];
        std::memcpy(best_x, population[best_idx], sizeof(double) * D);

        // Dynamic parameters
        double mu = CFG_MU0;
        double m = CFG_M0;

        // ====================================================
        // STEP 4: Print header (matching reference format)
        // ====================================================
        logger.log(epc_header_block(N, D, CFG_TMAX, best_f));
        {
            std::ostringstream oss;
            oss << "Update Method: B_R | pairs_per_penguin(k)=" << CFG_K;
            logger.log(oss.str());
        }

        // Seed for update operations
        unsigned int current_seed = seed + 1000;

        // ====================================================
        // STEP 5: Main Iteration Loop
        // ====================================================
        int it = 0;
        for (it = 1; it <= CFG_TMAX; ++it)
        {

            // --- Update all penguins via ASIC_Update ---
            if (NUM_ASIC == 1)
            {
                // Sequential mode: preserve exact seed chain
                for (int i = 0; i < N; ++i)
                {
                    if (i == best_idx)
                        continue; // Elitism

                    UpdateReq ureq;
                    ureq.penguin_idx = i;
                    ureq.D = D;
                    std::memcpy(ureq.x_i, population[i], sizeof(double) * D);
                    std::memcpy(ureq.x_best, best_x, sizeof(double) * D);
                    std::memcpy(ureq.LB, LB, sizeof(double) * D);
                    std::memcpy(ureq.UB, UB, sizeof(double) * D);
                    ureq.mu = mu;
                    ureq.m = m;
                    ureq.a = CFG_A;
                    ureq.b = CFG_B;
                    ureq.tiny = CFG_TINY;
                    ureq.k_pairs = CFG_K;
                    ureq.seed = current_seed;

                    update_req_out[0].write(ureq);            // CPU -> ASIC
                    UpdateRes ures = update_res_in[0].read(); // ASIC -> CPU

                    std::memcpy(population[ures.penguin_idx],
                                ures.x_new, sizeof(double) * D);
                    current_seed = ures.seed_out;
                }
            }
            else
            {
                // Parallel mode: round-robin to NUM_ASIC units
                int count_per_asic[NUM_ASIC];
                for (int a = 0; a < NUM_ASIC; ++a)
                    count_per_asic[a] = 0;

                int assign = 0;
                for (int i = 0; i < N; ++i)
                {
                    if (i == best_idx)
                        continue;

                    int target = assign % NUM_ASIC;

                    UpdateReq ureq;
                    ureq.penguin_idx = i;
                    ureq.D = D;
                    std::memcpy(ureq.x_i, population[i], sizeof(double) * D);
                    std::memcpy(ureq.x_best, best_x, sizeof(double) * D);
                    std::memcpy(ureq.LB, LB, sizeof(double) * D);
                    std::memcpy(ureq.UB, UB, sizeof(double) * D);
                    ureq.mu = mu;
                    ureq.m = m;
                    ureq.a = CFG_A;
                    ureq.b = CFG_B;
                    ureq.tiny = CFG_TINY;
                    ureq.k_pairs = CFG_K;
                    ureq.seed = current_seed + i * 9973; // Per-penguin seed

                    update_req_out[target].write(ureq);
                    count_per_asic[target]++;
                    assign++;
                }

                // Collect all results
                for (int a = 0; a < NUM_ASIC; ++a)
                {
                    for (int c = 0; c < count_per_asic[a]; ++c)
                    {
                        UpdateRes ures = update_res_in[a].read();
                        std::memcpy(population[ures.penguin_idx],
                                    ures.x_new, sizeof(double) * D);
                    }
                }

                current_seed += N * 10000; // Advance seed
            }

            // --- Re-evaluate fitness via ASIC_Fitness (parallel) ---
            {
                int fit_count[NUM_ASIC];
                for (int a = 0; a < NUM_ASIC; ++a)
                    fit_count[a] = 0;

                for (int i = 0; i < N; ++i)
                {
                    int target = i % NUM_ASIC;

                    FitnessReq freq;
                    freq.penguin_idx = i;
                    freq.D = D;
                    std::memcpy(freq.x, population[i], sizeof(double) * D);
                    std::strncpy(freq.func_name, problem_name.c_str(), sizeof(freq.func_name) - 1);
                    freq.func_name[sizeof(freq.func_name) - 1] = '\0';

                    fitness_req_out[target].write(freq);
                    fit_count[target]++;
                }

                for (int a = 0; a < NUM_ASIC; ++a)
                {
                    for (int c = 0; c < fit_count[a]; ++c)
                    {
                        FitnessRes fres = fitness_res_in[a].read();
                        fitness[fres.penguin_idx] = fres.fitness;
                    }
                }
            }

            // --- Update best ---
            int curr_best_idx = 0;
            double curr_best_f = fitness[0];
            if (MAX_FITNES)
            {
                for (int i = 1; i < N; ++i)
                {
                    if (fitness[i] > curr_best_f)
                    {
                        curr_best_f = fitness[i];
                        curr_best_idx = i;
                    }
                }
            }
            else
            {
                for (int i = 1; i < N; ++i)
                {
                    if (fitness[i] < curr_best_f)
                    {
                        curr_best_f = fitness[i];
                        curr_best_idx = i;
                    }
                }
            }

            best_idx = curr_best_idx;
            best_f = curr_best_f;
            std::memcpy(best_x, population[curr_best_idx], sizeof(double) * D);

            // --- Log iteration ---
            logger.log(epc_iter_line(it, CFG_TMAX, best_f, mu, m));

            // --- Check convergence ---
            if (best_f <= CFG_EPSILON)
            {
                break;
            }

            // --- Decay parameters ---
            mu *= CFG_MU_DECAY;
            m *= CFG_M_DECAY;
        }

        // Record result
        all_best_fs.push_back(best_f);
        logger.close();
    }

    auto t_total_end = std::chrono::high_resolution_clock::now();
    total_time = std::chrono::duration<double>(t_total_end - t_total_start).count();

    // ========================================================
    // Print Benchmark Summary
    // ========================================================
    SimpleLogger summary_logger(LOG_PATH);
    summary_logger.log("");
    summary_logger.log(epc_summary_banner());
    summary_logger.log(summary_line(problem_name, D, all_best_fs, total_time));
    summary_logger.close();

    // ========================================================
    // End simulation
    // ========================================================
    sc_stop();
}
