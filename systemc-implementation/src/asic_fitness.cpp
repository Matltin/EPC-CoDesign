#include "asic_fitness.h"
#include "epc_kernels.h"

// ============================================================
// ASIC_Fitness::process() - Fitness Evaluation Thread
// ============================================================
// Infinite loop: waits for FitnessReq (N individuals),
// evaluates sphere function for each, sends FitnessRes.
// ============================================================

void ASIC_Fitness::process() {
    while (true) {
        // Blocking read: wait for CPU request
        FitnessReq req = req_in.read();

        // Compute: evaluate sphere for all N individuals
        FitnessRes res;
        kernel_evaluate_population(res.fitness, req.population,
                                   req.N, req.D);

        // Blocking write: send result to CPU
        res_out.write(res);
    }
}
