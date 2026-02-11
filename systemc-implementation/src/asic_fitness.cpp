#include "asic_fitness.h"
#include "epc_kernels.h"

// ============================================================
// ASIC_Fitness::process() - Per-Penguin Fitness Evaluation
// ============================================================
// Infinite loop: waits for FitnessReq (single penguin),
// evaluates fitness sequentially over D dimensions,
// sends FitnessRes with penguin index and result.
// NUM_ASIC instances run concurrently for parallel throughput.
// ============================================================

void ASIC_Fitness::process() {
    while (true) {
        // Blocking read: wait for CPU request (one penguin)
        FitnessReq req = req_in.read();

        // Compute: evaluate fitness for this penguin over D dims
        FitnessRes res;
        res.penguin_idx = req.penguin_idx;

        if (std::string(req.func_name) == "Rosenbrock") {
            res.fitness = kernel_rosenbrock(req.x, req.D);
        } else {
            res.fitness = kernel_sphere(req.x, req.D);
        }

        // Blocking write: send result to CPU
        res_out.write(res);
    }
}
