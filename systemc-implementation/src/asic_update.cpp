#include "asic_update.h"
#include "epc_kernels.h"

// ============================================================
// ASIC_Update::process() - Penguin Update Thread
// ============================================================
// Infinite loop: waits for UpdateReq, performs full Method B_R
// update (distance, Q, sampling, spiral, mutation, clip),
// sends UpdateRes with new position and evolved seed.
// NUM_ASIC instances run concurrently for parallel throughput.
// ============================================================

void ASIC_Update::process() {
    while (true) {
        // Blocking read: wait for CPU request
        UpdateReq req = req_in.read();

        // Compute: full penguin update
        UpdateRes res;
        res.penguin_idx = req.penguin_idx;

        unsigned int seed = req.seed;
        kernel_update_penguin(
            res.x_new,
            req.x_i,
            req.x_best,
            req.mu, req.m, req.a, req.b,
            req.LB, req.UB,
            req.D, req.tiny, req.k_pairs,
            seed
        );
        res.seed_out = seed;

        // Blocking write: send result to CPU
        res_out.write(res);
    }
}
