#include "asic_init.h"
#include "epc_kernels.h"

// ============================================================
// ASIC_Init::process() - Population Initialization Thread
// ============================================================
// Infinite loop: waits for InitReq, generates random population,
// sends InitRes back to CPU.
// ============================================================

void ASIC_Init::process() {
    while (true) {
        // Blocking read: wait for CPU request
        InitReq req = req_in.read();

        // Compute: generate N x D random population
        InitRes res;
        kernel_init_population(res.population, req.N, req.D,
                               req.LB, req.UB, req.seed);

        // Blocking write: send result to CPU
        res_out.write(res);
    }
}
