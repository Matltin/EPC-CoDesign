#ifndef ASIC_FITNESS_H
#define ASIC_FITNESS_H

#include <systemc.h>
#include "epc_defs.h"

// ============================================================
// ASIC Module: Fitness Evaluation (per-penguin, sequential D)
// ============================================================
// Evaluates one penguin at a time, accumulating over D dimensions.
// NUM_ASIC instances run in parallel for scalable throughput.
// Supports Sphere and Rosenbrock via func_name dispatch.
// ============================================================

SC_MODULE(ASIC_Fitness) {
    sc_fifo_in<FitnessReq>   req_in;
    sc_fifo_out<FitnessRes>  res_out;

    void process();

    SC_CTOR(ASIC_Fitness) {
        SC_THREAD(process);
    }
};

#endif // ASIC_FITNESS_H
