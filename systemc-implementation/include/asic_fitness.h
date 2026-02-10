#ifndef ASIC_FITNESS_H
#define ASIC_FITNESS_H

#include <systemc.h>
#include "epc_defs.h"

// ============================================================
// ASIC Module: Fitness Evaluation (Sphere Function)
// ============================================================
// Evaluates f(x) = sum(x_i^2) for N individuals
// Hardware: N parallel MAC units + tree reduction
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
