#ifndef ASIC_INIT_H
#define ASIC_INIT_H

#include <systemc.h>
#include "epc_defs.h"

// ============================================================
// ASIC Module: Population Initialization
// ============================================================
// Generates N x D random values in [LB, UB]
// Hardware: N parallel RNG units, pipeline depth O(log D)
// ============================================================

SC_MODULE(ASIC_Init) {
    sc_fifo_in<InitReq>   req_in;
    sc_fifo_out<InitRes>  res_out;

    void process();

    SC_CTOR(ASIC_Init) {
        SC_THREAD(process);
    }
};

#endif // ASIC_INIT_H
