#ifndef ASIC_UPDATE_H
#define ASIC_UPDATE_H

#include <systemc.h>
#include "epc_defs.h"

// ============================================================
// ASIC Module: Penguin Update (Method B_R)
// ============================================================
// Full update pipeline per penguin:
//   distance -> Q -> sample K pairs -> spiral update -> mutation -> clip
// Hardware: CORDIC + exp/log + sqrt + RNG units
// NUM_ASIC instances run in parallel for scalable throughput
// ============================================================

SC_MODULE(ASIC_Update) {
    sc_fifo_in<UpdateReq>   req_in;
    sc_fifo_out<UpdateRes>  res_out;

    void process();

    SC_CTOR(ASIC_Update) {
        SC_THREAD(process);
    }
};

#endif // ASIC_UPDATE_H
