#ifndef HW_TOP_H
#define HW_TOP_H

#include <systemc.h>
#include "epc_defs.h"
#include "cpu_module.h"
#include "asic_init.h"
#include "asic_fitness.h"
#include "asic_update.h"

// ============================================================
// HW_TOP: Top-Level Module (CPU + ASIC Wiring)
// ============================================================
// Instantiates:
//   - 1 CPU_Module
//   - 1 ASIC_Init
//   - 1 ASIC_Fitness
//   - NUM_ASIC ASIC_Update instances
// Connects all via sc_fifo channels
// ============================================================

SC_MODULE(HW_TOP) {
    // Sub-modules
    CPU_Module*    cpu;
    ASIC_Init*     asic_init;
    ASIC_Fitness*  asic_fitness[NUM_ASIC];
    ASIC_Update*   asic_update[NUM_ASIC];

    // FIFOs
    sc_fifo<InitReq>*    fifo_init_req;
    sc_fifo<InitRes>*    fifo_init_res;
    sc_fifo<FitnessReq>* fifo_fit_req[NUM_ASIC];
    sc_fifo<FitnessRes>* fifo_fit_res[NUM_ASIC];
    sc_fifo<UpdateReq>*  fifo_upd_req[NUM_ASIC];
    sc_fifo<UpdateRes>*  fifo_upd_res[NUM_ASIC];

    SC_CTOR(HW_TOP);
    ~HW_TOP();
};

#endif // HW_TOP_H
