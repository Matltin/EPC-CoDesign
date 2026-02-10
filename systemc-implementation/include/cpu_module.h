#ifndef CPU_MODULE_H
#define CPU_MODULE_H

#include <systemc.h>
#include "epc_defs.h"

// ============================================================
// CPU Module: Main Orchestration and Control
// ============================================================
// SC_THREAD(run) contains the full optimization flow:
//   - Loop over benchmark cases (Sphere, D=10)
//   - Loop over seeds (3 runs)
//   - EPC optimization: init -> eval -> iterate -> summary
//   - All ASIC communication via sc_fifo blocking read/write
// ============================================================

SC_MODULE(CPU_Module) {
    // Ports to ASIC_Init
    sc_fifo_out<InitReq>    init_req_out;
    sc_fifo_in<InitRes>     init_res_in;

    // Ports to ASIC_Fitness
    sc_fifo_out<FitnessReq> fitness_req_out;
    sc_fifo_in<FitnessRes>  fitness_res_in;

    // Ports to NUM_ASIC ASIC_Update instances (using sc_vector)
    sc_vector<sc_fifo_out<UpdateReq>> update_req_out;
    sc_vector<sc_fifo_in<UpdateRes>>  update_res_in;

    // Main thread
    void run();

    CPU_Module(sc_module_name name)
        : sc_module(name)
        , update_req_out("update_req_out", NUM_ASIC)
        , update_res_in("update_res_in", NUM_ASIC)
    {
        SC_THREAD(run);
    }
};

#endif // CPU_MODULE_H
