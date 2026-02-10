#include "hw_top.h"
#include <string>

// ============================================================
// HW_TOP Constructor: Create and Wire All Modules
// ============================================================

HW_TOP::HW_TOP(sc_module_name name) : sc_module(name) {
    // --- Create FIFOs ---
    fifo_init_req = new sc_fifo<InitReq>("fifo_init_req", 1);
    fifo_init_res = new sc_fifo<InitRes>("fifo_init_res", 1);
    fifo_fit_req  = new sc_fifo<FitnessReq>("fifo_fit_req", 1);
    fifo_fit_res  = new sc_fifo<FitnessRes>("fifo_fit_res", 1);

    for (int i = 0; i < NUM_ASIC; ++i) {
        std::string rq = "fifo_upd_req_" + std::to_string(i);
        std::string rs = "fifo_upd_res_" + std::to_string(i);
        fifo_upd_req[i] = new sc_fifo<UpdateReq>(rq.c_str(), FIFO_DEPTH);
        fifo_upd_res[i] = new sc_fifo<UpdateRes>(rs.c_str(), FIFO_DEPTH);
    }

    // --- Create Modules ---
    cpu          = new CPU_Module("cpu");
    asic_init    = new ASIC_Init("asic_init");
    asic_fitness = new ASIC_Fitness("asic_fitness");

    for (int i = 0; i < NUM_ASIC; ++i) {
        std::string uname = "asic_update_" + std::to_string(i);
        asic_update[i] = new ASIC_Update(uname.c_str());
    }

    // --- Wire CPU <-> ASIC_Init ---
    cpu->init_req_out(*fifo_init_req);
    cpu->init_res_in(*fifo_init_res);
    asic_init->req_in(*fifo_init_req);
    asic_init->res_out(*fifo_init_res);

    // --- Wire CPU <-> ASIC_Fitness ---
    cpu->fitness_req_out(*fifo_fit_req);
    cpu->fitness_res_in(*fifo_fit_res);
    asic_fitness->req_in(*fifo_fit_req);
    asic_fitness->res_out(*fifo_fit_res);

    // --- Wire CPU <-> ASIC_Update[i] ---
    for (int i = 0; i < NUM_ASIC; ++i) {
        cpu->update_req_out[i](*fifo_upd_req[i]);
        cpu->update_res_in[i](*fifo_upd_res[i]);
        asic_update[i]->req_in(*fifo_upd_req[i]);
        asic_update[i]->res_out(*fifo_upd_res[i]);
    }
}

HW_TOP::~HW_TOP() {
    delete cpu;
    delete asic_init;
    delete asic_fitness;
    for (int i = 0; i < NUM_ASIC; ++i) {
        delete asic_update[i];
        delete fifo_upd_req[i];
        delete fifo_upd_res[i];
    }
    delete fifo_init_req;
    delete fifo_init_res;
    delete fifo_fit_req;
    delete fifo_fit_res;
}
