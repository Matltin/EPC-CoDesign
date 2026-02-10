#include <systemc.h>
#include "hw_top.h"

// ============================================================
// sc_main: SystemC Simulation Entry Point
// ============================================================
// Creates the top-level HW_TOP module (CPU + ASIC wiring)
// and starts the event-driven simulation.
// The simulation runs until CPU_Module::run() calls sc_stop().
// ============================================================

int sc_main(int argc, char* argv[]) {
    (void)argc;
    (void)argv;

    HW_TOP top("top");

    sc_start();  // Run until sc_stop() is called by CPU

    return 0;
}
