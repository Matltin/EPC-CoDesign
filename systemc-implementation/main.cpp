#include <systemc.h>
#include <iostream>
#include "cpu.cpp"   // فایل کد CPU
#include "asic.cpp"  // فایل کد ASIC

int sc_main(int argc, char* argv[]) {
    sc_signal<int> cpu_to_asic[5];  // سیگنال ارتباطی از CPU به ۵ ASIC
    sc_signal<int> asic_to_cpu[5];  // سیگنال ارتباطی از ۵ ASIC به CPU

    // ایجاد یک نمونه از CPU
    CPU cpu;

    // ایجاد ۵ نمونه از ASIC
    ASIC asic1("ASIC1"), asic2("ASIC2"), asic3("ASIC3"), asic4("ASIC4"), asic5("ASIC5");

    // اتصال سیگنال‌ها به یکدیگر
    for (int i = 0; i < 5; ++i) {
        cpu.data_out[i](cpu_to_asic[i]);   // CPU به ASIC داده می‌دهد
        cpu.data_in[i](asic_to_cpu[i]);    // CPU نتیجه را از ASIC می‌گیرد
        // اتصال ASICها به CPU
        switch(i) {
            case 0: asic1.data_in(cpu_to_asic[i]); asic1.data_out(asic_to_cpu[i]); break;
            case 1: asic2.data_in(cpu_to_asic[i]); asic2.data_out(asic_to_cpu[i]); break;
            case 2: asic3.data_in(cpu_to_asic[i]); asic3.data_out(asic_to_cpu[i]); break;
            case 3: asic4.data_in(cpu_to_asic[i]); asic4.data_out(asic_to_cpu[i]); break;
            case 4: asic5.data_in(cpu_to_asic[i]); asic5.data_out(asic_to_cpu[i]); break;
        }
    }

    // شروع شبیه‌سازی
    sc_start(100, SC_NS);  // مدت زمان شبیه‌سازی، به اندازه کافی برای پردازش

    return 0;
}
