#include <iostream>
#include <systemc.h>

// CPU: بخش نرم‌افزاری (C++)
class CPU {
public:
    sc_out<int> data_out[5];  // خروجی داده‌ها به ۵ ASIC
    sc_in<int> data_in[5];    // ورودی داده‌ها از ۵ ASIC

    void process() {
        // پردازش ورودی‌های اولیه
        for (int i = 0; i < 5; ++i) {
            // ارسال داده به ASIC
            data_out[i].write(i + 1);  // داده‌هایی از ۱ تا ۵ به هر ASIC ارسال می‌کنیم

            // شبیه‌سازی ارسال داده و گرفتن نتیجه از ASIC
            wait(10, SC_NS);  // منتظر بودن برای پردازش در ASIC

            // دریافت نتیجه از ASIC
            int result = data_in[i].read();
            std::cout << "Result for input to ASIC " << i + 1 << ": " << result << std::endl;
        }
    }

    SC_CTOR(CPU) {
        SC_THREAD(process);  // استفاده از SC_THREAD برای متد process
        for (int i = 0; i < 5; ++i) {
            sensitive << data_in[i];  // حساس به تغییرات در داده‌های ورودی
        }
    }
};
