#include <systemc.h>
#include <cmath>

// ASIC: بخش سخت‌افزاری (SystemC)
SC_MODULE(ASIC) {
    sc_in<int> data_in;   // ورودی داده‌ها از CPU
    sc_out<int> data_out; // خروجی داده‌ها به CPU

    void process() {
        int input = data_in.read(); // دریافت ورودی از CPU

        // اعمال عملیات‌ها روی داده
        int result = input * 2;     // دو برابر کردن
        result += 4;                // اضافه کردن ۴
        result = sqrt(result);      // گرفتن جذر

        data_out.write(result);     // ارسال نتیجه به CPU
    }

    SC_CTOR(ASIC) {
        SC_METHOD(process);  // استفاده از SC_METHOD برای پردازش داده‌ها
        sensitive << data_in;  // حساس به تغییرات در data_in
    }
};
