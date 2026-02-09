# مسیرهای فایل‌های منبع و اجرایی
SRC = main.cpp cpu.cpp asic.cpp
OBJ = $(SRC:.cpp=.o)
EXEC = main

# تنظیمات کامپایلر
CXX = g++
CXXFLAGS = -std=c++11 -I/usr/include/systemc

# تنظیمات لینک‌دهی با SystemC
LDFLAGS = -L/usr/lib/systemc -lsystemc

# ساخت پروژه
all: $(EXEC)

$(EXEC): $(OBJ)
	$(CXX) $(OBJ) -o $(EXEC) $(LDFLAGS)

# کامپایل فایل‌های .cpp به .o
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# حذف فایل‌های تولید شده
clean:
	rm -f $(OBJ) $(EXEC)
