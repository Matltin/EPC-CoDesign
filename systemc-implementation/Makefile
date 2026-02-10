# ---------------------------
#   SystemC EPC Makefile
# ---------------------------

SYSTEMC_HOME ?= /usr/local/systemc

CXX = g++
CXXFLAGS = -std=c++17 -O2 -DSC_INCLUDE_DYNAMIC_PROCESSES \
           -I$(SYSTEMC_HOME)/include -Iinclude
LDFLAGS = -L$(SYSTEMC_HOME)/lib-linux64 -lsystemc

SRCS = src/main.cpp \
       src/cpu_module.cpp \
       src/asic_init.cpp \
       src/asic_fitness.cpp \
       src/asic_update.cpp \
       src/hw_top.cpp

OBJS = $(SRCS:src/%.cpp=build/%.o)
HDRS = $(wildcard include/*.h)

TARGET = build/epc_sim

all: $(TARGET)

build/%.o: src/%.cpp $(HDRS) | build_dir
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(TARGET): $(OBJS)
	$(CXX) $^ -o $@ $(LDFLAGS)

build_dir:
	@mkdir -p build

run: $(TARGET)
	LD_LIBRARY_PATH=$(SYSTEMC_HOME)/lib-linux64:$$LD_LIBRARY_PATH ./$(TARGET)

clean:
	rm -rf build/* out.txt

.PHONY: all run clean build_dir
