# استفاده از تصویر پایه Ubuntu
FROM ubuntu:22.04

# نصب وابستگی‌ها
RUN apt update && apt install -y \
    build-essential \
    g++ \
    make \
    wget \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# تنظیم مسیر کاری
WORKDIR /opt

# دانلود و نصب SystemC
RUN wget https://github.com/accellera-official/systemc/archive/refs/tags/3.0.2.tar.gz && \
    tar -xvf 3.0.2.tar.gz && \
    cd systemc-3.0.2 && \
    mkdir build && cd build && \
    ../configure --prefix=/usr/local/systemc && \
    make -j$(nproc) && \
    make install

# تنظیم متغیرهای محیطی
ENV SYSTEMC_HOME=/usr/local/systemc
ENV LD_LIBRARY_PATH=$SYSTEMC_HOME/lib-linux64:$LD_LIBRARY_PATH

# تنظیم مسیر کاری برای پروژه‌ها
WORKDIR /workspace

# نمایش اطلاعات محیط
CMD ["bash"]
