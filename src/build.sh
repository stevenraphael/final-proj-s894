#!/bin/bash

# Output build directory path.
CTR_BUILD_DIR=/build

echo "Building the project..."

# ----------------------------------------------------------------------------------
# ------------------------ PUT YOUR BULDING COMMAND(s) HERE ------------------------
# ----------------------------------------------------------------------------------
# ----- This sctipt is executed inside the development container:
# -----     * the current workdir contains all files from your src/
# -----     * all output files (e.g. generated binaries, test inputs, etc.) must be places into $CTR_BUILD_DIR
# ----------------------------------------------------------------------------------
# Build code.
cp test.py ${CTR_BUILD_DIR}/test.py
nvcc -arch=sm_80 -O3 kmeans.cu -o ${CTR_BUILD_DIR}/kmeans
git clone https://github.com/krulis-martin/cuda-kmeans.git ${CTR_BUILD_DIR}/krulis
git clone https://github.com/krulis-martin/bpplib.git ${CTR_BUILD_DIR}/bpplib

sed -i 's/-arch=sm_52/-arch=sm_80/g' ${CTR_BUILD_DIR}/krulis/experimental/k-means/k-means/Makefile
sed -i 's| \.\./\.\./bpplib/include| \.\./\.\./\.\./\.\./bpplib/include|g' ${CTR_BUILD_DIR}/krulis/experimental/k-means/k-means/Makefile
cd ${CTR_BUILD_DIR}/krulis/experimental/k-means/k-means/

sed -i 's|^LIBS=.*|LIBS=cudart cublas curand|' Makefile

# Update the LDFLAGS line to ensure it uses the correct library directory
sed -i 's|^LDFLAGS=.*|LDFLAGS=$(addprefix -L,$(LIBDIRS)) $(addprefix -l,$(LIBS))|' Makefile

# Ensure LIBDIRS is correct, pointing to /usr/local/cuda/lib64
sed -i 's|^LIBDIRS=.*|LIBDIRS=/usr/local/cuda/lib64|' Makefile

sed -i 's|^\%.obj: \%.cpp.*|%.obj: %.cpp $(HEADERS)\n\t@echo Compiling CPP file $< ...\n\t@$(CPP) $(CFLAGS) $(addprefix -I,$(INCLUDE)) -L$(LIBDIRS) -lcudart -c $< -o $@|' Makefile
# apt update
# apt install cmake -y

# apt-get install python3-pip -y

# export CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda
# cd ${CTR_BUILD_DIR}
# mkdir dependencies

# pip install --target=dependencies\
#     --extra-index-url=https://pypi.nvidia.com \
#     "cuml-cu12==24.10.*"

# ls /usr/local/cuda/lib64
make


