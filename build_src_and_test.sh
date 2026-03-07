#!/usr/bin/env bash
set -euo pipefail
cd /workspace/balance_ep
rm -rf build
mkdir -p build
cd build
cmake -DCMAKE_CUDA_ARCHITECTURES=90 -DCMAKE_CXX_COMPILER=/usr/bin/g++ -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DMPI_HOME=/usr/local/openmpi -DMPI_CXX_COMPILER=/usr/local/openmpi/bin/mpicxx ..
cmake --build . -j
