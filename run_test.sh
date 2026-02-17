#!/usr/bin/env bash
NVSHMEM_BOOTSTRAP=mpi NVSHMEM_BOOTSTRAP_LIBRARY=/workspace/nvshmem_install/lib/nvshmem_bootstrap_mpi.so /usr/local/openmpi/bin/mpirun --allow-run-as-root -np 4 ./build/test_dispatch
