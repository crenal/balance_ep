cmake_minimum_required(VERSION 3.19)

project(balance_ep LANGUAGES CUDA CXX)

if(NOT CMAKE_CXX_COMPILER)
  set(CMAKE_CXX_COMPILER "/usr/bin/g++" CACHE FILEPATH "C++ compiler" FORCE)
endif()

if(NOT CMAKE_CUDA_COMPILER)
  set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc" CACHE FILEPATH "CUDA compiler" FORCE)
endif()

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
  set(CMAKE_CUDA_ARCHITECTURES "90" CACHE STRING "CUDA architectures" FORCE)
endif()

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(NVSHMEM_ROOT "/workspace/nvshmem_install" CACHE PATH "Path to NVSHMEM install")
list(APPEND CMAKE_PREFIX_PATH "${NVSHMEM_ROOT}")

set(MPI_HOME "/usr/local/openmpi" CACHE PATH "Path to MPI install")
set(MPI_CXX_COMPILER "${MPI_HOME}/bin/mpicxx" CACHE FILEPATH "MPI C++ compiler" FORCE)
list(APPEND CMAKE_PREFIX_PATH "${MPI_HOME}")

find_package(CUDAToolkit REQUIRED)
find_package(MPI REQUIRED)
find_package(NVSHMEM REQUIRED CONFIG HINTS "${NVSHMEM_ROOT}/lib/cmake/nvshmem")

add_library(balance_ep_headers INTERFACE)
target_include_directories(balance_ep_headers INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_library(balance_ep_core
  src/dispatch.cu
  src/dispatch_fast.cu
)
target_link_libraries(balance_ep_core PUBLIC
  balance_ep_headers
  MPI::MPI_CXX
  CUDA::cudart
  nvshmem::nvshmem_host
  nvshmem::nvshmem_device
)
set_target_properties(balance_ep_core PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  BUILD_RPATH "${NVSHMEM_ROOT}/lib"
  INSTALL_RPATH "${NVSHMEM_ROOT}/lib"
)

add_executable(test_dispatch
  test/test_dispatch.cu
  test/test_dispatch_buffers.cu
  test/test_dispatch_check.cu
  test/test_dispatch_env.cu
  test/test_dispatch_inputs.cu
)
target_link_libraries(test_dispatch PRIVATE
  balance_ep_core
)
set_target_properties(test_dispatch PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  BUILD_RPATH "${NVSHMEM_ROOT}/lib"
  INSTALL_RPATH "${NVSHMEM_ROOT}/lib"
)
