cmake_minimum_required(VERSION 3.20)
project(cifar10_server CXX)

set(CMAKE_CXX_STANDARD 17)

find_package(CUDAToolkit REQUIRED)

# 🔹 ONNX Runtime paths
set(ORT_ROOT ~/mlops/onnxruntime)
include_directories(
    ${ORT_ROOT}/include
    third_party
    ${CUDA_INCLUDE_DIRS}
)

link_directories(
    ${ORT_ROOT}/build/Linux/Release
)

add_executable(cifar10_server
    src/main.cpp
    src/ort_infer.cpp
    src/batch_infer_engine.cpp
)

target_link_libraries(cifar10_server
    $ENV{ORT_ROOT}/lib/libonnxruntime.so
    pthread
    CUDA::cudart
)

# Runtime library path
set_target_properties(cifar10_server PROPERTIES
    INSTALL_RPATH "${ORT_ROOT}/build/Linux/Release"
)
