cmake_minimum_required(VERSION 3.20)
project(welvet VERSION 0.1.31 LANGUAGES CXX)

# One tree → many OS: Linux, macOS, Windows, Android, iOS (toolchains under cmake/).
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(WELVET_BUILD_SHARED "Build shared library (C ABI / NAC)" ON)
option(WELVET_BUILD_TOOLS "Build CLI tools for w2a go_v_cpp" ON)
option(WELVET_BUILD_EXAMPLES "Build standalone examples" ON)
option(WELVET_ENABLE_CUDA "gap? NVIDIA CUDA backend" OFF)
option(WELVET_ENABLE_ROCM "gap? AMD ROCm/HIP backend" OFF)
option(WELVET_ENABLE_METAL "gap? Apple Metal backend" OFF)
option(WELVET_ENABLE_ONEAPI "gap? Intel oneAPI/Level Zero backend" OFF)
option(WELVET_ENABLE_QNN "gap? Qualcomm QNN/NPU backend" OFF)
option(WELVET_ENABLE_TPU "gap? Google TPU backend" OFF)
option(WELVET_ENABLE_AVX2 "Enable AVX2/FMA/F16C for imported SIMD kernels" ON)

if(APPLE)
  set(WELVET_PLATFORM_APPLE ON)
elseif(ANDROID)
  set(WELVET_PLATFORM_ANDROID ON)
elseif(WIN32)
  set(WELVET_PLATFORM_WINDOWS ON)
elseif(UNIX)
  set(WELVET_PLATFORM_LINUX ON)
endif()

include(GNUInstallDirs)

set(WELVET_CORE_SOURCES
  src/core/status.cpp
  src/core/dtype.cpp
  src/core/activate.cpp
  src/core/tensor.cpp
  src/core/float_codecs.cpp
  src/weights/store.cpp
  src/weights/native_codec.cpp
  src/layers/dense/dense.cpp
  src/layers/dense/simd_bridge.cpp
  src/layers/mha/mha.cpp
  src/layers/swiglu/swiglu.cpp
  src/layers/rmsnorm/rmsnorm.cpp
  src/layers/layernorm/layernorm.cpp
  src/layers/cnn1/cnn1.cpp
  src/layers/cnn2/cnn2.cpp
  src/layers/cnn3/cnn3.cpp
  src/layers/convt1/convt1.cpp
  src/layers/convt2/convt2.cpp
  src/layers/convt3/convt3.cpp
  src/layers/rnn/rnn.cpp
  src/layers/lstm/lstm.cpp
  src/layers/embedding/embedding.cpp
  src/layers/softmax/softmax.cpp
  src/layers/kmeans/kmeans.cpp
  src/layers/sequential/sequential.cpp
  src/layers/residual/residual.cpp
  src/layers/mamba/mamba.cpp
  src/layers/metacognition/metacognition.cpp
  src/layers/gdn/gdn.cpp
  src/layers/parallel/parallel.cpp
  src/architecture/grid.cpp
  src/runtime/forward.cpp
  src/runtime/backward.cpp
  src/runtime/training.cpp
  src/backends/registry.cpp
  src/backends/cpu.cpp
  src/backends/nvidia.cpp
  src/backends/amd.cpp
  src/backends/intel.cpp
  src/backends/apple.cpp
  src/backends/qualcomm.cpp
  src/backends/google_tpu.cpp
  src/gaps/registry.cpp
  src/interop/c_abi.cpp
)

add_library(welvet_obj OBJECT ${WELVET_CORE_SOURCES})
set_target_properties(welvet_obj PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(welvet_obj
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_compile_definitions(welvet_obj
  PRIVATE
    $<$<BOOL:${WELVET_PLATFORM_LINUX}>:WELVET_OS_LINUX=1>
    $<$<BOOL:${WELVET_PLATFORM_APPLE}>:WELVET_OS_APPLE=1>
    $<$<BOOL:${WELVET_PLATFORM_WINDOWS}>:WELVET_OS_WINDOWS=1>
    $<$<BOOL:${WELVET_PLATFORM_ANDROID}>:WELVET_OS_ANDROID=1>
    $<$<BOOL:${IOS}>:WELVET_OS_IOS=1>
)
if(WELVET_ENABLE_AVX2 AND CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64")
  if(MSVC)
    target_compile_options(welvet_obj PRIVATE /arch:AVX2)
  else()
    target_compile_options(welvet_obj PRIVATE -mavx2 -mfma -mf16c -ffp-contract=off)
  endif()
endif()

if(WELVET_BUILD_SHARED)
  add_library(welvet SHARED $<TARGET_OBJECTS:welvet_obj>)
else()
  add_library(welvet STATIC $<TARGET_OBJECTS:welvet_obj>)
endif()
add_library(welvet::welvet ALIAS welvet)
target_include_directories(welvet
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
set_target_properties(welvet PROPERTIES
  OUTPUT_NAME welvet
  VERSION ${PROJECT_VERSION}
  SOVERSION 0
  POSITION_INDEPENDENT_CODE ON
)
if(WIN32)
  set_target_properties(welvet PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()

if(WELVET_BUILD_TOOLS)
  add_executable(welvet_dense_fwd tools/dense_fwd/main.cpp)
  target_link_libraries(welvet_dense_fwd PRIVATE welvet)
endif()

if(WELVET_BUILD_EXAMPLES)
  add_executable(example_dense_forward examples/dense_forward/main.cpp)
  target_link_libraries(example_dense_forward PRIVATE welvet)
endif()

install(TARGETS welvet
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
if(WELVET_BUILD_TOOLS)
  install(TARGETS welvet_dense_fwd RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()

message(STATUS "welvet.cpp platform: ${CMAKE_SYSTEM_NAME}/${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "accel CUDA=${WELVET_ENABLE_CUDA} ROCm=${WELVET_ENABLE_ROCM} Metal=${WELVET_ENABLE_METAL} oneAPI=${WELVET_ENABLE_ONEAPI} QNN=${WELVET_ENABLE_QNN} TPU=${WELVET_ENABLE_TPU}")
