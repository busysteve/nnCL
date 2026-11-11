cmake_minimum_required(VERSION 3.24)

project(nnCL LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCL REQUIRED)

# auto keeps the first usable GPU. The other values require that vendor's device.
# intel uses an Intel GPU, or the Intel CPU device when no Intel GPU is listed.
set(NNCL_OPENCL "auto" CACHE STRING "OpenCL vendor: auto, radeon, intel, apple, or nvidia")
set_property(CACHE NNCL_OPENCL PROPERTY STRINGS auto radeon intel apple nvidia)
string(TOLOWER "${NNCL_OPENCL}" NNCL_OPENCL_LOWER)
if(NOT NNCL_OPENCL_LOWER MATCHES "^(auto|radeon|intel|apple|nvidia)$")
  message(FATAL_ERROR "NNCL_OPENCL must be auto, radeon, intel, apple, or nvidia")
endif()
message(STATUS "nnCL OpenCL vendor: ${NNCL_OPENCL_LOWER}")

add_library(nn STATIC
  src/context.cpp
  src/ocl.cpp
  src/tensor.cpp
  src/index_tensor.cpp
  src/init.cpp
  src/config.cpp
  src/linear.cpp
  src/embedding.cpp
  src/relu.cpp
  src/gelu.cpp
  src/swiglu.cpp
  src/layer_norm.cpp
  src/rms_norm.cpp
  src/dropout.cpp
  src/multi_head_attention.cpp
  src/transformer_block.cpp
  src/transformer.cpp
  src/rnn.cpp
  src/last_step.cpp
  src/conv2d.cpp
  src/max_pool2d.cpp
  src/batch_norm2d.cpp
  src/flatten.cpp
  src/global_avg_pool2d.cpp
  src/residual_block.cpp
  src/network.cpp
  src/optim.cpp
  src/loss.cpp
  src/checkpoint.cpp
  src/builders.cpp
)

target_include_directories(nn
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

target_link_libraries(nn PUBLIC OpenCL::OpenCL)

if(APPLE)
  set(NNCL_CL_TARGET 120)
else()
  set(NNCL_CL_TARGET 300)
endif()

target_compile_definitions(nn PRIVATE
  CL_TARGET_OPENCL_VERSION=${NNCL_CL_TARGET}
  NNCL_KERNELS_FILE="${CMAKE_CURRENT_SOURCE_DIR}/src/kernels.cl")
if(NOT NNCL_OPENCL_LOWER STREQUAL "auto")
  string(TOUPPER "${NNCL_OPENCL_LOWER}" NNCL_OPENCL_UPPER)
  target_compile_definitions(nn PRIVATE NNCL_OPENCL_${NNCL_OPENCL_UPPER}=1)
endif()

set_target_properties(nn PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_executable(nn_tests
  tests/test_main.cpp
  tests/test_shapes.cpp
  tests/test_gradients.cpp
  tests/test_checkpoint.cpp
  tests/test_mnist.cpp
  tests/mnist.cpp
)

target_link_libraries(nn_tests PRIVATE nn)
target_include_directories(nn_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)

enable_testing()
add_test(NAME shapes COMMAND nn_tests shapes)
add_test(NAME grads COMMAND nn_tests grads)
add_test(NAME ckpt COMMAND nn_tests ckpt)
add_test(NAME mnist COMMAND nn_tests mnist)
set_tests_properties(shapes grads ckpt PROPERTIES TIMEOUT 900)
set_tests_properties(mnist PROPERTIES TIMEOUT 7200)
