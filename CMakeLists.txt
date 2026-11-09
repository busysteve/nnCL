cmake_minimum_required(VERSION 3.24)

project(nnCL LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCL REQUIRED)

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

target_compile_definitions(nn PRIVATE
  CL_TARGET_OPENCL_VERSION=300
  NNCL_KERNELS_FILE="${CMAKE_CURRENT_SOURCE_DIR}/src/kernels.cl")

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
