# nnCL

OpenCL implementation of the [busysteve/nn](https://github.com/busysteve/nn) neural-net API. The static library target is still named `nn`, and the public headers live in `nn/`. [busysteve/llm](https://github.com/busysteve/llm) links against it by pointing `NN_SOURCE` at this tree.

Dense float32, ranks 1–4, NCHW convolutions. One in-order command queue per `nn::Context`. The first OpenCL GPU whose context can be created is used. `CL_DEVICE_TYPE_GPU` skips non-GPU accelerators. Do not set `RUSTICL_ENABLE`; on this machine `RUSTICL_ENABLE=all` hides the GPU.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/nn_tests shapes
./build/nn_tests grads
./build/nn_tests ckpt
```

Kernels are read at runtime from `src/kernels.cl` (the path compiled into the library). An `nn::Error` message looks like `call: CL_STATUS_NAME`.

`IndexTensor::to_host()` returns the index buffer. The CUDA nn library reads `data()` with `cudaMemcpy` instead. llm's data test should call `to_host()`.

To build llm against nnCL, configure llm with `-DNN_SOURCE=/path/to/nnCL`. llm itself must be C++ only (`project(llm LANGUAGES CXX)`), because this backend does not use CUDA.
