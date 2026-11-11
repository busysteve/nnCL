# nnCL

OpenCL implementation of the [busysteve/nn](https://github.com/busysteve/nn) neural-net API. The static library target is still named `nn`, and the public headers live in `nn/`. [busysteve/llm](https://github.com/busysteve/llm) links against it by pointing `NN_SOURCE` at this tree.

Dense float32, ranks 1–4, NCHW convolutions. One in-order command queue per `nn::Context`. A device is used only when its max work-group size is at least 256, because the kernels launch 256-wide groups. Do not set `RUSTICL_ENABLE`; on this machine `RUSTICL_ENABLE=all` hides the GPU.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DNNCL_OPENCL=nvidia
cmake --build build
./build/nn_tests shapes
./build/nn_tests grads
./build/nn_tests ckpt
```

`NNCL_OPENCL` chooses the device. The kernel build then uses `-cl-std=CL1.2` or `-cl-std=CL2.0` from that device's OpenCL C version. A device that reports 2.0 or 3.0 is built as OpenCL C 2.0. A 1.2 device, including current NVIDIA and Apple GPUs, is built as OpenCL C 1.2.

| Value | Device |
|---|---|
| `auto` | first GPU whose context can be created. This is the default |
| `radeon` | AMD or Radeon GPU |
| `intel` | Intel GPU, or the Intel CPU device when no Intel GPU is listed |
| `apple` | Apple GPU |
| `nvidia` | NVIDIA GPU |

`auto` does not use a CPU device. A requested vendor that is not installed fails with `OpenCL device`.

`Context()` uses device 0 from that list. `Context(int device)` uses that index. `device_count()` and `device_name(int)` report the same list: GPUs with a large enough work-group, or the Intel CPU devices when `intel` found no GPU. An index outside the list throws `device`.

`detect_devices()` returns that list. `select_device("auto")` and `Context("auto")` use the first entry. `select_device("1")` and `Context("1")` select that index by hand. This `auto` is not the `NNCL_OPENCL` vendor filter. A blank choice, a word other than `auto`, or an index past the list throws `device`.

An Apple host includes `<OpenCL/cl.h>` and creates the queue with `clCreateCommandQueue`. Other hosts use the OpenCL 3.0 headers and `clCreateCommandQueueWithProperties`.

Kernels are read at runtime from `src/kernels.cl` (the path compiled into the library). An `nn::Error` message looks like `call: CL_STATUS_NAME`.

`IndexTensor::to_host()` returns the index buffer. The CUDA nn library reads `data()` with `cudaMemcpy` instead. llm's data test should call `to_host()`.

To build llm against nnCL, configure llm with `-DNN_SOURCE=/path/to/nnCL`. llm itself must be C++ only (`project(llm LANGUAGES CXX)`), because this backend does not use CUDA.
