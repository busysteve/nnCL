# nnCL

OpenCL implementation of the [busysteve/nnCUDA](https://github.com/busysteve/nnCUDA) neural-net API. The same checkpoint file loads in [nnROCm](https://github.com/busysteve/nnROCm) and [nnArc](https://github.com/busysteve/nnArc). The static library target is still named `nn`, and the public headers live in `nn/`. The shared contract is [docs/api.md](docs/api.md). This tree has no `USAGE.md`. The file map and the transformer ownership diagram are in [docs/code-structure.md](docs/code-structure.md). [busysteve/llm](https://github.com/busysteve/llm) links against it with `./build.sh opencl`, which points `NN_SOURCE` at this tree.

Parameters, gradients, optimizer state, and checkpoints are dense float32. Retained activations are bfloat16 unless `Context::set_activation_dtype(Float32)` selects float32. `nnArc` has no `dtype.hpp` and keeps activations in float32. Ranks are 1–4. Convolutions are NCHW. One in-order command queue per `nn::Context`. A device is used only when its max work-group size is at least 256, because the kernels launch 256-wide groups. Do not set `RUSTICL_ENABLE`; on this machine `RUSTICL_ENABLE=all` hides the GPU.

`Network::save` writes version 1 for SGD or no optimizer and version 2 for AdamW. `Transformer::save` writes versions 1 through 8. A classic block is version 1 with no optimizer and version 2 with one. A Qwen-style block is version 3 or 4. A bidirectional block or a segment table is version 5 or 6. A projection-bias block is version 7 or 8. The even versions store the optimizer. The trunk is causal unless the config sets bidirectional attention.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DNNCL_OPENCL=nvidia
cmake --build build
./build/nn_tests shapes
./build/nn_tests grads
./build/nn_tests ckpt
./build/nn_tests bf16
MNIST_DIR=/path/to/mnist ./build/nn_tests mnist
```

`mnist` reads `train-images-idx3-ubyte`, `train-labels-idx1-ubyte`, `t10k-images-idx3-ubyte`, and `t10k-labels-idx1-ubyte` from `MNIST_DIR`. This tree does not ship those files. `bf16` checks bfloat16 activations.

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

`IndexTensor::to_host()` returns the index buffer. A CUDA build of llm reads `data()` with `cudaMemcpy` when `LLM_NN_CUDA` is set. The OpenCL build of llm uses `to_host()`.

`./build.sh opencl` in the llm tree configures against a sibling `../nnCL`. A hand configure can pass `-DNN_SOURCE=/path/to/nnCL`. llm itself stays C++ only, because this backend does not use CUDA.
