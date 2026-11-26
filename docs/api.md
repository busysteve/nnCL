# API specification

This is the contract of the C++ API in namespace `nn`. Two libraries implement it.

| Library | Tree | Backend |
|---|---|---|
| `nn` | [busysteve/nnCUDA](https://github.com/busysteve/nnCUDA) | CUDA, cuBLAS, cuDNN |
| `nnCL` | [busysteve/nnCL](https://github.com/busysteve/nnCL) | OpenCL |

The public headers under `include/nn/` are the same in both trees. The only header that differs is `nn/context.hpp`, and the difference is comments. A checkpoint written by one library loads in the other. `src/checkpoint.cpp` is the same file in both trees.

`nn/USAGE.md` is a programmer's guide for the CUDA tree. This document is the contract. It includes the Qwen-style attention fields and checkpoint versions 1 through 8.

The CMake target is `nn` in both trees. Headers do not include CUDA, OpenCL, cuBLAS, or cuDNN headers. A program that links `libnn.a` by hand also links the backend libraries that archive was built against: `CUDA::cudart`, `CUDA::cublas`, and cuDNN for `nn`; the OpenCL loader for `nnCL`.

## 1. Program model

The language is C++17. Everything public is in namespace `nn`.

Create one `Context`. Pass that same `Context&` to every tensor, index buffer, layer, loss, and optimizer that must share work. Each context owns one in-order queue: a non-default CUDA stream, or one OpenCL command queue. Destroy tensors, layers, networks, and optimizers before the context.

`forward` and `backward` return `const Tensor&` into storage owned by that layer. The reference stays valid until the next `forward` or `backward` on the same layer, or until the layer is destroyed. Keep values with `to_host` or `copy_from`.

`NamedParam` and `NamedBuffer` hold raw pointers into layer storage. `parameters()` and `buffers()` return a fresh vector of those pointers. The storage lives as long as the owning layer. Moving a `Network`, `ConvResNet`, or `Transformer` moves the `unique_ptr<Layer>` elements. The layer objects stay where they are, so an optimizer built from `parameters()` remains valid after the move.

Weight and bias gradients accumulate. Call `zero_grad` once per step. `zero_grad` does not change parameter values, optimizer state, or batch-norm buffers. A new network is in train mode. Call `eval()` before inference when the network contains batch-norm or dropout.

Parameters, the gradients Adam sees, optimizer state, and checkpoints are dense float32. Retained activations (layer outputs and caches, attention scores, and the KV cache) are bfloat16 unless `set_activation_dtype(Float32)` selects float32. A matmul reads bfloat16 operands and accumulates float32. Softmax, layer-norm, and RMSNorm compute in float32. `to_host()` widens bfloat16 to float. `data()` on a bfloat16 tensor throws `tensor dtype`. Index buffers are dense `int64`. Layout:

| Rank | Shape | Order |
|---|---|---|
| 1 | `(F,)` | contiguous |
| 2 | `(N, F)` | row-major, feature index contiguous |
| 3 | `(N, T, F)` | feature index contiguous |
| 4 | `(N, C, H, W)` | NCHW |

A tensor element count must fit in a signed 32-bit integer. Every dimension is at least 1.

## 2. Errors

Two exception types are part of the contract. The message text is part of the contract. Callers match it.

`nn::Error` extends `std::runtime_error`. It reports a failed backend call, a bad tensor or layer argument, a device failure, or a checkpoint problem. A backend status is `call: STATUS_NAME`. CUDA examples look like `cudnnConvolutionForward: CUDNN_STATUS_BAD_PARAM` or `cudaMemcpyAsync: <cuda error name>`. OpenCL examples look like `clFinish: CL_STATUS_NAME`.

`std::invalid_argument` reports a rejected config field or one of the call-site checks named in the sections below. `validate` throws `std::invalid_argument` and the message is the field name.

`Context` is neither copyable nor movable. `Tensor`, `IndexTensor`, `DeviceIndices`, `Network`, `ConvResNet`, `Transformer`, `Sgd`, and `AdamW` are movable and not copyable. `LossResult` is movable and not copyable.

## 3. Context and devices

Header: `nn/context.hpp`

```cpp
struct DeviceInfo {
  int index;
  std::string name;
};

int device_count();
std::string device_name(int device);
std::vector<DeviceInfo> detect_devices();
int select_device(const std::string& choice);

class Context {
 public:
  Context();
  explicit Context(int device);
  explicit Context(const std::string& choice);
  void synchronize() const;
  void set_activation_dtype(DType dtype);
  DType activation_dtype() const;
};
```

`device_count()` is the length of the list below. `device_name(int)` is that entry's name. An index outside `0 .. device_count()-1` throws `nn::Error` with message `device`. `detect_devices()` returns the same list in index order. `select_device("auto")` returns the first entry. Any other choice is a decimal index into that list, after leading and trailing whitespace is removed. `auto` is matched case-insensitively. A blank choice, a word other than `auto`, an index past the list, or an empty list throws `device`.

`synchronize()` waits until work queued on that context has finished.

`set_activation_dtype` accepts `Float32` or `BFloat16` and throws `dtype` otherwise. The default is `BFloat16`. It selects the storage of retained activations. Parameters, Adam state, and checkpoints stay float32. Call it before creating tensors or layers.

`to_host`, `kaiming_normal`, and `normal_` synchronize for you. `copy_from_host` queues the upload. Call `synchronize()` before reusing or freeing the host buffer.

| | `nn` (CUDA) | `nnCL` (OpenCL) |
|---|---|---|
| `Context()` | Uses the current CUDA device. Does not call `cudaSetDevice`. | Opens index 0 of the filtered list below. |
| `Context(int)` | Calls `cudaSetDevice`, then opens that device. | Opens that index. Does not reorder the process-wide device. |
| `Context(string)` | `select_device`, then `cudaSetDevice`. | `select_device`, then open that index. |
| Visible devices | Every visible CUDA device. | Devices accepted by the build filter below. |
| Queue | One non-default CUDA stream, plus a cuDNN handle and a cuBLAS handle bound to it. | One in-order OpenCL command queue. |
| `data()` | A CUDA device pointer. A CUDA caller may `cudaMemcpy` it on this stream. | Not a host pointer. Read it with `to_host()`. |

`nnCL` device filter. The CMake cache variable `NNCL_OPENCL` is `auto`, `radeon`, `intel`, `apple`, or `nvidia`. It is fixed at compile time. `auto` keeps GPUs and does not keep a CPU device. `radeon`, `apple`, and `nvidia` keep that vendor's GPUs. `intel` keeps an Intel GPU, or the Intel CPU device when no Intel GPU is listed. A device is kept only when its max work-group size is at least 256. A requested vendor that is not installed fails at context creation with `device`. The kernel source is `src/kernels.cl`, compiled at runtime with `-cl-std=CL1.2` or `-cl-std=CL2.0` from the device's OpenCL C version. A device that reports 2.0 or 3.0 is built as OpenCL C 2.0. A 1.2 device is built as OpenCL C 1.2. An Apple host includes `<OpenCL/cl.h>` and creates the queue with `clCreateCommandQueue`. Other hosts use the OpenCL 3.0 headers and `clCreateCommandQueueWithProperties`.

`select_device("auto")` is the first device in the list above. It is not the `NNCL_OPENCL` vendor filter.

## 4. Tensor

Header: `nn/tensor.hpp`

```cpp
class Tensor {
 public:
  Tensor(Context& ctx, std::vector<int> shape);
  Tensor(Context& ctx, std::vector<int> shape, DType dtype);
  Context& context() const;
  int rank() const;
  const std::vector<int>& shape() const;
  int64_t numel() const;
  DType dtype() const;
  float* data();
  const float* data() const;
  uint16_t* bf16_data();
  const uint16_t* bf16_data() const;
  void zero();
  void fill(float value);
  void scale(float factor);
  void copy_from(const Tensor& src);
  void copy_from_host(const float* src, int64_t n);
  void copy_from_host(const std::vector<float>& src);
  std::vector<float> to_host() const;
  Tensor reshape(const std::vector<int>& shape) const;
};
```

The constructor allocates device memory. `Tensor(ctx, shape)` is float32. Fresh memory is uninitialized. Call `zero`, `fill`, or `copy_from_host` before a read. `data()` is the float32 pointer and throws `tensor dtype` when the tensor is bfloat16. `bf16_data()` is the bfloat16 pointer and throws `tensor dtype` when the tensor is float32. `to_host()` returns float and widens bfloat16. `copy_from` converts when the dtypes differ. `reshape` keeps the dtype.

| Condition | Exception | Message |
|---|---|---|
| Rank outside 1..4 | `nn::Error` | `tensor rank out of range` |
| A dimension below 1 | `nn::Error` | `tensor dimension out of range` |
| Element count does not fit in `int32` | `nn::Error` | `tensor numel out of range` |
| `copy_from` element counts differ | `nn::Error` | `tensor copy numel mismatch` |
| `copy_from_host` count differs from `numel()` | `nn::Error` | `tensor host copy numel mismatch` |
| `reshape` element counts differ | `nn::Error` | `tensor reshape numel mismatch` |
| `data()` on bfloat16, or `bf16_data()` on float32 | `nn::Error` | `tensor dtype` |

`copy_from` requires the same element count. The source is complete before the destination write is queued. On CUDA, a source on another context is synchronized and the copy is `cudaMemcpyAsync` on the destination stream. On OpenCL, a source on another context is read with `to_host` and then uploaded. `scale` multiplies every element in place on the context queue and does not synchronize. `to_host` synchronizes, then returns a `std::vector<float>`. `reshape` allocates a new tensor on the same context and copies. The destructor synchronizes the queue and frees the device memory.

The portable way to read a tensor is `to_host()`. The portable way to write one is `copy_from_host`.

## 5. DeviceIndices

Header: `nn/loss.hpp`

```cpp
class DeviceIndices {
 public:
  DeviceIndices(Context& ctx, int64_t n);
  void copy_from_host(const int64_t* src, int64_t n);
  const int64_t* data() const;
  int64_t size() const;
  Context& context() const;
};
```

Label buffer for `softmax_cross_entropy`. `n` is the batch size and must be positive. `n <= 0` throws `nn::Error` with message `empty labels`. Values are `int64` class indices on the device. `copy_from_host` requires `n == size()` and throws `label count mismatch` otherwise. The destructor synchronizes and frees the buffer.

## 6. IndexTensor

Header: `nn/index_tensor.hpp`

```cpp
class IndexTensor {
 public:
  IndexTensor(Context& ctx, std::vector<int> shape);
  void copy_from_host(const int64_t* src, int64_t n);
  std::vector<int64_t> to_host() const;
  const int64_t* data() const;
  const std::vector<int>& shape() const;
  int64_t numel() const;
  Context& context() const;
};
```

Dense `int64` index buffer for `Embedding` and for `sequence_cross_entropy`. It is not a float `Tensor`. Rank is 1 `(N,)` or 2 `(N, T)`.

| Condition | Message |
|---|---|
| Rank other than 1 or 2 | `index tensor rank out of range` |
| A dimension below 1 | `index tensor dimension out of range` |
| Element count does not fit in `int32` | `index tensor numel out of range` |
| `copy_from_host` count differs | `index tensor host copy numel mismatch` |

Those four throw `nn::Error`. `copy_from_host` is asynchronous. `to_host` synchronizes, then returns the indices. The destructor synchronizes and frees the buffer. On both backends, `to_host()` is the portable read. Do not `cudaMemcpy` `data()` in a program that also builds against `nnCL`.

## 7. Loss

Header: `nn/loss.hpp`

```cpp
struct LossResult {
  float loss = 0.f;
  Tensor grad_logits;
};

LossResult softmax_cross_entropy(Context& ctx, const Tensor& logits, const DeviceIndices& labels);
LossResult sequence_cross_entropy(Context& ctx, const Tensor& logits, const IndexTensor& labels,
                                  int64_t ignore_index = -100);
```

Both throw `nn::Error`. `LossResult` owns `grad_logits`.

`softmax_cross_entropy`. `logits` is rank 2, shape `(N, C)`, with `C > 1`. `labels.size()` equals `N`. Each label is in `[0, C)`. `loss` is the mean softmax cross-entropy over the batch. `grad_logits` has the same shape as `logits` and holds `(softmax - one_hot) / N`.

| Condition | Message |
|---|---|
| Rank other than 2 | `logits rank` |
| `N <= 0` or `C <= 1` | `logits shape` |
| Label count differs from `N` | `label count mismatch` |
| A label outside `[0, C)` | `label out of range` |

`softmax_cross_entropy` rejects every negative label, including `-100`.

`sequence_cross_entropy`. Mean softmax cross-entropy over rows that are already aligned with the labels. This function does not shift tokens. `logits` is `(N, T, V)` with `labels` `(N, T)`, or `logits` is `(rows, V)` with `labels` `(rows,)`. `V > 1`. `grad_logits` has the same shape as `logits`.

A label equal to `ignore_index` is left out of the mean, and that row of `grad_logits` is zero. The default `ignore_index` is `-100`. Any other label outside `[0, V)` throws `label out of range`. The mean and the gradients divide by the number of labels that were kept. If every label is ignored, the loss is `0`, `grad_logits` is all zeros, and the function does not divide by zero.

| Condition | Message |
|---|---|
| Rank other than 2 or 3 | `logits rank` |
| `V <= 1` | `logits shape` |
| Label rank or dimensions do not match `logits` | `label shape` |
| A label other than `ignore_index` outside `[0, V)` | `label out of range` |

## 8. Parameters, buffers, and Layer

Header: `nn/layer.hpp`

```cpp
struct NamedParam {
  std::string name;
  Tensor* value = nullptr;
  Tensor* grad = nullptr;
};

struct NamedBuffer {
  std::string name;
  Tensor* value = nullptr;
  uint64_t* batches = nullptr;
};

class Layer {
 public:
  virtual const Tensor& forward(const Tensor& input) = 0;
  virtual const Tensor& backward(const Tensor& grad_output) = 0;
  virtual std::vector<NamedParam> parameters() const = 0;
  virtual std::vector<NamedBuffer> buffers() const;
  virtual void zero_grad() = 0;
  virtual void set_training(bool);
  const std::string& name() const;
  void set_name(std::string name);
};
```

A parameter is a trainable tensor plus its gradient. A buffer is state that is saved and restored and is left alone by `zero_grad` and by the optimizers. `batches` points at the batch-norm counter shared by that layer's `running_mean` and `running_var`.

`Layer::parameters()` returns names local to the layer, such as `weight` and `bias`. `Network::parameters()` prefixes the layer name. A layer named `fc.0` contributes `fc.0.weight` and `fc.0.bias`. An empty layer name contributes the local name unchanged. `ResidualBlock` and `TransformerBlock` prefix their children the same way.

`buffers()` defaults to an empty list. `set_training` defaults to a no-op. `BatchNorm2d`, `Dropout`, `ResidualBlock`, and `TransformerBlock` honor it. `LayerNorm` and `RMSNorm` ignore it: train and eval use the same math.

`backward` reads the activation saved by the matching `forward`. Call `forward` on that layer before `backward`. Unless a section below names `std::invalid_argument`, a failed layer check throws `nn::Error` and the message is the phrase in that section.

## 9. Layers

### Linear

Header: `nn/layers/linear.hpp`

```cpp
Linear(Context& ctx, int in_features, int out_features, std::mt19937& rng);
Linear(Context& ctx, int in_features, int out_features, bool use_bias, std::mt19937& rng);
```

The three-argument constructor calls the four-argument constructor with `use_bias` true. `in_features` and `out_features` are at least 1 (`Linear size`). The constructor draws `weight` with `kaiming_normal` and fan-in `in_features`. When `use_bias` is true, `bias` starts at zero and consumes no random numbers.

| Tensor | Shape | Name |
|---|---|---|
| weight | `(out_features, in_features)` | `weight` |
| bias, when `use_bias` | `(out_features,)` | `bias` |

`forward` takes rank 2 `(N, in_features)` or rank 3 `(N, T, in_features)` and returns `(N, out_features)` or `(N, T, out_features)`:

```
y = x W^T + b
```

When `use_bias` is false, `b` is omitted. Rank 3 is the same product with `N * T` rows. A rank other than 2 or 3, or a feature count other than `in_features`, throws `Linear input shape`. `backward` takes a gradient of the forward output shape, returns the input shape, and adds into `grad_weight` and, when bias is on, `grad_bias`. A mismatched gradient throws `Linear backward shape`.

### Embedding

Header: `nn/layers/embedding.hpp`

```cpp
Embedding(Context& ctx, int num_embeddings, int embedding_dim, std::mt19937& rng);
const Tensor& forward_indices(const IndexTensor& indices);
```

`num_embeddings` and `embedding_dim` are at least 1 (`Embedding size`). The constructor draws `weight` with `normal_` at mean 0 and standard deviation 0.02. There is no padding index. The index tensor must come from the same context (`Embedding context`).

| Tensor | Shape | Name |
|---|---|---|
| weight | `(num_embeddings, embedding_dim)` | `weight` |

`forward(const Tensor&)` throws `Embedding requires indices`. Call `forward_indices`.

`forward_indices` reads rank 1 `(N,)` or rank 2 `(N, T)` and returns `(N, embedding_dim)` or `(N, T, embedding_dim)`. It copies the ids to the host and checks them before the gather, which synchronizes. Each id must be in `[0, num_embeddings)`. An id outside that range throws `index out of range` and leaves the previous output in place. The output row is `weight[id]`.

`backward` takes a gradient of that output shape and scatter-adds it into `grad_weight`. Duplicate ids in one batch accumulate. `backward` does not clear `grad_weight`. A call without a successful `forward_indices` throws `Embedding backward without forward`. A mismatched gradient throws `Embedding backward shape`. Token ids are not differentiable. The returned tensor has the same shape as `grad_output` and is filled with zeros.

### ReLU

Header: `nn/layers/relu.hpp`

```cpp
explicit ReLU(Context& ctx);
```

No parameters. `forward` and `backward` accept rank 2, 3, or 4 and return the same shape. `y = max(x, 0)`. Rank 2 is presented to the backend as `(N, 1, 1, F)`. Rank 3 `(N, T, F)` is presented as `(N, 1, T, F)`. Any other rank throws `ReLU rank`. A gradient whose shape differs from the saved forward output throws `ReLU backward shape`. `zero_grad` is empty.

### GELU

Header: `nn/layers/gelu.hpp`

```cpp
explicit GELU(Context& ctx);
```

No parameters. `forward` and `backward` accept rank 2, 3, or 4 and return the same shape. Any other rank throws `GELU rank`.

```
GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
```

This is the erf form, not the tanh approximation. `forward` stores the input. `backward` multiplies the incoming gradient by `Φ(x) + x φ(x)`, where `Φ` is the standard normal CDF and `φ` is its density. A gradient whose shape differs from the saved input throws `GELU backward shape`. `zero_grad` is empty.

### SwiGLU

Header: `nn/layers/swiglu.hpp`

```cpp
explicit SwiGLU(Context& ctx);
```

No parameters. Rank 2 `(N, 2F)` returns `(N, F)`. Rank 3 `(N, T, 2F)` returns `(N, T, F)`. Rank 4 `(N, 2C, H, W)` returns `(N, C, H, W)`. Rank 2 and rank 4 split dimension 1. Rank 3 splits the last dimension. The first half is the gate `a` and the second half is the value `b`:

```
y = SiLU(a) * b
SiLU(a) = a * sigmoid(a)
```

Any other rank throws `SwiGLU rank`. An odd size on the split dimension throws `SwiGLU size`. `backward` returns the input shape. A gradient whose shape differs from the forward output throws `SwiGLU backward shape`. `zero_grad` is empty.

### RNN

Header: `nn/layers/rnn.hpp`

```cpp
RNN(Context& ctx, int input_size, int hidden_size, std::mt19937& rng);
```

One Elman cell. `input_size` and `hidden_size` are at least 1 (`RNN size`). `weight_ih` is drawn with fan-in `input_size`. `weight_hh` is drawn with fan-in `hidden_size`. Both biases start at zero and consume no random numbers.

| Tensor | Shape | Name |
|---|---|---|
| weight_ih | `(hidden_size, input_size)` | `weight_ih` |
| weight_hh | `(hidden_size, hidden_size)` | `weight_hh` |
| bias_ih | `(hidden_size,)` | `bias_ih` |
| bias_hh | `(hidden_size,)` | `bias_hh` |

`forward` takes `(N, T, input_size)` and returns `(N, T, hidden_size)`. The hidden state before step 0 is zero.

```
h_t = tanh(x_t W_ih^T + b_ih + h_{t-1} W_hh^T + b_hh)
```

A rank other than 3, or a feature count other than `input_size`, throws `RNN input shape`. `backward` walks the sequence from the last step to the first, returns `(N, T, input_size)`, and adds into all four gradients. A gradient whose shape differs from the forward output, including a backward with no forward, throws `RNN backward shape`.

### LastStep

Header: `nn/layers/last_step.hpp`

```cpp
class LastStep : public Layer { /* default constructor */ };
```

No context and no parameters. `forward` takes `(N, T, F)` and returns `(N, F)`, the values at time `T - 1`. A rank other than 3 throws `LastStep input rank`. `backward` writes the incoming gradient into that last step and leaves every earlier step at zero. A backward with no forward throws `LastStep backward without forward`. A mismatched gradient throws `LastStep backward shape`. `zero_grad` is empty.

### Conv2d

Header: `nn/layers/conv2d.hpp`

```cpp
Conv2d(Context& ctx, int in_channels, int out_channels, int kernel, int stride, int pad,
       bool use_bias, std::mt19937& rng);
```

The kernel is square. `in_channels` and `out_channels` are at least 1 (`Conv2d channels`). `kernel` and `stride` are at least 1, and `pad` is at least 0. Those three checks go through `conv_out_size` and throw `std::invalid_argument` with message `kernel`, `stride`, or `pad`.

Weight shape is `(out_channels, in_channels, kernel, kernel)`. Fan-in is `in_channels * kernel * kernel`. Bias, when `use_bias` is true, has shape `(out_channels,)` and starts at zero. Parameter names are `weight`, then `bias` when bias is on.

`forward` takes `(N, in_channels, H, W)` and returns `(N, out_channels, H_out, W_out)`:

```
H_out = (H + 2 * pad - kernel) / stride + 1
```

`W_out` uses the same formula. Integer division truncates. A non-positive result throws `std::invalid_argument` with message `spatial`. A rank other than 4, or a channel count other than `in_channels`, throws `Conv2d input shape`.

The convolution is cross-correlation, dilation 1, groups 1. The first call at a given spatial size allocates one workspace, at most 64 MiB. A larger request throws `conv workspace <bytes> exceeds 64 MiB budget`. A later shape that needs more memory frees and reallocates.

`backward` accumulates `grad_weight` and, when bias is on, `grad_bias`. The gradient's shape matches the forward output. A mismatch throws `Conv2d backward shape`.

### MaxPool2d

Header: `nn/layers/max_pool2d.hpp`

```cpp
MaxPool2d(Context& ctx, int kernel, int stride);
```

The window is square and the padding is 0. `kernel` and `stride` are at least 1, checked by `conv_out_size` (`kernel`, `stride`, or `spatial` as `std::invalid_argument`). `forward` takes rank 4 and returns `(N, C, H_out, W_out)` with pad 0. A rank other than 4 throws `MaxPool2d input rank`. A gradient whose shape differs from the forward output throws `MaxPool2d backward shape`. There are no parameters. `zero_grad` is empty.

### BatchNorm2d

Header: `nn/layers/batch_norm2d.hpp`

```cpp
BatchNorm2d(Context& ctx, int channels);
```

`channels` is at least 1 (`BatchNorm2d channels`).

| Tensor | Shape | Name | Initial value |
|---|---|---|---|
| weight (scale) | `(channels,)` | `weight` | 1 |
| bias (shift) | `(channels,)` | `bias` | 0 |
| running mean | `(channels,)` | `running_mean` | 0 |
| running variance | `(channels,)` | `running_var` | 1 |

The batch counter starts at 0. `parameters()` returns `weight` and `bias`. `buffers()` returns `running_mean` and `running_var`. Both buffers share the counter through `NamedBuffer::batches`.

`forward` takes `(N, channels, H, W)` and returns the same shape. A rank other than 4, or a channel mismatch, throws `BatchNorm2d input shape`. The mode is spatial and epsilon is `1e-5`.

In train mode the layer requires `N * H * W > 1`. A smaller count throws `BatchNorm train mode needs more than one value per channel`. Each train forward increments the counter, then updates the running statistics with factor `1 / batches`. Eval forward uses the stored mean and variance and leaves them unchanged.

`backward` runs in train mode and adds into the scale and shift gradients. Eval mode throws `BatchNorm backward requires train mode`. A missing forward or a mismatched gradient throws `BatchNorm2d backward shape`.

`set_training(true)` selects train mode. `set_training(false)` selects eval mode. `Network::train` and `Network::eval` call `set_training` on every layer.

### LayerNorm

Header: `nn/layers/layer_norm.hpp`

```cpp
LayerNorm(Context& ctx, int normalized_shape, float eps = 1e-5f);
```

`normalized_shape` is at least 1 (`LayerNorm size`). `eps` is finite and greater than 0 (`LayerNorm eps`). The default is `1e-5`.

`weight` (gamma) has shape `(normalized_shape,)` and starts at 1. `bias` (beta) has the same shape and starts at 0. Names are `weight` and `bias`. There are no buffers. `set_training` does not change the result. Construction consumes no random numbers.

`forward` accepts rank 2 `(N, normalized_shape)` or rank 3 `(N, T, normalized_shape)` and returns the same shape. Mean and variance are over the last dimension. Variance is the mean of squared deviations, divided by `normalized_shape`, not `normalized_shape - 1`:

```
y = (x - mean) / sqrt(var + eps) * weight + bias
```

A rank other than 2 or 3, or a last dimension other than `normalized_shape`, throws `LayerNorm input shape`. `backward` returns the input shape and adds into `grad_weight` and `grad_bias`. A missing forward or a mismatched gradient throws `LayerNorm backward shape`.

On the learned-absolute transformer path, `ln1`, `ln2`, and `final_ln` are this layer with the default epsilon. `TransformerConfig::rms_eps` does not change them.

### RMSNorm

Header: `nn/layers/rms_norm.hpp`

```cpp
RMSNorm(Context& ctx, int normalized_shape, float eps = 1e-5f);
```

`normalized_shape` is at least 1 (`RMSNorm size`). `eps` is finite and greater than 0 (`RMSNorm eps`). The default is `1e-5`.

`weight` has shape `(normalized_shape,)` and starts at 1. There is no bias. The name is `weight`. There are no buffers. `set_training` does not change the result. Construction consumes no random numbers.

`forward` accepts rank 2 `(N, normalized_shape)` or rank 3 `(N, T, normalized_shape)` and returns the same shape:

```
y = x * rsqrt(mean(x^2) + eps) * weight
```

The mean is over the last dimension. A rank other than 2 or 3, or a last dimension other than `normalized_shape`, throws `RMSNorm input shape`. `backward` returns the input shape and adds into `grad_weight`. A missing forward or a mismatched gradient throws `RMSNorm backward shape`.

On the RoPE transformer path, `ln1`, `ln2`, and `final_ln` are this layer with `TransformerConfig::rms_eps`. QK-norm is this layer with width `head_dim` and the same epsilon.

### Dropout

Header: `nn/layers/dropout.hpp`

```cpp
Dropout(Context& ctx, float p);
```

`p` is in `[0, 1)`. Any other value, including NaN, throws `Dropout p`. The layer owns an `std::mt19937` seeded from `std::random_device`. There is no seed setter. The network seed does not determine dropout masks. There are no parameters and no buffers.

`set_training(true)` is train mode, which is the default. `set_training(false)` is eval mode. `p == 0` is the identity in both modes. Eval mode is the identity for every legal `p`.

In train mode with `p > 0` the layer draws an inverted-dropout mask on the host and uploads it. A kept element is multiplied by `1 / (1 - p)`. A dropped element is 0. `backward` multiplies by that same mask. Each train `forward` draws a new mask.

`forward` accepts any rank a tensor can store and returns the same shape. A backward with no forward, or a gradient whose shape differs from the forward output, throws `Dropout backward shape`.

### MultiHeadAttention

Header: `nn/layers/multi_head_attention.hpp`

```cpp
MultiHeadAttention(Context& ctx, int d_model, int num_heads, bool causal, bool use_bias,
                   bool use_rope, std::mt19937& rng, int head_dim = 0, int num_kv_heads = 0,
                   bool qk_norm = false, float rope_theta = 10000.f, float rms_eps = 1e-5f,
                   std::optional<bool> qkv_bias = std::nullopt,
                   std::optional<bool> attn_out_bias = std::nullopt);

struct Cache {
  Tensor k;  // (N, num_heads, T_max, head_dim) after KV repeat
  Tensor v;
  int t = 0;  // next write index
};

void reset_cache(Cache& cache, int batch, int t_max) const;
const Tensor& forward_cached(const Tensor& x_t, Cache& cache);
```

`d_model` and `num_heads` are at least 1. `head_dim == 0` means `head_dim = d_model / num_heads`, and `d_model` must be divisible by `num_heads`. `num_kv_heads == 0` means one KV head per query head. Otherwise `num_kv_heads` is in `1 .. num_heads` and `num_heads % num_kv_heads == 0`. A failed size check throws `MultiHeadAttention size`. `head_dim * num_heads` and `head_dim * num_kv_heads` must fit in a positive `int`.

`use_rope == true` rotates Q and K after QK-norm: adjacent pairs inside each head, base `rope_theta`, over the whole head dimension. V is not rotated. An odd head dimension throws `std::invalid_argument` with message `use_rope`. `rope_theta` must be finite and greater than 0 (`rope_theta`). `rms_eps` must be finite and greater than 0 (`rms_eps`). Those two checks run even when `use_rope` and `qk_norm` are false. There is no attention dropout. `set_training` does not change the result.

The layer owns four `Linear` children, constructed in order `q`, `k`, `v`, `out`. `q` maps `d_model` to `num_heads * head_dim`. `k` and `v` map `d_model` to `num_kv_heads * head_dim`. `out` maps `num_heads * head_dim` back to `d_model`. Each child uses `use_bias` when the matching optional is unset. `qkv_bias` covers `q`, `k`, and `v`. `attn_out_bias` covers `out`. Each child draws its weight from `rng`. Bias, when present, starts at 0 and consumes no random numbers. When `head_dim == 0` and `num_kv_heads == 0`, `q`, `k`, `v`, and `out` are square at `d_model`.

When `qk_norm` is true, the layer also owns `q_norm` and `k_norm`, each an `RMSNorm` of width `head_dim` and epsilon `rms_eps`. Their weights start at 1 and consume no random numbers.

Parameter order:

| Name | Shape | Present |
|---|---|---|
| `q.weight` | `(num_heads * head_dim, d_model)` | always |
| `q.bias` | `(num_heads * head_dim,)` | when `use_bias` |
| `k.weight`, `v.weight` | `(num_kv_heads * head_dim, d_model)` | always |
| `k.bias`, `v.bias` | `(num_kv_heads * head_dim,)` | when `use_bias` |
| `out.weight` | `(d_model, num_heads * head_dim)` | always |
| `out.bias` | `(d_model,)` | when `use_bias` |
| `q_norm.weight`, `k_norm.weight` | `(head_dim,)` | when `qk_norm` |

`forward` takes `(N, T, d_model)` and returns the same shape. A rank other than 3, or a last dimension other than `d_model`, throws `MultiHeadAttention input shape`.

The score path is:

1. Project Q, K, and V with the child linears.
2. Split into heads. Q has `num_heads` heads. K and V have `num_kv_heads` heads.
3. When `qk_norm` is set, RMS-normalize each Q head and each K head. V is not normalized.
4. Repeat each KV head `num_heads / num_kv_heads` times so K and V have `num_heads` heads.
5. When `use_rope` is set, rotate Q and K at positions `0 .. T-1`. V is not rotated.
6. Scores are `Q K^T / sqrt(head_dim)` with shape `(N, num_heads, T, T)`.
7. When `causal` is true, entries with key index `j > i` are set to `-1e9` before the softmax over the last dimension. The value is `-1e9`, not `-inf`. A row of infinities makes the max-subtract softmax NaN. When `causal` is false, every query attends to every key.
8. The attention output is merged and projected by `out`.

`backward` returns the input shape and adds into each child's gradients. Masked scores have zero gradient. A missing forward or a mismatched gradient throws `MultiHeadAttention backward shape`. `zero_grad` clears every child, including the QK-norms when they exist. Full-sequence backward undoes the Q and K rotation before the linear gradients.

`reset_cache` allocates `k` and `v` on this context with shape `(batch, num_heads, t_max, head_dim)`, zeros them, and sets `t = 0`. The head axis is the query-head count, after KV repeat. `batch < 1` throws `std::invalid_argument` with message `batch`. `t_max < 1` throws `t_max`.

`forward_cached` is the generation step. `x_t` must be `(N, 1, d_model)` on this context (`x_t`). The cache must be on this context, with batch `N`, query-head count `num_heads`, and head dimension `head_dim`, and `cache.t` must be inside `T_max` (`cache`). The step projects that token, applies QK-norm, repeats KV, rotates Q and K at absolute position `cache.t` when `use_rope` is set, and writes K and V at `cache.t`. Scores cover keys `0 .. cache.t` inclusive. The unwritten tail is not read. There is no `(T, T)` causal mask, because the future keys are not in the product. The return is `(N, 1, d_model)`. `cache.t` then increases by 1. `backward` after `forward_cached` throws `std::invalid_argument` with message `forward_cached`. A later `forward` clears that state.

### TransformerBlock

Header: `nn/layers/transformer_block.hpp`

```cpp
struct TransformerBlockSpec {
  int d_model = 0;
  int num_heads = 0;
  int d_ff = 0;           // width after the gate; SwiGLU's first linear writes 2*d_ff
  float dropout = 0.f;    // in [0, 1)
  bool bias = false;
  bool use_rope = false;
  TransformerActivation activation = TransformerActivation::SwiGLU;
  int head_dim = 0;
  int num_kv_heads = 0;
  bool qk_norm = false;
  float rope_theta = 10000.f;
  float rms_eps = 1e-5f;
  bool causal = true;
  bool qkv_bias = false;
  bool attn_out_bias = false;
  bool mlp_bias = false;
  bool lm_head_bias = false;
  bool split_bias = false;
};

TransformerBlock(Context& ctx, const TransformerBlockSpec& spec, std::mt19937& rng);
MultiHeadAttention& attn();
const Tensor& forward_cached(const Tensor& x_t, MultiHeadAttention::Cache& cache);
```

`TransformerActivation` is defined in `nn/config.hpp`: `Gelu = 1`, `SwiGLU = 2`.

Pre-norm block. `d_model`, `num_heads`, and `d_ff` are at least 1. Head size follows the `MultiHeadAttention` rules. A failed size check throws `TransformerBlock size`, including a SwiGLU `d_ff` above `INT_MAX / 2`. `dropout` is in `[0, 1)` (`TransformerBlock dropout`). Any other `activation` throws `TransformerBlock activation`. `use_rope` false builds `LayerNorm` for `ln1` and `ln2` with epsilon `1e-5`. `use_rope` true builds `RMSNorm` for both norms with epsilon `rms_eps` and passes `use_rope` to attention. An odd head dimension with `use_rope` throws `std::invalid_argument` with message `use_rope`. `rope_theta` and `rms_eps` throw `std::invalid_argument` under the same rules as attention. `causal` is the attention mask and defaults to true.

```
h = x + drop1(mha(ln1(x)))
y = h + drop2(ffn(ln2(h)))
```

`d_ff` is the width after the gate. GELU is `Linear(d_model → d_ff)`, `GELU`, `Linear(d_ff → d_model)`. SwiGLU is `Linear(d_model → 2*d_ff)`, `SwiGLU`, `Linear(d_ff → d_model)`. Both linears use `mlp_bias` when `split_bias` is set, and `bias` otherwise. Q, K, and V use `qkv_bias` the same way. The attention output uses `attn_out_bias`. `attn()` is the owned attention layer. `set_training` reaches both dropouts and every child. Child parameter prefixes are `ln1.`, `attn.`, `drop1.`, `ln2.`, `fc1.`, `fc2.`. Dropout and the activation have no parameters. `forward` takes `(N, T, d_model)` and returns the same shape. A rank other than 3, or a last dimension other than `d_model`, throws `TransformerBlock input shape`. A missing forward or a mismatched gradient throws `TransformerBlock backward shape`.

`forward_cached` is the same residual with `mha->forward_cached` in place of `mha->forward`. `x_t` and `y` are `(N, 1, d_model)`. Dropout follows `set_training`. Generation calls `eval()`, so dropout is the identity.

### Flatten

Header: `nn/layers/flatten.hpp`

```cpp
class Flatten : public Layer { /* default constructor */ };
```

No context and no parameters. `forward` takes `(N, C, H, W)` and returns `(N, C*H*W)`, copying the values in NCHW order. A rank other than 4 throws `Flatten input rank`. A feature count that does not fit in a signed 32-bit integer throws `Flatten features out of range`. `backward` takes `(N, C*H*W)` and returns `(N, C, H, W)`. A backward with no forward throws `Flatten backward without forward`. A mismatched gradient throws `Flatten backward shape`.

### GlobalAvgPool2d

Header: `nn/layers/global_avg_pool2d.hpp`

```cpp
explicit GlobalAvgPool2d(Context& ctx);
```

No parameters. `forward` takes `(N, C, H, W)` and returns `(N, C, 1, 1)`. A rank other than 4 throws `GlobalAvgPool2d input rank`. `backward` spreads the `(N, C, 1, 1)` gradient evenly across the spatial map. A mismatched gradient throws `GlobalAvgPool2d backward shape`.

### ResidualBlock

Header: `nn/layers/residual_block.hpp`

```cpp
ResidualBlock(Context& ctx, int in_channels, int out_channels, int stride, std::mt19937& rng);
```

`in_channels`, `out_channels`, and `stride` are at least 1 (`ResidualBlock size`). Basic residual unit:

```
main = bn2(conv2(relu(bn1(conv1(x)))))
skip = x                                 when stride == 1 and in_channels == out_channels
skip = proj_bn(proj(x))                  otherwise
y    = relu(main + skip)
```

`conv1` is 3×3 with the given stride and pad 1. `conv2` is 3×3 with stride 1 and pad 1. `proj` is 1×1 with the given stride and pad 0. Every convolution has bias off. Each one consumes `rng` through `kaiming_normal`, in order `conv1`, `conv2`, then `proj` when the projection exists. Batch-norm scale, bias, and running statistics use the `BatchNorm2d` initial values and consume no random numbers.

`forward` takes `(N, in_channels, H, W)`. Output spatial size follows the 3×3 convolution with the block stride and pad 1. Output channels are `out_channels`.

| Name | Present |
|---|---|
| `conv1.weight` | always |
| `bn1.weight`, `bn1.bias` | always |
| `conv2.weight` | always |
| `bn2.weight`, `bn2.bias` | always |
| `proj.weight` | when the skip projects |
| `proj_bn.weight`, `proj_bn.bias` | when the skip projects |

Buffers follow the same prefixes: `bn1.running_mean`, `bn1.running_var`, `bn2.running_mean`, `bn2.running_var`, and `proj_bn.running_mean`, `proj_bn.running_var` when the skip projects.

`set_training` forwards to every child. Train mode follows the `BatchNorm2d` rules on each norm in the block, including `N * H * W > 1` at that child's spatial size. A skip whose shape differs from the main path throws `ResidualBlock skip shape`. A backward gradient mismatch throws `ResidualBlock grad shape`.

## 10. Configuration

Header: `nn/config.hpp`

Builders call `validate` before they allocate. A caller may call it directly. A failed check throws `std::invalid_argument` and the message is the field name.

```cpp
int conv_out_size(int in, int kernel, int stride, int pad);
```

The result is `(in + 2 * pad - kernel) / stride + 1`. `kernel` and `stride` are at least 1 (`kernel`, `stride`). `pad` is at least 0 (`pad`). A result below 1 throws `spatial`.

```cpp
enum class Activation : uint32_t { Relu = 1, Gelu = 2, SwiGLU = 3 };
enum class TransformerActivation : uint32_t { Gelu = 1, SwiGLU = 2 };
enum class PositionalEncoding : uint32_t { LearnedAbsolute = 1, Rope = 2 };
```

### MlpConfig

```cpp
struct MlpConfig {
  int input_size = 0;
  std::vector<int> hidden;
  int output_size = 0;
  Activation activation = Activation::Relu;
};
```

`input_size`, every hidden width, and `output_size` are at least 1. `activation` is `Relu`, `Gelu`, or `SwiGLU`. Any other value throws `activation`. A SwiGLU hidden width above `INT_MAX / 2` throws `hidden`.

`build_mlp` emits `Linear`, the activation, `Linear`, the activation, and so on, then a last `Linear` with no activation. Names are `fc.0`, `fc.1`, ... An empty `hidden` is one linear layer. `Relu` and `Gelu` keep each hidden width. `SwiGLU` uses `H` as the width after the gate: the linear before each `SwiGLU` writes `2H`, `SwiGLU` returns `H`, and the next linear reads `H`. The last linear writes `output_size` and has no gate. Input is `(N, input_size)`. Output is `(N, output_size)`.

### ConvSpec, PoolSpec, ConvConfig

```cpp
struct ConvSpec {
  int out_channels = 0;
  int kernel = 0;
  int stride = 0;
  int pad = 0;
};

struct PoolSpec {
  int kernel = 0;
  int stride = 0;
};

struct ConvConfig {
  int in_channels = 0;
  int in_height = 0;
  int in_width = 0;
  std::vector<ConvSpec> convs;
  PoolSpec pool;
  std::vector<int> fc;
};
```

`in_channels`, `in_height`, and `in_width` are at least 1. `convs` and `fc` are nonempty. Each `out_channels` and each fully connected width is at least 1. `validate` walks the spatial size through every convolution and the following pool, and rejects a map that would shrink below 1×1.

`build_conv` emits, for each entry in `convs`, `Conv2d` with bias on, `ReLU`, and `MaxPool2d`. Names are `conv.i` and `pool.i`. It then appends `Flatten` and `fc` linear layers named `fc.i`, with `ReLU` between linear layers. The first linear layer sees `channels * height * width` after the last pool. `PoolSpec` is a square max pool with pad 0.

### ResNetConfig

```cpp
struct ResNetConfig {
  int in_channels = 0;
  int in_height = 0;
  int in_width = 0;
  int stem_channels = 0;
  std::vector<int> blocks_per_stage;
  std::vector<int> stage_widths;
  int output_size = 0;
};
```

`in_channels`, `in_height`, `in_width`, `stem_channels`, and `output_size` are at least 1. `blocks_per_stage` and `stage_widths` have the same nonempty length. Each stage width is at least 1. Each stage has at least one block.

`build_resnet` emits:

1. `Conv2d` 3×3, stride 1, pad 1, bias off, named `stem`.
2. `BatchNorm2d` named `stem.bn`.
3. `ReLU`.
4. One `ResidualBlock` per block, named `stages.{s}.blocks.{b}`. Stage 0 uses stride 1. The first block of every later stage uses stride 2. The other blocks use stride 1.
5. `GlobalAvgPool2d`, `Flatten`, and `Linear` named `head`.

The head's input features are the last stage width. The stem keeps the input spatial size. A projection appears when stride is not 1 or the channel count changes.

### ConvResNetConfig

```cpp
struct ConvResNetConfig {
  int in_channels = 0;
  int in_height = 0;
  int in_width = 0;
  std::vector<ConvSpec> convs;
  PoolSpec pool;
  int stem_channels = 0;
  std::vector<int> blocks_per_stage;
  std::vector<int> stage_widths;
  std::vector<int> fc;
};
```

One image network with a conv+pool trunk, residual stages, or both. `fc` is always nonempty, and every width in it is at least 1.

| Choice | Rule |
|---|---|
| Trunk | `convs` nonempty. Each stage is `Conv2d` (bias on), `ReLU`, `MaxPool2d` using `pool`. |
| Residual stem | `stem_channels > 0` inserts the 3×3 stride-1 pad-1 bias-off stem, `BatchNorm2d`, and `ReLU`. This requires residual stages. |
| Residual stages | `blocks_per_stage` nonempty and the same length as `stage_widths`. Stage 0 uses stride 1. Later stages downsample on the first block. |
| Head | `fc` linear layers named `fc.i`, with `ReLU` between them. |

At least one of `convs` and `blocks_per_stage` is nonempty. `stem_channels < 0` throws `stem_channels`. When stages are present, the builder appends `GlobalAvgPool2d` and `Flatten`, and the first linear layer sees the last stage width. When stages are absent, it appends `Flatten`, and the first linear layer sees `channels * height * width` after the trunk. `pool` is stored in the checkpoint even when `convs` is empty. `validate` checks `pool` when the trunk is present.

Names match the other builders: `conv.i`, `pool.i`, `stem`, `stem.bn`, `stages.{s}.blocks.{b}`, and `fc.i`.

Random draws follow construction order. Each `conv.i.weight` is drawn, then `stem.weight` when the stem is present, then each block's `conv1.weight`, `conv2.weight`, and `proj.weight` when that block projects, then each `fc.i.weight`.

### RnnConfig

```cpp
struct RnnConfig {
  int input_size = 0;
  std::vector<int> hidden;
  int output_size = 0;
};
```

`input_size` and `output_size` are at least 1. `hidden` is nonempty, and every width is at least 1 (`hidden`).

`build_rnn` emits one `RNN` per hidden width, named `rnn.i`, then `LastStep`, then `Linear` named `head`. Layer 0 reads `input_size`. Layer `k` reads `hidden[k - 1]`. The state before the first step is zero and is not a parameter. Each layer reads the full sequence from the layer below. `LastStep` keeps time `T - 1`. The head maps that vector to `output_size`.

Input is `(N, T, input_size)`. Output is `(N, output_size)`. `T` is at least 1. Labels for `softmax_cross_entropy` have length `N`.

For each recurrent layer, `rnn.i.weight_ih` is drawn with fan-in equal to that layer's input width, then `rnn.i.weight_hh` with fan-in equal to its hidden width, then `head.weight`.

### TransformerConfig

```cpp
struct TransformerConfig {
  int vocab_size = 0;
  int d_model = 0;
  int num_heads = 0;
  int num_layers = 0;
  int d_ff = 0;              // post-gate width H; SwiGLU fc1 writes 2H
  int max_seq_len = 0;
  float dropout = 0.f;
  bool tie_embeddings = false;
  bool bias = false;
  TransformerActivation activation = TransformerActivation::SwiGLU;
  PositionalEncoding positional = PositionalEncoding::LearnedAbsolute;
  int head_dim = 0;          // 0 keeps head_dim = d_model / num_heads
  int num_kv_heads = 0;      // 0 keeps one KV head per query head
  bool qk_norm = false;      // RMSNorm on Q and K, width head_dim, before RoPE
  float rope_theta = 10000.f;
  float rms_eps = 1e-5f;
  bool causal = true;        // false is bidirectional attention
  int type_vocab_size = 0;   // 0 omits the segment table
  bool qkv_bias = false;
  bool attn_out_bias = false;
  bool mlp_bias = false;
  bool lm_head_bias = false;
  bool split_bias = false;   // not stored. false makes the four flags follow bias
};
```

`validate` requires `vocab_size`, `d_model`, `num_heads`, `num_layers`, `d_ff`, and `max_seq_len` to be at least 1. `head_dim == 0` requires `d_model % num_heads == 0` (`num_heads`). `head_dim < 0` throws `head_dim`. `num_kv_heads == 0` means `num_heads`. Otherwise `num_kv_heads` is in `1 .. num_heads` and divides `num_heads` (`num_kv_heads`). `head_dim * num_heads` and `head_dim * num_kv_heads` must fit in a positive `int` (`head_dim`). The resolved head dimension is `head_dim` when it is positive, and `d_model / num_heads` otherwise.

`dropout` is in `[0, 1)` (`dropout`). `tie_embeddings == true` throws `tie_embeddings`. Embeddings are never tied to the output projection. `activation` is `Gelu` or `SwiGLU` (`activation`). `positional` is `LearnedAbsolute` or `Rope` (`positional`). `Rope` with an odd resolved head dimension throws `use_rope`. `rope_theta` must be finite and greater than 0 (`rope_theta`). `rms_eps` must be finite and greater than 0 (`rms_eps`). Those two checks run for every positional encoding. SwiGLU with `d_ff > INT_MAX / 2` throws `d_ff`. `type_vocab_size < 0` throws `type_vocab_size`. `0` omits the segment table. A positive value builds `type_emb`. `causal` false is bidirectional attention. The default is causal. `split_bias` false sets `qkv_bias`, `attn_out_bias`, `mlp_bias`, and `lm_head_bias` equal to `bias` before the linears are built. `split_bias` true keeps the four flags. `qkv_bias` covers Q, K, and V. `mlp_bias` covers `fc1` and `fc2`. A negative `rope_dims`, an odd or too-wide non-zero `rope_dims`, a negative `sliding_window`, `rope_scaling` above 1, a llama3 `rope_factor` that is not finite and greater than 0, or `mlm_head` on a causal model throws that field name. `validate` also rejects `gelu_tanh`, a non-zero `rope_dims` or `sliding_window`, a non-zero `rope_scaling`, `post_norm`, `embed_norm`, `final_norm` false, and `mlm_head`. Those fields are stored on a version-7 file and are not computed yet.

`rope_theta` is applied only when `positional` is `Rope`. `rms_eps` is applied to RMSNorm: the block norms and `final_ln` on the RoPE path, and QK-norm whenever `qk_norm` is true. Learned-absolute `LayerNorm` stays at epsilon `1e-5`. A `rope_theta` other than `10000` or an `rms_eps` other than `1e-5` still selects an extended checkpoint version, including on a learned-absolute model that does not apply that value.

`Family::Transformer` is `6`.

## 11. Builders and Network

Header: `nn/builders.hpp`, `nn/network.hpp`

```cpp
enum class Family : uint32_t {
  Mlp = 1, Conv = 2, ResNet = 3, ConvResNet = 4, Rnn = 5, Transformer = 6
};

Network build_mlp(Context& ctx, const MlpConfig& config, uint32_t seed);
Network build_conv(Context& ctx, const ConvConfig& config, uint32_t seed);
Network build_resnet(Context& ctx, const ResNetConfig& config, uint32_t seed);
Network build_conv_resnet(Context& ctx, const ConvResNetConfig& config, uint32_t seed);
Network build_rnn(Context& ctx, const RnnConfig& config, uint32_t seed);
Network build_transformer_network(Context& ctx, const TransformerConfig& config, uint32_t seed);

class Network {
 public:
  Network(Context& ctx, Family family, std::vector<std::unique_ptr<Layer>> layers,
          std::variant<MlpConfig, ConvConfig, ResNetConfig, ConvResNetConfig, RnnConfig,
                       TransformerConfig> config);

  const Tensor& forward(const Tensor& input);
  const Tensor& backward(const Tensor& grad_output);
  void zero_grad();
  void train();
  void eval();
  bool training() const;
  Family family() const;
  const std::variant<MlpConfig, ConvConfig, ResNetConfig, ConvResNetConfig, RnnConfig,
                     TransformerConfig>& config() const;

  std::vector<NamedParam> parameters() const;
  std::vector<NamedBuffer> buffers() const;

  void save(const std::filesystem::path& path, const Sgd* optimizer) const;
  void save(const std::filesystem::path& path, const AdamW* optimizer) const;
  void save(const std::filesystem::path& path, std::nullptr_t) const;

  struct Loaded {
    Network network;
    std::optional<Sgd> optimizer;
    std::optional<AdamW> adamw;
    uint64_t step = 0;
  };
  static Loaded load(Context& ctx, const std::filesystem::path& path, LoadMode mode);
};
```

`save(path, nullptr)` calls the `Sgd*` overload with a null pointer.

Each builder validates the config, seeds one `std::mt19937` with `seed`, and returns a network in train mode. `load` rebuilds with seed 0 and then overwrites every parameter from the file. The seed used at load time does not change the restored weights.

`forward` runs the layers in order and returns the last layer's output. `backward` runs them in reverse and returns the gradient of the network input. An empty layer list throws `empty network`.

`train()` and `eval()` set the network flag and call `set_training` on every layer. `training()` reports that flag. `zero_grad()` zeros every parameter gradient.

`config()` returns the variant stored at construction. Read it with `std::get`.

The public constructor is for a caller who already owns the layers. `save` writes `family` and `config`. `load` rebuilds with the builder for that family, then requires the parameter and batch-norm records to match that builder's names and shapes. A checkpoint round-trip is defined for a network produced by one of the six builders.

`build_transformer_network` emits `num_layers` blocks named `blocks.i`, then `final_ln`, then `Linear` named `lm_head` from `d_model` to `vocab_size`. The trunk does not own token embeddings, a position table, or a segment table. Each block sets `use_rope` from `positional == Rope`, and copies `head_dim`, `num_kv_heads`, `qk_norm`, `rope_theta`, `rms_eps`, and `causal`. The final norm is `RMSNorm` with `rms_eps` on the RoPE path and `LayerNorm` with epsilon `1e-5` otherwise. `lm_head` uses `lm_head_bias` when `split_bias` is set, and `bias` otherwise. `max_seq_len` is stored on the config and in the checkpoint. The trunk forward does not check `T` against `max_seq_len`.

Trunk input is a float tensor `(N, T, d_model)`. Trunk output is `(N, T, vocab_size)`.

Random draws for the trunk follow construction order. For each block, `blocks.i.attn.q.weight`, `k.weight`, `v.weight`, and `out.weight` are drawn, then `blocks.i.fc1.weight` and `blocks.i.fc2.weight`. `lm_head.weight` is drawn last. Norm scales are filled with 1 and do not consume the generator. The same seed passed to `Transformer` does not reproduce these trunk weights, because the façade draws embeddings from that generator first.

## 12. ConvResNet

Header: `nn/conv_resnet.hpp`

```cpp
class ConvResNet {
 public:
  ConvResNet(Context& ctx, const ConvResNetConfig& config, uint32_t seed);
  explicit ConvResNet(Network network);

  Network& network();
  const Network& network() const;
  const Tensor& forward(const Tensor& input);
  const Tensor& backward(const Tensor& grad_output);
  void zero_grad();
  void train();
  void eval();
  bool training() const;
  Family family() const;
  const ConvResNetConfig& config() const;
  std::vector<NamedParam> parameters() const;
  std::vector<NamedBuffer> buffers() const;
  void save(const std::filesystem::path& path, const Sgd* optimizer) const;
  void save(const std::filesystem::path& path, const AdamW* optimizer) const;
  struct Loaded {
    ConvResNet model;
    std::optional<Sgd> optimizer;
    std::optional<AdamW> adamw;
    uint64_t step = 0;
  };
  static Loaded load(Context& ctx, const std::filesystem::path& path, LoadMode mode);
};
```

`ConvResNet` owns one `Network` of family `ConvResNet`. The config constructor calls `build_conv_resnet`. The `Network` constructor moves the network in and throws `ConvResNet family` when `family()` is anything else.

`forward`, `backward`, `zero_grad`, `train`, `eval`, `parameters`, `buffers`, and `save` forward to that network. `config()` returns the `ConvResNetConfig`. `load` uses `Network::load`, then requires family `ConvResNet`. A file of another family throws `bad checkpoint family`.

## 13. Transformer

Header: `nn/transformer.hpp`

```cpp
class Transformer {
 public:
  Transformer(Context& ctx, const TransformerConfig& config, uint32_t seed);

  const Tensor& forward(const IndexTensor& tokens);
  const Tensor& forward(const IndexTensor& tokens, const IndexTensor& type_ids);
  const Tensor& forward(const Tensor& embeddings);
  const Tensor& backward(const Tensor& grad_logits);

  struct GenerateState {
    std::vector<MultiHeadAttention::Cache> layer_caches;
    int t = 0;  // next absolute position; tokens consumed so far
  };

  void reset_generate_state(GenerateState& st, int batch, int t_max);
  const Tensor& decode_step(const IndexTensor& next_ids, GenerateState& st);
  const Tensor& prefill(const IndexTensor& prompt_ids, GenerateState& st, int t_max);

  void zero_grad();
  void train();
  void eval();
  bool training() const;
  std::vector<NamedParam> parameters() const;
  const Network& network() const;

  void save(const std::filesystem::path& path, const Sgd* optimizer) const;
  void save(const std::filesystem::path& path, const AdamW* optimizer) const;
  void save(const std::filesystem::path& path, std::nullptr_t) const;

  struct Loaded {
    Transformer model;
    std::optional<Sgd> optimizer;
    uint64_t step = 0;
    std::optional<AdamW> adamw;
  };
  static Loaded load(Context& ctx, const std::filesystem::path& path, LoadMode mode);
};
```

`save(path, nullptr)` calls the `Sgd*` overload with a null pointer.

`Transformer` owns `tok_emb` outside the trunk, `pos_emb` when `positional` is `LearnedAbsolute`, and `type_emb` when `type_vocab_size > 0`. RoPE leaves `pos_emb` null and does not add a position vector. The trunk is a family-6 `Network`: `blocks.i`, `final_ln`, and `lm_head`. Those embeddings are not layers of `network()`.

Construction draws from one `std::mt19937` seeded with `seed`, in order: `tok_emb`, then `pos_emb` when it exists, then `type_emb` when it exists, then the trunk draws of section 11. Embedding tables use `normal_` with mean 0 and standard deviation 0.02. `pos_emb` has `max_seq_len` rows. `type_emb` has `type_vocab_size` rows. All three tables have width `d_model`.

`parameters()` is `tok_emb.weight`, then `pos_emb.weight` when that embedding exists, then `type_emb.weight` when `type_vocab_size > 0`, then `network().parameters()`. `network().parameters()` does not include those embedding tables. A RoPE parameter list has no `pos_emb.weight` and no `.bias` on `ln1`, `ln2`, or `final_ln`. A QK-norm list adds `blocks.i.attn.q_norm.weight` and `blocks.i.attn.k_norm.weight` after that block's `out` parameters.

`forward(const IndexTensor&)` reads token ids of shape `(N, T)` and returns logits `(N, T, vocab_size)`. Rank other than 2 throws `std::invalid_argument` with message `tokens`. `T > max_seq_len` throws `max_seq_len` before either embedding runs. Learned-absolute positions are the column index `0 .. T-1` on every row, added to the token embedding. RoPE forwards the token embedding alone and rotates Q and K inside each block. When `type_vocab_size > 0`, that call adds segment id 0 at every position. `forward(tokens, type_ids)` requires the segment table and a `type_ids` tensor of the same shape, or it throws `type_ids`. `forward(const Tensor&)` reads a float tensor `(N, T, d_model)` and skips the embeddings. A rank or width mismatch throws `std::invalid_argument` with message `embeddings`. Backward after that hook does not update embedding gradients. Full `forward` does not fill the KV cache. Backward of a token forward updates `tok_emb`, `pos_emb` when it exists, and `type_emb` when it exists.

`reset_generate_state` builds one cache per block and sets `st.t = 0`. `decode_step` and `prefill` throw `std::invalid_argument` with message `causal` when `causal` is false, before any other check. `decode_step` requires `eval()`. `training()` throws `decode_step`. A causal model that also has a segment table adds segment id 0 on the decode step. `next_ids` is `(N, 1)` (`next_ids`) and `N` must match the cache batch (`cache`). `st.t >= max_seq_len` throws `max_seq_len` before the position upload and before the cache write. Learned-absolute positions are the index `st.t` on every row. RoPE rotates Q and K at that same index and does not add a position vector. Each block writes its cache at that index, then `st.t` increases by 1. The return is the `lm_head` tensor `(N, 1, vocab_size)`. `prefill` resets, then runs `decode_step` once per prompt column, and returns the last step's logits. `prompt_ids` is `(N, T_prompt)` with `1 <= T_prompt <= t_max` (`prompt_ids`, `t_max`). `GenerateState` is not written by `save`.

`Transformer::load` requires family `Transformer`. Any other family throws `bad checkpoint family`. The façade file has no batch-norm records. A non-empty batch-norm section throws `checkpoint tensor mismatch: batch-norm buffers`. `Network::load` of a façade file fails on parameter count, because the trunk has no embedding tables.

## 14. Optimizers

Header: `nn/optim.hpp`, `nn/checkpoint.hpp`

```cpp
enum class LoadMode { Inference, Resume };

class Sgd {
 public:
  Sgd(std::vector<NamedParam> params, float learning_rate, float momentum);
  void step();
  void set_learning_rate(float learning_rate);
  float learning_rate() const;
  float momentum() const;
  uint64_t step_count() const;
  void set_step_count(uint64_t step);
  const std::vector<Tensor>& velocity() const;
  std::vector<Tensor>& velocity();
  const std::vector<NamedParam>& params() const;
};

struct AdamWGroup {
  std::vector<NamedParam> params;
  float weight_decay = 0.f;
};

class AdamW {
 public:
  AdamW(std::vector<AdamWGroup> groups, float lr, float beta1, float beta2, float eps);
  AdamW(std::vector<NamedParam> params, float lr, float beta1, float beta2, float eps,
        float weight_decay);
  void step();
  void set_learning_rate(float lr);
  float learning_rate() const;
  float beta1() const;
  float beta2() const;
  float eps() const;
  uint64_t step_count() const;
  void set_step_count(uint64_t step);
  const std::vector<AdamWGroup>& groups() const;
  const std::vector<Tensor>& m() const;
  std::vector<Tensor>& m();
  const std::vector<Tensor>& v() const;
  std::vector<Tensor>& v();
};

float clip_grad_norm(const std::vector<NamedParam>& params, float max_norm);
```

Construct `Sgd` from `parameters()` after the network exists. An empty list throws `SGD has no parameters`. The optimizer copies the parameter pointers and allocates one velocity tensor per parameter on the first parameter's context. Velocity starts at zero. `step_count()` starts at 0. `set_learning_rate` stores the new rate with no finite check. Momentum stays at the constructor value.

`step()` applies Polyak momentum on the context queue, then increments `step_count` by one:

```
velocity = momentum * velocity + gradient
parameter = parameter - learning_rate * velocity
```

`AdamW` takes one or more groups. An empty group list throws `std::invalid_argument` with message `groups`. A group with no parameters, a null value or gradient, a gradient whose element count differs from its parameter, or the same parameter listed twice throws `params`. A non-finite learning rate, beta, epsilon, or group weight decay throws `lr`, `beta1`, `beta2`, `eps`, or `weight_decay`. The sign of weight decay is not checked. The second constructor is one group. `set_learning_rate` rejects a non-finite rate.

Moments start at zero, match each parameter's shape, and live on the first parameter's context. `m()` and `v()` follow `groups()` flattened. `step()` increments the step count, then applies decoupled AdamW:

```
m = beta1 * m + (1 - beta1) * gradient
v = beta2 * v + (1 - beta2) * gradient^2
m_hat = m / (1 - beta1^t)
v_hat = v / (1 - beta2^t)
parameter = parameter - lr * (m_hat / (sqrt(v_hat) + eps) + weight_decay * parameter)
```

`t` is the step count after the increment. Bias correction uses that power.

`clip_grad_norm` returns the L2 norm of the gradients before scaling. It scales every non-null gradient in place by `min(1, max_norm / (total_norm + 1e-6))`. A null `grad` is skipped. An empty list returns `0` and does nothing. `max_norm <= 0`, including NaN, throws `std::invalid_argument` with message `max_norm`.

Pass the same optimizer to `save`. `save` checks pointer identity, in order, against the parameters being saved. A mismatch throws `optimizer does not belong to this network`. For `Network::save` and `ConvResNet::save`, that list is `network.parameters()`. For `Transformer::save`, that list is `Transformer::parameters()`, which includes the embedding tables.

## 15. Checkpoints

Header: `nn/checkpoint.hpp` for `LoadMode`. The byte layout is produced by `Network::save`, `ConvResNet::save`, and `Transformer::save`.

`save` synchronizes, copies device state to the host, and writes `path` with a `.tmp` suffix. It `fsync`s, closes, and renames that file onto `path`. A failure removes the temporary file and leaves an existing `path` in place. Host file errors throw `checkpoint write: ` plus `strerror`.

The file is little-endian. Integers are `int32` or `uint32` unless noted. Floats are IEEE-754 binary32, stored as their bits. Strings are a `uint32` byte length, the bytes, then zero padding to a multiple of 8. A name longer than 256 bytes throws `checkpoint tensor mismatch: name`. Shapes are a `uint32` rank, that many `int32` dimensions, then padding to a multiple of 8. A checkpoint tensor rank is 1, 2, or 4. Any other rank throws `checkpoint tensor mismatch: rank`. Records pad to a multiple of 8 after the payload.

Header, 32 bytes:

| Offset | Type | Field |
|---|---|---|
| 0 | `uint32` | Magic `0x314E4E43` (`CNN1` on a little-endian disk) |
| 4 | `uint32` | Version, 1 through 8 |
| 8 | `uint32` | `Family`, 1 through 6 |
| 12 | `uint32` | `training`, 0 or 1 |
| 16 | `uint64` | Optimizer step, or 0 when no optimizer is stored |
| 24 | `uint32` | `optimizer_present`, 0 or 1 |
| 28 | `uint32` | Reserved. Must be 0 |

After the header comes the config for `family`, padded to 8, then the parameter records, then the batch-norm records, then the optimizer block when `optimizer_present` is 1.

`load` checks every record before it copies any parameter to the device. A file that fails validation leaves the caller with the exception and no half-updated network. The restored network is in the train or eval mode stored in the file.

`Loaded::step` is the step stored in the file. `LoadMode::Resume` sets `Loaded::optimizer` for an SGD block and `Loaded::adamw` for an AdamW block. At most one is set. `LoadMode::Inference` parses either optimizer block, checks it, and leaves both empty. `LoadMode::Resume` throws `optimizer block missing` when the file has no optimizer block.

### Version

A non-transformer file is version 1 from `save(path, const Sgd*)`, including a null `Sgd*`, and version 2 from `save(path, const AdamW*)`, including a null `AdamW*`.

A transformer file, whether saved from the trunk or from the façade, picks a version from the config. A projection-bias file wins over bert, and bert wins over the extended block. The projection-bias file is true when `qkv_bias`, `attn_out_bias`, `mlp_bias`, or `lm_head_bias` differs from `bias`, or when `gelu_tanh` is set, `rope_dims` is not 0, `sliding_window` is not 0, `rope_scaling` is not 0, `post_norm` is set, `embed_norm` is set, `final_norm` is false, or `mlm_head` is set. The extended block is true when any of these hold:

- `head_dim > 0` and either `d_model` is not divisible by `num_heads` or `head_dim` is not `d_model / num_heads`
- `num_kv_heads > 0` and `num_kv_heads != num_heads`
- `qk_norm` is true
- `rope_theta != 10000`
- `rms_eps != 1e-5`

Bert is true when `causal` is false or `type_vocab_size != 0`.

| Config | `Sgd*` overload, null optimizer | `Sgd*` overload, optimizer stored | `AdamW*` overload, null | `AdamW*` overload, optimizer stored |
|---|---|---|---|---|
| Classic | 1 | see below | 2 | 2 |
| Extended, not bert | 3 | see below | 4 | 4 |
| Bert | 5 | see below | 6 | 6 |
| Projection bias | 7 | see below | 7 | 8 |

On a classic, extended, or bert config, the `AdamW*` overload uses the even version, including when the pointer is null. A null `AdamW*` stores `optimizer_present` 0 and no optimizer bytes. On a projection-bias config, a null `AdamW*` writes version 7. An optimizer block on that config writes version 8.

The stored-SGD column depends on which object saves:

- `Network::save` and `ConvResNet::save` call the version helper with the kind word off. A stored SGD file stays version 1, 3, 5, or 7, and the optimizer bytes are the SGD body with no kind tag.
- `Transformer::save(const Sgd*)` calls the version helper with the kind word on when the pointer is non-null. A stored SGD file is version 2, 4, 6, or 8, and the optimizer bytes start with kind `1` followed by the SGD body.

Versions 1, 3, 5, and 7 store an SGD body with no kind tag when an optimizer is present. Versions 2, 4, 6, and 8 store a `uint32` kind when an optimizer is present: `1` for SGD, `2` for AdamW. Any other kind throws `unsupported optimizer kind N`.

Versions 1 and 2 do not store the extended fields. The loader uses `head_dim = 0`, `num_kv_heads = 0`, `qk_norm = false`, `rope_theta = 10000`, `rms_eps = 1e-5`, `causal = true`, and `type_vocab_size = 0`.

Versions 3 and 4 store the extended fields and leave `causal` true and `type_vocab_size` 0.

Versions 5 and 6 store the extended fields, then `causal`, then `type_vocab_size`.

Versions 7 and 8 store the extended fields, `causal`, `type_vocab_size`, then `qkv_bias`, `attn_out_bias`, `mlp_bias`, `lm_head_bias`, `gelu_tanh`, `rope_dims`, `sliding_window`, `rope_scaling`, `rope_factor`, `rope_low_freq_factor`, `rope_high_freq_factor`, `rope_original_context`, `post_norm`, `embed_norm`, `embed_norm_eps`, `final_norm`, and `mlm_head`. Versions 1 through 6 set the four projection flags equal to `bias` and leave `split_bias` false. Versions 7 and 8 store the four flags from the file and set `split_bias` true. `split_bias` is not written.

### Config bodies

Each config body ends with padding to 8 bytes. The transformer body is the only one that grows with the version.

`Mlp`: `input_size`, `output_size`, `activation` as `uint32`, hidden count, then that many hidden widths.

`Conv`: `in_channels`, `in_height`, `in_width`, `pool.kernel`, `pool.stride`, conv count, then each conv as `out_channels`, `kernel`, `stride`, `pad`, then fc count, then the fc widths.

`ResNet`: `in_channels`, `in_height`, `in_width`, `stem_channels`, `output_size`, stage count, then each stage as `blocks_per_stage`, `stage_widths`.

`ConvResNet`: `in_channels`, `in_height`, `in_width`, `stem_channels`, `pool.kernel`, `pool.stride`, conv count, the convs as in `Conv`, stage count, the stages as in `ResNet`, fc count, the fc widths.

`Rnn`: `input_size`, `output_size`, hidden count, then the hidden widths.

`Transformer`, every version: `vocab_size`, `d_model`, `num_heads`, `num_layers`, `d_ff`, `max_seq_len`, `dropout` as `float`, `tie_embeddings` as `uint32`, `bias` as `uint32`, `activation` as `uint32`, `positional` as `uint32`.

Versions 3 through 8 then store: `head_dim`, `num_kv_heads`, `qk_norm` as `uint32`, `rope_theta` as `float`, `rms_eps` as `float`.

Versions 5 through 8 then store: `causal` as `uint32`, `type_vocab_size`.

Versions 7 and 8 then store the seventeen words listed above. `rope_factor`, `rope_low_freq_factor`, `rope_high_freq_factor`, and `embed_norm_eps` are `float`. `rope_dims`, `sliding_window`, and `rope_original_context` are `int32`. The other words are `uint32`.

The loader runs `validate` on the rebuilt config.

### Parameter and batch-norm records

Parameter section: count as `uint32`, a zero `uint32`, then one record per parameter in `parameters()` order.

Each parameter record: name string, shape, dtype `uint32` equal to 1 (float32), `numel` as `uint32`, that many floats, pad to 8. The product of the shape must equal `numel`. A dtype other than 1, or a product mismatch, throws `checkpoint tensor mismatch: ` plus the name. Activations are not stored in the file.

Batch-norm section: count as `uint32`, a zero `uint32`, then one record per pair. The live buffers must be ordered `running_mean`, `running_var`, repeating. The prefixes of each pair must match.

Each batch-norm record: prefix string, `batches` as `uint64`, channel count as `uint32`, a zero `uint32`, that many mean floats, that many variance floats, pad to 8.

A façade file writes a batch-norm count of 0.

### Optimizer bodies

SGD body, with no kind tag on versions 1, 3, 5, and 7, and after kind `1` on versions 2, 4, 6, and 8: `learning_rate` as `float`, `momentum` as `float`, velocity count as `uint32`, a zero `uint32`, then each velocity as `numel` `uint32`, a zero `uint32`, that many floats, pad to 8. The count equals the parameter count. Each `numel` equals that parameter's element count.

AdamW body, after kind `2`: `lr`, `beta1`, `beta2`, `eps` as `float`, group count as `uint32`, a zero `uint32`, then each group. A group is `weight_decay` as `float`, parameter count as `uint32`, then each parameter's `m` and `v`. A moment is `numel` `uint32`, a zero `uint32`, that many floats, pad to 8. The flattened groups equal `parameters()` in order. A zero group, a count that walks off the parameter list, or a length mismatch throws `optimizer state mismatch`.

### Load errors

| Condition | Message |
|---|---|
| Unreadable, shorter than 32 bytes, or cut off mid-record | `truncated checkpoint` |
| Wrong magic | `bad checkpoint magic` |
| Version outside 1..8 | `unsupported checkpoint version N` |
| Kind other than 1 or 2 on a version that has a kind tag | `unsupported optimizer kind N` |
| Bad train flag, optimizer flag, or reserved word | `bad checkpoint header` |
| Family outside 1..6 | `bad checkpoint family` |
| `ConvResNet::load` or `Transformer::load` of another family | `bad checkpoint family` |
| Parameter or batch-norm record disagrees with the rebuilt network | `checkpoint tensor mismatch: ...` |
| `Resume` and the optimizer flag is 0 | `optimizer block missing` |
| Optimizer record count disagrees | `optimizer state mismatch` |
| `save` optimizer belongs to other parameters | `optimizer does not belong to this network` |

The file holds the config, the parameters, the batch-norm running statistics, and, when requested, SGD velocity or AdamW moments. It leaves out activation caches, gradients, workspaces, backend descriptors, and `GenerateState`.

## 16. Initialization

Header: `nn/init.hpp`

```cpp
void kaiming_normal(Tensor& weight, int fan_in, std::mt19937& rng);
void normal_(Tensor& tensor, float mean, float std, std::mt19937& rng);
```

`kaiming_normal` fills `weight` with a normal distribution of mean 0 and standard deviation `sqrt(2 / fan_in)`, copies to the device, then synchronizes. `fan_in` is at least 1 (`kaiming fan_in`). `Linear` uses `in_features`. `Conv2d` uses `in_channels * kernel * kernel`. `RNN` uses `input_size` for `weight_ih` and `hidden_size` for `weight_hh`.

`normal_` fills on the host from a normal distribution with the given mean and standard deviation, copies to the device, then synchronizes. `std` must be positive and finite (`normal_ std`). `mean` must be finite (`normal_ mean`). `Embedding` calls this with mean 0 and standard deviation 0.02.

Both throw `nn::Error`.
