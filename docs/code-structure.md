# Code structure

This tree is the OpenCL implementation of namespace `nn`. Public headers live under `include/nn/` and do not include OpenCL headers. The static library target is `nn`. The call contract is [api.md](api.md).

`src/checkpoint.cpp`, `include/nn/transformer.hpp`, and `include/nn/layer.hpp` are the same bytes in `nnCUDA`, `nnCL`, `nnROCm`, and `nnArc`. `include/nn/context.hpp` differs in comments and in how a device is opened. `nnArc` has no `include/nn/dtype.hpp` and no `Context::set_activation_dtype`. This tree's `Context` indexes the vendor list from `NNCL_OPENCL` and does not call `cudaSetDevice`.

## Where a call lands

```mermaid
graph TD
    Caller["caller, including llm"] --> Headers["include/nn"]
    Headers --> Src["src/*.cpp"]
    Src --> CL["OpenCL runtime"]
    Src --> Kernels["src/kernels.cl<br/>read at runtime"]

    subgraph Public["public headers"]
        Core["context, tensor, dtype, error"]
        Layers["include/nn/layers"]
        Trunk["network, transformer, builders"]
        Opt["optim, loss, checkpoint"]
    end
```

`include/` is on the public include path. `src/` is private. `kernels.cl` is not a translation unit. The build defines `NNCL_KERNELS_FILE` as its path, and `ocl.cpp` reads that file when a context opens.

## Public headers

| Header | Types |
|---|---|
| `nn/error.hpp` | `Error`. The class is inline. Status names are built in `src/ocl.cpp`. |
| `nn/dtype.hpp` | `DType::Float32`, `DType::BFloat16` |
| `nn/context.hpp` | `Context`, `DeviceInfo`, `detect_devices`, `select_device` |
| `nn/tensor.hpp` | `Tensor` |
| `nn/index_tensor.hpp` | `IndexTensor`, `DeviceIndices` |
| `nn/layer.hpp` | `Layer`, `NamedParam`, `NamedBuffer` |
| `nn/layers/*.hpp` | One class per file: `Linear`, `Embedding`, `ReLU`, `GELU`, `SwiGLU`, `Dropout`, `LayerNorm`, `RMSNorm`, `MultiHeadAttention`, `TransformerBlock`, `RNN`, `LastStep`, `Conv2d`, `MaxPool2d`, `BatchNorm2d`, `Flatten`, `GlobalAvgPool2d`, `ResidualBlock` |
| `nn/config.hpp` | Config structs and `validate` |
| `nn/network.hpp` | `Network`, `Family` |
| `nn/transformer.hpp` | `Transformer`, `Transformer::Loaded`, `GenerateState` |
| `nn/conv_resnet.hpp` | `ConvResNet` |
| `nn/builders.hpp` | `build_mlp`, `build_conv`, `build_resnet`, `build_conv_resnet`, `build_rnn`, `build_transformer_network` |
| `nn/optim.hpp` | `Sgd`, `AdamW`, `AdamWGroup` |
| `nn/loss.hpp` | `LossResult`, `softmax_cross_entropy`, `sequence_cross_entropy` |
| `nn/checkpoint.hpp` | `LoadMode` and the checkpoint helpers `Network::save` uses |
| `nn/init.hpp` | `kaiming_normal` |

`Family` is `Mlp` 1, `Conv` 2, `ResNet` 3, `ConvResNet` 4, `Rnn` 5, and `Transformer` 6. `build_transformer_network` returns a `Network` of that family. `llm` constructs `Transformer` directly. That object owns the embeddings and holds the `Network` for the blocks, the final norm, and the LM head.

## Transformer ownership

Embeddings sit outside the `Network`. Block pointers are captured before each block is moved into the network, so `GenerateState` can call `TransformerBlock::forward_cached` on the same objects.

```mermaid
graph TD
    T["Transformer"] --> Tok["Embedding tok_emb"]
    T --> Pos["Embedding pos_emb<br/>learned positions"]
    T --> Typ["Embedding type_emb<br/>type_vocab_size greater than 0"]
    T --> Net["Network Family::Transformer"]

    Net --> Blocks["TransformerBlock × num_layers"]
    Net --> Final["final_ln<br/>RMSNorm or LayerNorm"]
    Net --> Head["Linear lm_head"]

    Blocks --> LN1["ln1"]
    Blocks --> MHA["MultiHeadAttention"]
    Blocks --> Drop1["Dropout"]
    Blocks --> LN2["ln2"]
    Blocks --> FC1["Linear fc1"]
    Blocks --> Act["SwiGLU or GELU"]
    Blocks --> FC2["Linear fc2"]
    Blocks --> Drop2["Dropout"]

    MHA --> Q["Linear q, k, v, out"]
    MHA --> QN["RMSNorm q_norm, k_norm<br/>when qk_norm is set"]
```

`pos_emb` is absent when `positional` is RoPE. `ln1` and `ln2` are `RMSNorm` in that case and `LayerNorm` otherwise. `fc1` writes `2 * d_ff` for SwiGLU and `d_ff` for GELU. `q_norm` and `k_norm` exist when `qk_norm` is set. `forward` and `backward` return a `const Tensor&` into storage owned by that layer.

`Context` owns one OpenCL command queue. Tensors, layers, and optimizers borrow it. `Tensor` is the device-buffer owner. `NamedParam` holds pointers into those tensors. `Sgd` and `AdamW` copy the pointer list and own their moment tensors. Both optimizers are implemented in `src/optim.cpp`.

## Sources in this tree

`CMakeLists.txt` compiles every `src/*.cpp` listed here into `libnn.a`. Kernel launches go through `ocl.cpp`.

| Group | Files |
|---|---|
| Device | `context.cpp`, `ocl.cpp`, `tensor.cpp`, `index_tensor.cpp` |
| Layers | `linear.cpp`, `embedding.cpp`, `relu.cpp`, `gelu.cpp`, `swiglu.cpp`, `dropout.cpp`, `layer_norm.cpp`, `rms_norm.cpp`, `multi_head_attention.cpp`, `rnn.cpp`, `last_step.cpp`, `conv2d.cpp`, `max_pool2d.cpp`, `batch_norm2d.cpp`, `flatten.cpp`, `global_avg_pool2d.cpp`, `residual_block.cpp` |
| Trunk | `transformer_block.cpp`, `transformer.cpp`, `network.cpp`, `builders.cpp`, `config.cpp`, `init.cpp` |
| Optim and loss | `optim.cpp`, `loss.cpp` |
| Checkpoint | `checkpoint.cpp` |
| Kernels | `kernels.cl`, loaded by path |

Private headers stay in `src/`: `activate.hpp`, `ensure.hpp`, `ocl.hpp`. `Context` keeps the OpenCL handles behind `Context::Impl` in `ocl.cpp`.

## Tests

`tests/test_main.cpp` is `nn_tests`. The argument selects `shapes`, `grads`, `ckpt`, `mnist`, or `bf16`.

| File | Role |
|---|---|
| `test_shapes.cpp` | Shapes of the layers and the trunk |
| `test_gradients.cpp` | Analytic gradients against numeric gradients |
| `test_checkpoint.cpp` | Save and load |
| `test_bf16.cpp` | Bfloat16 activations |
| `test_mnist.cpp` | MNIST for the MLP, conv, ResNet, and ConvResNet builders |
| `mnist.cpp` | IDX reader used by the MNIST test |

## Sibling trees

| Tree | `llm` marker | Device code | Activations |
|---|---|---|---|
| `nnCUDA` | no marker file | `.cu` plus cuBLAS and cuDNN | bfloat16 by default |
| `nnCL` | `src/kernels.cl` | `kernels.cl`, loaded at runtime | bfloat16 by default |
| `nnROCm` | `src/elementwise.hip` | `.hip` plus hipBLAS and rocBLAS | bfloat16 by default |
| `nnArc` | `src/elementwise.cpp` | SYCL C++ and oneMKL | float32, no `dtype.hpp` |

`llm` is the sibling that calls `Transformer`, `sequence_cross_entropy`, and `AdamW`. Its layout is `docs/code-structure.md` in that repository.
