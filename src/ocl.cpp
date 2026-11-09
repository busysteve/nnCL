#include "ocl.hpp"

#include "nn/error.hpp"

#include <fstream>
#include <sstream>
#include <vector>

namespace nn {
namespace {

const char* cl_error_name(cl_int status) {
  switch (status) {
    case CL_SUCCESS:
      return "CL_SUCCESS";
    case CL_DEVICE_NOT_FOUND:
      return "CL_DEVICE_NOT_FOUND";
    case CL_DEVICE_NOT_AVAILABLE:
      return "CL_DEVICE_NOT_AVAILABLE";
    case CL_COMPILER_NOT_AVAILABLE:
      return "CL_COMPILER_NOT_AVAILABLE";
    case CL_MEM_OBJECT_ALLOCATION_FAILURE:
      return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
    case CL_OUT_OF_RESOURCES:
      return "CL_OUT_OF_RESOURCES";
    case CL_OUT_OF_HOST_MEMORY:
      return "CL_OUT_OF_HOST_MEMORY";
    case CL_BUILD_PROGRAM_FAILURE:
      return "CL_BUILD_PROGRAM_FAILURE";
    case CL_INVALID_VALUE:
      return "CL_INVALID_VALUE";
    case CL_INVALID_DEVICE_TYPE:
      return "CL_INVALID_DEVICE_TYPE";
    case CL_INVALID_PLATFORM:
      return "CL_INVALID_PLATFORM";
    case CL_INVALID_DEVICE:
      return "CL_INVALID_DEVICE";
    case CL_INVALID_CONTEXT:
      return "CL_INVALID_CONTEXT";
    case CL_INVALID_QUEUE_PROPERTIES:
      return "CL_INVALID_QUEUE_PROPERTIES";
    case CL_INVALID_COMMAND_QUEUE:
      return "CL_INVALID_COMMAND_QUEUE";
    case CL_INVALID_HOST_PTR:
      return "CL_INVALID_HOST_PTR";
    case CL_INVALID_MEM_OBJECT:
      return "CL_INVALID_MEM_OBJECT";
    case CL_INVALID_BINARY:
      return "CL_INVALID_BINARY";
    case CL_INVALID_BUILD_OPTIONS:
      return "CL_INVALID_BUILD_OPTIONS";
    case CL_INVALID_PROGRAM:
      return "CL_INVALID_PROGRAM";
    case CL_INVALID_PROGRAM_EXECUTABLE:
      return "CL_INVALID_PROGRAM_EXECUTABLE";
    case CL_INVALID_KERNEL_NAME:
      return "CL_INVALID_KERNEL_NAME";
    case CL_INVALID_KERNEL_DEFINITION:
      return "CL_INVALID_KERNEL_DEFINITION";
    case CL_INVALID_KERNEL:
      return "CL_INVALID_KERNEL";
    case CL_INVALID_ARG_INDEX:
      return "CL_INVALID_ARG_INDEX";
    case CL_INVALID_ARG_VALUE:
      return "CL_INVALID_ARG_VALUE";
    case CL_INVALID_ARG_SIZE:
      return "CL_INVALID_ARG_SIZE";
    case CL_INVALID_KERNEL_ARGS:
      return "CL_INVALID_KERNEL_ARGS";
    case CL_INVALID_WORK_DIMENSION:
      return "CL_INVALID_WORK_DIMENSION";
    case CL_INVALID_WORK_GROUP_SIZE:
      return "CL_INVALID_WORK_GROUP_SIZE";
    case CL_INVALID_WORK_ITEM_SIZE:
      return "CL_INVALID_WORK_ITEM_SIZE";
    case CL_INVALID_GLOBAL_OFFSET:
      return "CL_INVALID_GLOBAL_OFFSET";
    case CL_INVALID_EVENT_WAIT_LIST:
      return "CL_INVALID_EVENT_WAIT_LIST";
    case CL_INVALID_EVENT:
      return "CL_INVALID_EVENT";
    case CL_INVALID_OPERATION:
      return "CL_INVALID_OPERATION";
    case CL_INVALID_BUFFER_SIZE:
      return "CL_INVALID_BUFFER_SIZE";
    case CL_INVALID_GLOBAL_WORK_SIZE:
      return "CL_INVALID_GLOBAL_WORK_SIZE";
    case CL_INVALID_PROPERTY:
      return "CL_INVALID_PROPERTY";
    default:
      return "CL_UNKNOWN";
  }
}

size_t round_up(size_t n, size_t local) { return ((n + local - 1) / local) * local; }

class KernelCall {
 public:
  KernelCall(Context& ctx, const char* name) : queue_(ContextAccess::get(ctx).queue) {
    auto& impl = ContextAccess::get(ctx);
    const auto found = impl.kernels.find(name);
    if (found != impl.kernels.end()) {
      kernel_ = found->second;
      return;
    }
    cl_int err = CL_SUCCESS;
    kernel_ = clCreateKernel(impl.program, name, &err);
    check_cl(err, "clCreateKernel");
    impl.kernels.emplace(name, kernel_);
  }

  KernelCall& mem(cl_mem value) { return set(sizeof(value), &value); }
  KernelCall& i(cl_int value) { return set(sizeof(value), &value); }
  KernelCall& l(cl_long value) { return set(sizeof(value), &value); }
  KernelCall& f(cl_float value) { return set(sizeof(value), &value); }
  KernelCall& local(size_t bytes) {
    check_cl(clSetKernelArg(kernel_, arg_++, bytes, nullptr), "clSetKernelArg");
    return *this;
  }

  void over(int64_t n, int local_size) {
    if (n <= 0) return;
    const size_t local = static_cast<size_t>(local_size);
    const size_t global = round_up(static_cast<size_t>(n), local);
    nd(&global, &local, 1);
  }

  void groups(int n_groups, int local_size) {
    if (n_groups <= 0) return;
    const size_t local = static_cast<size_t>(local_size);
    const size_t global = static_cast<size_t>(n_groups) * local;
    nd(&global, &local, 1);
  }

  void nd(const size_t* global, const size_t* local, cl_uint dim) {
    check_cl(clEnqueueNDRangeKernel(queue_, kernel_, dim, nullptr, global, local, 0, nullptr,
                                    nullptr),
             "clEnqueueNDRangeKernel");
  }

 private:
  KernelCall& set(size_t bytes, const void* value) {
    check_cl(clSetKernelArg(kernel_, arg_++, bytes, value), "clSetKernelArg");
    return *this;
  }

  cl_kernel kernel_ = nullptr;
  cl_command_queue queue_ = nullptr;
  cl_uint arg_ = 0;
};

cl_command_queue queue_of(Context& ctx) { return ContextAccess::get(ctx).queue; }

void enqueue_gemm(Context& ctx, const char* name, float* C, const float* A, const float* B, int M,
                  int N, int K, int batches, int ldc, int lda, int ldb, int64_t stride_c,
                  int64_t stride_a, int64_t stride_b, float alpha, float beta) {
  if (M <= 0 || N <= 0 || batches <= 0) return;
  const auto& impl = ContextAccess::get(ctx);
  KernelCall call(ctx, name);
  call.mem(mem_of(C))
      .mem(mem_of(A))
      .mem(mem_of(B))
      .i(M)
      .i(N)
      .i(K)
      .i(batches)
      .i(ldc)
      .i(lda)
      .i(ldb)
      .l(static_cast<cl_long>(stride_c))
      .l(static_cast<cl_long>(stride_a))
      .l(static_cast<cl_long>(stride_b))
      .f(alpha)
      .f(beta);
  const size_t local[3] = {impl.gemm_lx, impl.gemm_ly, 1};
  const size_t global[3] = {round_up(static_cast<size_t>(N), local[0]),
                            round_up(static_cast<size_t>(M), local[1]),
                            static_cast<size_t>(batches)};
  call.nd(global, local, 3);
}

std::string read_file(const char* path) {
  std::ifstream in(path);
  if (!in) throw Error("OpenCL program");
  std::ostringstream buffer;
  buffer << in.rdbuf();
  return buffer.str();
}

}  // namespace

void check_cl(cl_int status, const char* call) {
  if (status == CL_SUCCESS) return;
  throw Error(std::string(call) + ": " + cl_error_name(status));
}

void Context::Impl::open() {
  cl_uint nplat = 0;
  cl_int err = clGetPlatformIDs(0, nullptr, &nplat);
  if (err != CL_SUCCESS || nplat == 0) throw Error("OpenCL device");
  std::vector<cl_platform_id> platforms(nplat);
  check_cl(clGetPlatformIDs(nplat, platforms.data(), nullptr), "clGetPlatformIDs");
  for (cl_platform_id platform : platforms) {
    cl_uint ndev = 0;
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &ndev);
    if (err != CL_SUCCESS || ndev == 0) continue;
    std::vector<cl_device_id> devices(ndev);
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, ndev, devices.data(), nullptr);
    if (err != CL_SUCCESS) continue;
    for (cl_device_id candidate : devices) {
      cl_context trial = clCreateContext(nullptr, 1, &candidate, nullptr, nullptr, &err);
      if (err != CL_SUCCESS || !trial) continue;
      size_t max_wg = 0;
      clGetDeviceInfo(candidate, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(max_wg), &max_wg, nullptr);
      if (max_wg < 256) {
        clReleaseContext(trial);
        continue;
      }
      context = trial;
      device = candidate;
      break;
    }
    if (context) break;
  }
  if (!context) throw Error("OpenCL device");

  cl_device_fp_config fp_config = 0;
  clGetDeviceInfo(device, CL_DEVICE_DOUBLE_FP_CONFIG, sizeof(fp_config), &fp_config, nullptr);
  fp64 = fp_config != 0;
  size_t item_sizes[3] = {1, 1, 1};
  clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(item_sizes), item_sizes, nullptr);
  size_t max_wg = 0;
  clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(max_wg), &max_wg, nullptr);
  if (max_wg >= 256 && item_sizes[0] >= 16 && item_sizes[1] >= 16) {
    gemm_lx = 16;
    gemm_ly = 16;
  } else if (max_wg >= 64 && item_sizes[0] >= 8 && item_sizes[1] >= 8) {
    gemm_lx = 8;
    gemm_ly = 8;
  }

  const std::string source = read_file(NNCL_KERNELS_FILE);
  const char* text = source.c_str();
  const size_t length = source.size();
  program = clCreateProgramWithSource(context, 1, &text, &length, &err);
  check_cl(err, "clCreateProgramWithSource");
  const char* options = fp64 ? "-cl-std=CL2.0 -DHAVE_FP64=1" : "-cl-std=CL2.0";
  err = clBuildProgram(program, 1, &device, options, nullptr, nullptr);
  if (err != CL_SUCCESS) {
    size_t log_size = 0;
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
    std::string log(log_size == 0 ? 0 : log_size - 1, '\0');
    if (log_size > 1) {
      clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
    }
    if (log.size() > 2000) log.resize(2000);
    throw Error(std::string("clBuildProgram: ") + cl_error_name(err) + " " + log);
  }

  queue = clCreateCommandQueueWithProperties(context, device, nullptr, &err);
  check_cl(err, "clCreateCommandQueueWithProperties");
  dummy = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float), nullptr, &err);
  check_cl(err, "clCreateBuffer");
  const float zero = 0.f;
  check_cl(clEnqueueFillBuffer(queue, dummy, &zero, sizeof(zero), 0, sizeof(zero), 0, nullptr,
                               nullptr),
           "clEnqueueFillBuffer");
}

Context::Impl::~Impl() {
  if (queue) clFinish(queue);
  for (auto& entry : kernels) {
    if (entry.second) clReleaseKernel(entry.second);
  }
  kernels.clear();
  if (dummy) clReleaseMemObject(dummy);
  if (queue) clReleaseCommandQueue(queue);
  if (program) clReleaseProgram(program);
  if (context) clReleaseContext(context);
  queue = nullptr;
  program = nullptr;
  context = nullptr;
  dummy = nullptr;
}

cl_mem alloc_mem(Context& ctx, size_t bytes) {
  if (bytes == 0) bytes = 4;
  cl_int err = CL_SUCCESS;
  cl_mem mem = clCreateBuffer(ContextAccess::get(ctx).context, CL_MEM_READ_WRITE, bytes, nullptr, &err);
  check_cl(err, "clCreateBuffer");
  return mem;
}

void free_mem(cl_mem mem) noexcept {
  if (mem) clReleaseMemObject(mem);
}

void write_mem(Context& ctx, cl_mem mem, const void* src, size_t bytes) {
  if (bytes == 0) return;
  // Blocking: callers pass temporaries. CUDA's pageable cudaMemcpyAsync copies before return.
  check_cl(clEnqueueWriteBuffer(queue_of(ctx), mem, CL_TRUE, 0, bytes, src, 0, nullptr, nullptr),
           "clEnqueueWriteBuffer");
}

void read_mem(Context& ctx, cl_mem mem, void* dst, size_t bytes) {
  if (bytes == 0) return;
  check_cl(clEnqueueReadBuffer(queue_of(ctx), mem, CL_TRUE, 0, bytes, dst, 0, nullptr, nullptr),
           "clEnqueueReadBuffer");
}

void fill_mem_f(Context& ctx, cl_mem mem, float value, size_t count) {
  if (count == 0) return;
  check_cl(clEnqueueFillBuffer(queue_of(ctx), mem, &value, sizeof(value), 0, count * sizeof(value),
                               0, nullptr, nullptr),
           "clEnqueueFillBuffer");
}

void fill_mem_i(Context& ctx, cl_mem mem, int value, size_t count) {
  if (count == 0) return;
  check_cl(clEnqueueFillBuffer(queue_of(ctx), mem, &value, sizeof(value), 0, count * sizeof(value),
                               0, nullptr, nullptr),
           "clEnqueueFillBuffer");
}

void copy_mem(Context& ctx, cl_mem dst, cl_mem src, size_t bytes) {
  if (bytes == 0 || dst == src) return;
  check_cl(clEnqueueCopyBuffer(queue_of(ctx), src, dst, 0, 0, bytes, 0, nullptr, nullptr),
           "clEnqueueCopyBuffer");
}

void launch_scale(Context& ctx, float* x, int64_t n, float factor) {
  KernelCall(ctx, "scale_f").mem(mem_of(x)).l(static_cast<cl_long>(n)).f(factor).over(n, 256);
}

void launch_add_bias(Context& ctx, float* y, const float* bias, int rows, int cols) {
  const int64_t n = static_cast<int64_t>(rows) * cols;
  KernelCall(ctx, "add_bias").mem(mem_of(y)).mem(mem_of(bias)).i(rows).i(cols).over(n, 256);
}

void launch_bias_grad(Context& ctx, const float* grad_y, float* grad_bias, int rows, int cols) {
  KernelCall call(ctx, "bias_grad");
  call.mem(mem_of(grad_y)).mem(mem_of(grad_bias)).i(rows).i(cols).local(256 * sizeof(float));
  call.groups(cols, 256);
}

void launch_add2(Context& ctx, float* dst, const float* a, const float* b, int64_t n) {
  KernelCall(ctx, "add2")
      .mem(mem_of(dst))
      .mem(mem_of(a))
      .mem(mem_of(b))
      .l(static_cast<cl_long>(n))
      .over(n, 256);
}

void launch_mul(Context& ctx, float* y, const float* a, const float* b, int64_t n) {
  KernelCall(ctx, "mul2")
      .mem(mem_of(y))
      .mem(mem_of(a))
      .mem(mem_of(b))
      .l(static_cast<cl_long>(n))
      .over(n, 256);
}

void launch_gelu(Context& ctx, float* y, const float* x, int64_t n) {
  KernelCall(ctx, "gelu_k").mem(mem_of(y)).mem(mem_of(x)).l(static_cast<cl_long>(n)).over(n, 256);
}

void launch_gelu_backward(Context& ctx, float* grad_x, const float* x, const float* grad_y,
                          int64_t n) {
  KernelCall(ctx, "gelu_backward_k")
      .mem(mem_of(grad_x))
      .mem(mem_of(x))
      .mem(mem_of(grad_y))
      .l(static_cast<cl_long>(n))
      .over(n, 256);
}

void launch_swiglu(Context& ctx, float* y, const float* x, int rows, int64_t inner) {
  const int64_t n = static_cast<int64_t>(rows) * inner;
  KernelCall(ctx, "swiglu_k")
      .mem(mem_of(y))
      .mem(mem_of(x))
      .i(rows)
      .l(static_cast<cl_long>(inner))
      .over(n, 256);
}

void launch_swiglu_backward(Context& ctx, float* grad_x, const float* x, const float* grad_y,
                            int rows, int64_t inner) {
  const int64_t n = static_cast<int64_t>(rows) * inner;
  KernelCall(ctx, "swiglu_backward_k")
      .mem(mem_of(grad_x))
      .mem(mem_of(x))
      .mem(mem_of(grad_y))
      .i(rows)
      .l(static_cast<cl_long>(inner))
      .over(n, 256);
}

void launch_tanh(Context& ctx, float* y, const float* x, int64_t n) {
  KernelCall(ctx, "tanh_k").mem(mem_of(y)).mem(mem_of(x)).l(static_cast<cl_long>(n)).over(n, 256);
}

void launch_tanh_backward(Context& ctx, float* grad_x, const float* y, const float* grad_y,
                          int64_t n) {
  KernelCall(ctx, "tanh_backward_k")
      .mem(mem_of(grad_x))
      .mem(mem_of(y))
      .mem(mem_of(grad_y))
      .l(static_cast<cl_long>(n))
      .over(n, 256);
}

void launch_relu(Context& ctx, float* y, const float* x, int64_t n) {
  KernelCall(ctx, "relu_k").mem(mem_of(y)).mem(mem_of(x)).l(static_cast<cl_long>(n)).over(n, 256);
}

void launch_relu_backward(Context& ctx, float* grad_x, const float* x, const float* grad_y,
                          int64_t n) {
  KernelCall(ctx, "relu_backward_k")
      .mem(mem_of(grad_x))
      .mem(mem_of(x))
      .mem(mem_of(grad_y))
      .l(static_cast<cl_long>(n))
      .over(n, 256);
}

void launch_gather_time(Context& ctx, float* dst, const float* src, int batch, int time,
                        int features, int index) {
  const int64_t n = static_cast<int64_t>(batch) * features;
  KernelCall(ctx, "gather_time")
      .mem(mem_of(dst))
      .mem(mem_of(src))
      .i(batch)
      .i(time)
      .i(features)
      .i(index)
      .over(n, 256);
}

void launch_scatter_time(Context& ctx, float* dst, const float* src, int batch, int time,
                         int features, int index) {
  const int64_t n = static_cast<int64_t>(batch) * features;
  KernelCall(ctx, "scatter_time")
      .mem(mem_of(dst))
      .mem(mem_of(src))
      .i(batch)
      .i(time)
      .i(features)
      .i(index)
      .over(n, 256);
}

void launch_sgd(Context& ctx, float* param, float* velocity, const float* grad, int64_t n, float lr,
                float momentum) {
  KernelCall(ctx, "sgd_k")
      .mem(mem_of(param))
      .mem(mem_of(velocity))
      .mem(mem_of(grad))
      .l(static_cast<cl_long>(n))
      .f(lr)
      .f(momentum)
      .over(n, 256);
}

void launch_adamw(Context& ctx, float* param, float* m, float* v, const float* grad, int64_t n,
                  float lr, float beta1, float beta2, float eps, float weight_decay, float bc1,
                  float bc2) {
  KernelCall(ctx, "adamw_k")
      .mem(mem_of(param))
      .mem(mem_of(m))
      .mem(mem_of(v))
      .mem(mem_of(grad))
      .l(static_cast<cl_long>(n))
      .f(lr)
      .f(beta1)
      .f(beta2)
      .f(eps)
      .f(weight_decay)
      .f(bc1)
      .f(bc2)
      .over(n, 256);
}

double launch_sum_squares(Context& ctx, const float* x, int64_t n) {
  if (n <= 0) return 0.0;
  constexpr int kLocal = 256;
  int groups = static_cast<int>((n + kLocal - 1) / kLocal);
  if (groups > 1024) groups = 1024;
  const bool fp64 = ContextAccess::get(ctx).fp64;
  if (fp64) {
    cl_mem partials = alloc_mem(ctx, static_cast<size_t>(groups) * sizeof(double));
    cl_mem out = alloc_mem(ctx, sizeof(double));
    KernelCall(ctx, "sum_squares_d")
        .mem(mem_of(x))
        .l(static_cast<cl_long>(n))
        .mem(partials)
        .local(kLocal * sizeof(double))
        .groups(groups, kLocal);
    KernelCall(ctx, "reduce_sum_d")
        .mem(partials)
        .i(groups)
        .mem(out)
        .local(kLocal * sizeof(double))
        .groups(1, kLocal);
    double total = 0.0;
    read_mem(ctx, out, &total, sizeof(total));
    free_mem(partials);
    free_mem(out);
    return total;
  }
  cl_mem partials = alloc_mem(ctx, static_cast<size_t>(groups) * sizeof(float));
  cl_mem out = alloc_mem(ctx, sizeof(float));
  KernelCall(ctx, "sum_squares_f")
      .mem(mem_of(x))
      .l(static_cast<cl_long>(n))
      .mem(partials)
      .local(kLocal * sizeof(float))
      .groups(groups, kLocal);
  KernelCall(ctx, "reduce_sum_f")
      .mem(partials)
      .i(groups)
      .mem(out)
      .local(kLocal * sizeof(float))
      .groups(1, kLocal);
  float total = 0.f;
  read_mem(ctx, out, &total, sizeof(total));
  free_mem(partials);
  free_mem(out);
  return static_cast<double>(total);
}

float launch_reduce_sum(Context& ctx, const float* x, int n) {
  if (n <= 0) return 0.f;
  cl_mem out = alloc_mem(ctx, sizeof(float));
  KernelCall(ctx, "reduce_sum_f")
      .mem(mem_of(x))
      .i(n)
      .mem(out)
      .local(256 * sizeof(float))
      .groups(1, 256);
  float total = 0.f;
  read_mem(ctx, out, &total, sizeof(total));
  free_mem(out);
  return total;
}

void gemm_nt(Context& ctx, float* C, const float* A, const float* B, int M, int N, int K,
             int batches, int ldc, int lda, int ldb, int64_t stride_c, int64_t stride_a,
             int64_t stride_b, float alpha, float beta) {
  enqueue_gemm(ctx, "gemm_nt", C, A, B, M, N, K, batches, ldc, lda, ldb, stride_c, stride_a,
               stride_b, alpha, beta);
}

void gemm_nn(Context& ctx, float* C, const float* A, const float* B, int M, int N, int K,
             int batches, int ldc, int lda, int ldb, int64_t stride_c, int64_t stride_a,
             int64_t stride_b, float alpha, float beta) {
  enqueue_gemm(ctx, "gemm_nn", C, A, B, M, N, K, batches, ldc, lda, ldb, stride_c, stride_a,
               stride_b, alpha, beta);
}

void gemm_tn(Context& ctx, float* C, const float* A, const float* B, int M, int N, int K,
             int batches, int ldc, int lda, int ldb, int64_t stride_c, int64_t stride_a,
             int64_t stride_b, float alpha, float beta) {
  enqueue_gemm(ctx, "gemm_tn", C, A, B, M, N, K, batches, ldc, lda, ldb, stride_c, stride_a,
               stride_b, alpha, beta);
}

void launch_embedding_gather(Context& ctx, float* out, const float* weight, const int64_t* indices,
                             int64_t rows, int dim) {
  const int64_t n = rows * static_cast<int64_t>(dim);
  KernelCall(ctx, "embedding_gather")
      .mem(mem_of(out))
      .mem(mem_of(weight))
      .mem(mem_of(indices))
      .l(static_cast<cl_long>(rows))
      .i(dim)
      .over(n, 256);
}

void launch_embedding_scatter(Context& ctx, float* grad_weight, const float* grad_output,
                              const int64_t* indices, int64_t rows, int dim) {
  const int64_t n = rows * static_cast<int64_t>(dim);
  KernelCall(ctx, "embedding_scatter")
      .mem(mem_of(grad_weight))
      .mem(mem_of(grad_output))
      .mem(mem_of(indices))
      .l(static_cast<cl_long>(rows))
      .i(dim)
      .over(n, 256);
}

void launch_layer_norm_forward(Context& ctx, float* y, float* mean, float* inv_std, const float* x,
                               const float* gamma, const float* beta, int rows, int cols, float eps) {
  KernelCall call(ctx, "layer_norm_forward");
  call.mem(mem_of(y))
      .mem(mem_of(mean))
      .mem(mem_of(inv_std))
      .mem(mem_of(x))
      .mem(mem_of(gamma))
      .mem(mem_of(beta))
      .i(cols)
      .f(eps)
      .local(256 * sizeof(float));
  call.groups(rows, 256);
}

void launch_layer_norm_backward_input(Context& ctx, float* dx, const float* dy, const float* x,
                                      const float* mean, const float* inv_std, const float* gamma,
                                      int rows, int cols) {
  KernelCall call(ctx, "layer_norm_backward_input");
  call.mem(mem_of(dx))
      .mem(mem_of(dy))
      .mem(mem_of(x))
      .mem(mem_of(mean))
      .mem(mem_of(inv_std))
      .mem(mem_of(gamma))
      .i(cols)
      .local(256 * sizeof(float));
  call.groups(rows, 256);
}

void launch_layer_norm_param_grad(Context& ctx, const float* dy, const float* x, const float* mean,
                                  const float* inv_std, float* grad_gamma, float* grad_beta, int rows,
                                  int cols) {
  KernelCall call(ctx, "layer_norm_param_grad");
  call.mem(mem_of(dy))
      .mem(mem_of(x))
      .mem(mem_of(mean))
      .mem(mem_of(inv_std))
      .mem(mem_of(grad_gamma))
      .mem(mem_of(grad_beta))
      .i(rows)
      .i(cols)
      .local(256 * sizeof(float));
  call.groups(cols, 256);
}

void launch_rms_norm_forward(Context& ctx, float* y, float* inv_rms, const float* x,
                             const float* weight, int rows, int cols, float eps) {
  KernelCall call(ctx, "rms_norm_forward");
  call.mem(mem_of(y))
      .mem(mem_of(inv_rms))
      .mem(mem_of(x))
      .mem(mem_of(weight))
      .i(cols)
      .f(eps)
      .local(256 * sizeof(float));
  call.groups(rows, 256);
}

void launch_rms_norm_backward_input(Context& ctx, float* dx, const float* dy, const float* x,
                                    const float* inv_rms, const float* weight, int rows, int cols) {
  KernelCall call(ctx, "rms_norm_backward_input");
  call.mem(mem_of(dx))
      .mem(mem_of(dy))
      .mem(mem_of(x))
      .mem(mem_of(inv_rms))
      .mem(mem_of(weight))
      .i(cols)
      .local(256 * sizeof(float));
  call.groups(rows, 256);
}

void launch_rms_norm_param_grad(Context& ctx, const float* dy, const float* x, const float* inv_rms,
                                float* grad_weight, int rows, int cols) {
  KernelCall call(ctx, "rms_norm_param_grad");
  call.mem(mem_of(dy))
      .mem(mem_of(x))
      .mem(mem_of(inv_rms))
      .mem(mem_of(grad_weight))
      .i(rows)
      .i(cols)
      .local(256 * sizeof(float));
  call.groups(cols, 256);
}

void launch_split_heads(Context& ctx, float* dst, const float* src, int time, int heads, int dh,
                        int64_t n) {
  KernelCall(ctx, "split_heads")
      .mem(mem_of(dst))
      .mem(mem_of(src))
      .i(time)
      .i(heads)
      .i(dh)
      .l(static_cast<cl_long>(n))
      .over(n, 256);
}

void launch_merge_heads(Context& ctx, float* dst, const float* src, int time, int heads, int dh,
                        int64_t n) {
  KernelCall(ctx, "merge_heads")
      .mem(mem_of(dst))
      .mem(mem_of(src))
      .i(time)
      .i(heads)
      .i(dh)
      .l(static_cast<cl_long>(n))
      .over(n, 256);
}

void launch_rope(Context& ctx, float* q, float* k, int time, int dh, int pos0, int64_t npairs,
                 int inverse) {
  KernelCall(ctx, "rope_qk")
      .mem(mem_of(q))
      .mem(mem_of(k))
      .i(time)
      .i(dh)
      .i(pos0)
      .l(static_cast<cl_long>(npairs))
      .i(inverse)
      .over(npairs, 256);
}

void launch_causal_mask(Context& ctx, float* scores, int time, int64_t n) {
  KernelCall(ctx, "causal_mask")
      .mem(mem_of(scores))
      .i(time)
      .l(static_cast<cl_long>(n))
      .over(n, 256);
}

void launch_write_kv(Context& ctx, float* dst, const float* src, int t, int t_max, int dh,
                     int64_t n) {
  KernelCall(ctx, "write_kv")
      .mem(mem_of(dst))
      .mem(mem_of(src))
      .i(t)
      .i(t_max)
      .i(dh)
      .l(static_cast<cl_long>(n))
      .over(n, 256);
}

void launch_softmax_rows(Context& ctx, float* out, const float* in, int rows, int cols) {
  KernelCall call(ctx, "softmax_rows");
  call.mem(mem_of(out)).mem(mem_of(in)).i(cols).local(256 * sizeof(float));
  call.groups(rows, 256);
}

void launch_softmax_backward(Context& ctx, float* ds, const float* probs, const float* dp, int rows,
                             int cols, int causal) {
  KernelCall call(ctx, "softmax_backward");
  call.mem(mem_of(ds)).mem(mem_of(probs)).mem(mem_of(dp)).i(cols).i(causal).local(256 * sizeof(float));
  call.groups(rows, 256);
}

void launch_softmax_ce(Context& ctx, const float* logits, const int64_t* labels, float* grad,
                       float* per_sample, int n, int classes, int* bad) {
  KernelCall call(ctx, "softmax_ce");
  call.mem(mem_of(logits))
      .mem(mem_of(labels))
      .mem(mem_of(grad))
      .mem(mem_of(per_sample))
      .i(n)
      .i(classes)
      .mem(mem_of(bad))
      .local(128 * sizeof(float));
  call.groups(n, 128);
}

void launch_sequence_label_count(Context& ctx, const int64_t* labels, int rows, int classes,
                                 int64_t ignore_index, int* n_valid, int* bad) {
  KernelCall(ctx, "sequence_label_count")
      .mem(mem_of(labels))
      .i(rows)
      .i(classes)
      .l(static_cast<cl_long>(ignore_index))
      .mem(mem_of(n_valid))
      .mem(mem_of(bad))
      .over(rows, 256);
}

void launch_sequence_ce(Context& ctx, const float* logits, const int64_t* labels, float* grad,
                        float* per_sample, int rows, int classes, int64_t ignore_index, int n_valid,
                        int* bad) {
  KernelCall call(ctx, "sequence_ce");
  call.mem(mem_of(logits))
      .mem(mem_of(labels))
      .mem(mem_of(grad))
      .mem(mem_of(per_sample))
      .i(rows)
      .i(classes)
      .l(static_cast<cl_long>(ignore_index))
      .i(n_valid)
      .mem(mem_of(bad))
      .local(128 * sizeof(float));
  call.groups(rows, 128);
}

void launch_conv2d_forward(Context& ctx, float* y, const float* x, const float* w, const float* bias,
                           int has_bias, int N, int IC, int OC, int H, int W, int K, int OH, int OW,
                           int stride, int pad) {
  const cl_mem bias_mem = bias ? mem_of(bias) : ContextAccess::get(ctx).dummy;
  const int64_t n = static_cast<int64_t>(N) * OC * OH * OW;
  KernelCall(ctx, "conv2d_forward")
      .mem(mem_of(y))
      .mem(mem_of(x))
      .mem(mem_of(w))
      .mem(bias_mem)
      .i(has_bias)
      .i(N)
      .i(IC)
      .i(OC)
      .i(H)
      .i(W)
      .i(K)
      .i(OH)
      .i(OW)
      .i(stride)
      .i(pad)
      .over(n, 256);
}

void launch_conv2d_backward_data(Context& ctx, float* dx, const float* dy, const float* w, int N,
                                 int IC, int OC, int H, int W, int K, int OH, int OW, int stride,
                                 int pad) {
  const int64_t n = static_cast<int64_t>(N) * IC * H * W;
  KernelCall(ctx, "conv2d_backward_data")
      .mem(mem_of(dx))
      .mem(mem_of(dy))
      .mem(mem_of(w))
      .i(N)
      .i(IC)
      .i(OC)
      .i(H)
      .i(W)
      .i(K)
      .i(OH)
      .i(OW)
      .i(stride)
      .i(pad)
      .over(n, 256);
}

void launch_conv2d_backward_filter(Context& ctx, float* dw, const float* x, const float* dy, int N,
                                   int IC, int OC, int H, int W, int K, int OH, int OW, int stride,
                                   int pad) {
  const int64_t n = static_cast<int64_t>(OC) * IC * K * K;
  KernelCall(ctx, "conv2d_backward_filter")
      .mem(mem_of(dw))
      .mem(mem_of(x))
      .mem(mem_of(dy))
      .i(N)
      .i(IC)
      .i(OC)
      .i(H)
      .i(W)
      .i(K)
      .i(OH)
      .i(OW)
      .i(stride)
      .i(pad)
      .over(n, 256);
}

void launch_conv_bias_grad(Context& ctx, const float* dy, float* grad_bias, int N, int OC, int OH,
                           int OW) {
  KernelCall call(ctx, "conv_bias_grad");
  call.mem(mem_of(dy)).mem(mem_of(grad_bias)).i(N).i(OC).i(OH).i(OW).local(256 * sizeof(float));
  call.groups(OC, 256);
}

void launch_max_pool_forward(Context& ctx, float* y, int* index, const float* x, int N, int C, int H,
                             int W, int K, int OH, int OW, int stride) {
  const int64_t n = static_cast<int64_t>(N) * C * OH * OW;
  KernelCall(ctx, "max_pool_forward")
      .mem(mem_of(y))
      .mem(mem_of(index))
      .mem(mem_of(x))
      .i(N)
      .i(C)
      .i(H)
      .i(W)
      .i(K)
      .i(OH)
      .i(OW)
      .i(stride)
      .over(n, 256);
}

void launch_max_pool_backward(Context& ctx, float* dx, const float* dy, const int* index, int N,
                              int C, int H, int W, int OH, int OW) {
  const int64_t n = static_cast<int64_t>(N) * C * OH * OW;
  KernelCall(ctx, "max_pool_backward")
      .mem(mem_of(dx))
      .mem(mem_of(dy))
      .mem(mem_of(index))
      .i(N)
      .i(C)
      .i(H)
      .i(W)
      .i(OH)
      .i(OW)
      .over(n, 256);
}

void launch_global_avg_forward(Context& ctx, float* y, const float* x, int N, int C, int H, int W) {
  KernelCall call(ctx, "global_avg_forward");
  call.mem(mem_of(y)).mem(mem_of(x)).i(N).i(C).i(H).i(W).local(256 * sizeof(float));
  call.groups(N * C, 256);
}

void launch_global_avg_backward(Context& ctx, float* dx, const float* dy, int N, int C, int H,
                                int W) {
  const int64_t n = static_cast<int64_t>(N) * C * H * W;
  KernelCall(ctx, "global_avg_backward")
      .mem(mem_of(dx))
      .mem(mem_of(dy))
      .i(N)
      .i(C)
      .i(H)
      .i(W)
      .over(n, 256);
}

void launch_batch_norm_train(Context& ctx, float* y, const float* x, const float* gamma,
                             const float* beta, float* running_mean, float* running_var,
                             float* saved_mean, float* saved_inv, int N, int C, int H, int W,
                             float eps, float factor) {
  KernelCall call(ctx, "batch_norm_train");
  call.mem(mem_of(y))
      .mem(mem_of(x))
      .mem(mem_of(gamma))
      .mem(mem_of(beta))
      .mem(mem_of(running_mean))
      .mem(mem_of(running_var))
      .mem(mem_of(saved_mean))
      .mem(mem_of(saved_inv))
      .i(N)
      .i(C)
      .i(H)
      .i(W)
      .f(eps)
      .f(factor)
      .local(256 * sizeof(float));
  call.groups(C, 256);
}

void launch_batch_norm_eval(Context& ctx, float* y, const float* x, const float* gamma,
                            const float* beta, const float* running_mean, const float* running_var,
                            int N, int C, int H, int W, float eps) {
  const int64_t n = static_cast<int64_t>(N) * C * H * W;
  KernelCall(ctx, "batch_norm_eval")
      .mem(mem_of(y))
      .mem(mem_of(x))
      .mem(mem_of(gamma))
      .mem(mem_of(beta))
      .mem(mem_of(running_mean))
      .mem(mem_of(running_var))
      .i(N)
      .i(C)
      .i(H)
      .i(W)
      .f(eps)
      .over(n, 256);
}

void launch_batch_norm_backward(Context& ctx, float* dx, const float* dy, const float* x,
                                const float* gamma, const float* saved_mean, const float* saved_inv,
                                float* grad_gamma, float* grad_beta, int N, int C, int H, int W) {
  KernelCall call(ctx, "batch_norm_backward");
  call.mem(mem_of(dx))
      .mem(mem_of(dy))
      .mem(mem_of(x))
      .mem(mem_of(gamma))
      .mem(mem_of(saved_mean))
      .mem(mem_of(saved_inv))
      .mem(mem_of(grad_gamma))
      .mem(mem_of(grad_beta))
      .i(N)
      .i(C)
      .i(H)
      .i(W)
      .local(256 * sizeof(float));
  call.groups(C, 256);
}

}  // namespace nn
