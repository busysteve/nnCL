// OpenCL C kernels for nnCL. One in-order queue. float32, row-major, NCHW.
// OpenCL C 1.2 has no FP_CONTRACT pragma. NVIDIA and Apple report 1.2.
#if defined(__OPENCL_C_VERSION__) && __OPENCL_C_VERSION__ >= 200
#pragma OPENCL FP_CONTRACT OFF
#endif

float block_sum(float value, __local float* buffer) {
  const int tid = get_local_id(0);
  const int width = get_local_size(0);
  buffer[tid] = value;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int offset = width >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) buffer[tid] += buffer[tid + offset];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  const float total = buffer[0];
  barrier(CLK_LOCAL_MEM_FENCE);
  return total;
}

float block_max(float value, __local float* buffer) {
  const int tid = get_local_id(0);
  const int width = get_local_size(0);
  buffer[tid] = value;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int offset = width >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) buffer[tid] = fmax(buffer[tid], buffer[tid + offset]);
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  const float total = buffer[0];
  barrier(CLK_LOCAL_MEM_FENCE);
  return total;
}

// libclc's reciprocal is an ulp low here (12/3 -> 3.99999976). Normalization
// multiplies that error by rsqrt(eps), which misses a 1e-5 check. One FMA
// residual step restores the correctly rounded quotient.
float divf(float a, float b) {
  const float q = a / b;
  const float r = fma(-q, b, a);
  return fma(r, 1.f / b, q);
}

void atomic_add_f(__global float* addr, float val) {
  __global int* iaddr = (__global int*)addr;
  int expected = atomic_cmpxchg(iaddr, 0, 0);
  while (1) {
    union {
      int i;
      float f;
    } next;
    next.f = as_float(expected) + val;
    const int old = atomic_cmpxchg(iaddr, expected, next.i);
    if (old == expected) return;
    expected = old;
  }
}

float gelu_forward(float x) {
  const float k_inv_sqrt2 = 0.7071067811865476f;
  return 0.5f * x * (1.f + erf(x * k_inv_sqrt2));
}

float gelu_grad(float x) {
  const float k_inv_sqrt2 = 0.7071067811865476f;
  const float k_inv_sqrt2pi = 0.3989422804014327f;
  const float cdf = 0.5f * (1.f + erf(x * k_inv_sqrt2));
  const float pdf = k_inv_sqrt2pi * exp(-0.5f * x * x);
  return cdf + x * pdf;
}

float sigmoid_stable(float x) {
  if (x >= 0.f) {
    const float z = exp(-x);
    return 1.f / (1.f + z);
  }
  const float z = exp(x);
  return z / (1.f + z);
}

float silu(float x) { return x * sigmoid_stable(x); }

float silu_grad(float x) {
  const float s = sigmoid_stable(x);
  return s * (1.f + x * (1.f - s));
}

__kernel void scale_f(__global float* x, long n, float factor) {
  const long i = (long)get_global_id(0);
  if (i < n) x[i] *= factor;
}

__kernel void add_bias(__global float* y, __global const float* bias, int rows, int cols) {
  const long n = (long)rows * cols;
  const long i = (long)get_global_id(0);
  if (i < n) y[i] += bias[i % cols];
}

__kernel void bias_grad(__global const float* grad_y, __global float* grad_bias, int rows, int cols,
                        __local float* buffer) {
  const int col = (int)get_group_id(0);
  if (col >= cols) return;
  float sum = 0.f;
  for (int row = (int)get_local_id(0); row < rows; row += (int)get_local_size(0)) {
    sum += grad_y[(long)row * cols + col];
  }
  sum = block_sum(sum, buffer);
  if (get_local_id(0) == 0) grad_bias[col] += sum;
}

__kernel void add2(__global float* dst, __global const float* a, __global const float* b, long n) {
  const long i = (long)get_global_id(0);
  if (i < n) dst[i] = a[i] + b[i];
}

__kernel void mul2(__global float* y, __global const float* a, __global const float* b, long n) {
  const long i = (long)get_global_id(0);
  if (i < n) y[i] = a[i] * b[i];
}

__kernel void gelu_k(__global float* y, __global const float* x, long n) {
  const long i = (long)get_global_id(0);
  if (i < n) y[i] = gelu_forward(x[i]);
}

__kernel void gelu_backward_k(__global float* grad_x, __global const float* x,
                              __global const float* grad_y, long n) {
  const long i = (long)get_global_id(0);
  if (i < n) grad_x[i] = grad_y[i] * gelu_grad(x[i]);
}

__kernel void swiglu_k(__global float* y, __global const float* x, int rows, long inner) {
  const long n = (long)rows * inner;
  const long i = (long)get_global_id(0);
  if (i >= n) return;
  const long row = i / inner;
  const long col = i - row * inner;
  const long base = row * (inner * 2) + col;
  y[i] = silu(x[base]) * x[base + inner];
}

__kernel void swiglu_backward_k(__global float* grad_x, __global const float* x,
                                __global const float* grad_y, int rows, long inner) {
  const long n = (long)rows * inner;
  const long i = (long)get_global_id(0);
  if (i >= n) return;
  const long row = i / inner;
  const long col = i - row * inner;
  const long base = row * (inner * 2) + col;
  const float gate = x[base];
  const float value = x[base + inner];
  const float dy = grad_y[i];
  grad_x[base] = dy * value * silu_grad(gate);
  grad_x[base + inner] = dy * silu(gate);
}

__kernel void tanh_k(__global float* y, __global const float* x, long n) {
  const long i = (long)get_global_id(0);
  if (i < n) y[i] = tanh(x[i]);
}

__kernel void tanh_backward_k(__global float* grad_x, __global const float* y,
                              __global const float* grad_y, long n) {
  const long i = (long)get_global_id(0);
  if (i >= n) return;
  const float h = y[i];
  grad_x[i] = grad_y[i] * (1.f - h * h);
}

__kernel void relu_k(__global float* y, __global const float* x, long n) {
  const long i = (long)get_global_id(0);
  if (i < n) y[i] = x[i] > 0.f ? x[i] : 0.f;
}

__kernel void relu_backward_k(__global float* grad_x, __global const float* x,
                              __global const float* grad_y, long n) {
  const long i = (long)get_global_id(0);
  if (i < n) grad_x[i] = x[i] > 0.f ? grad_y[i] : 0.f;
}

__kernel void gather_time(__global float* dst, __global const float* src, int batch, int time,
                          int features, int index) {
  const long n = (long)batch * features;
  const long i = (long)get_global_id(0);
  if (i >= n) return;
  const long row = i / features;
  const long col = i - row * features;
  dst[i] = src[(row * time + index) * features + col];
}

__kernel void scatter_time(__global float* dst, __global const float* src, int batch, int time,
                           int features, int index) {
  const long n = (long)batch * features;
  const long i = (long)get_global_id(0);
  if (i >= n) return;
  const long row = i / features;
  const long col = i - row * features;
  dst[(row * time + index) * features + col] = src[i];
}

__kernel void sgd_k(__global float* param, __global float* velocity, __global const float* grad,
                    long n, float lr, float momentum) {
  const long i = (long)get_global_id(0);
  if (i >= n) return;
  const float updated = momentum * velocity[i] + grad[i];
  velocity[i] = updated;
  param[i] = param[i] - lr * updated;
}

__kernel void adamw_k(__global float* param, __global float* m, __global float* v,
                      __global const float* grad, long n, float lr, float beta1, float beta2,
                      float eps, float weight_decay, float bc1, float bc2) {
  const long i = (long)get_global_id(0);
  if (i >= n) return;
  const float g = grad[i];
  const float m_new = beta1 * m[i] + (1.f - beta1) * g;
  const float v_new = beta2 * v[i] + (1.f - beta2) * g * g;
  m[i] = m_new;
  v[i] = v_new;
  const float m_hat = m_new / bc1;
  const float v_hat = v_new / bc2;
  const float update = m_hat / (sqrt(v_hat) + eps) + weight_decay * param[i];
  param[i] = param[i] - lr * update;
}

__kernel void sum_squares_f(__global const float* x, long n, __global float* partials,
                            __local float* buffer) {
  float sum = 0.f;
  const long stride = (long)get_local_size(0) * get_num_groups(0);
  for (long i = (long)get_global_id(0); i < n; i += stride) {
    const float value = x[i];
    sum += value * value;
  }
  sum = block_sum(sum, buffer);
  if (get_local_id(0) == 0) partials[get_group_id(0)] = sum;
}

#ifdef HAVE_FP64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void sum_squares_d(__global const float* x, long n, __global double* partials,
                            __local double* buffer) {
  const int tid = get_local_id(0);
  double sum = 0.0;
  const long stride = (long)get_local_size(0) * get_num_groups(0);
  for (long i = (long)get_global_id(0); i < n; i += stride) {
    const double value = (double)x[i];
    sum += value * value;
  }
  buffer[tid] = sum;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int offset = get_local_size(0) >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) buffer[tid] += buffer[tid + offset];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (tid == 0) partials[get_group_id(0)] = buffer[0];
}

__kernel void reduce_sum_d(__global const double* x, int n, __global double* out,
                           __local double* buffer) {
  const int tid = get_local_id(0);
  double sum = 0.0;
  for (int i = tid; i < n; i += (int)get_local_size(0)) sum += x[i];
  buffer[tid] = sum;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int offset = get_local_size(0) >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) buffer[tid] += buffer[tid + offset];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (tid == 0) *out = buffer[0];
}
#endif

__kernel void reduce_sum_f(__global const float* x, int n, __global float* out,
                           __local float* buffer) {
  float sum = 0.f;
  for (int i = (int)get_local_id(0); i < n; i += (int)get_local_size(0)) sum += x[i];
  sum = block_sum(sum, buffer);
  if (get_local_id(0) == 0) *out = sum;
}

// C[b,i,j] = alpha * sum_p A[b,i,p] * B[b,j,p] + beta * C
__kernel void gemm_nt(__global float* C, __global const float* A, __global const float* B, int M,
                      int N, int K, int batches, int ldc, int lda, int ldb, long stride_c,
                      long stride_a, long stride_b, float alpha, float beta) {
  const int j = (int)get_global_id(0);
  const int i = (int)get_global_id(1);
  const int b = (int)get_global_id(2);
  if (i >= M || j >= N || b >= batches) return;
  const __global float* a = A + b * stride_a + (long)i * lda;
  const __global float* brow = B + b * stride_b + (long)j * ldb;
  float sum = 0.f;
  for (int p = 0; p < K; ++p) sum += a[p] * brow[p];
  __global float* c = C + b * stride_c + (long)i * ldc + j;
  *c = (beta == 0.f) ? alpha * sum : alpha * sum + beta * (*c);
}

// C[b,i,j] = alpha * sum_p A[b,i,p] * B[b,p,j] + beta * C
__kernel void gemm_nn(__global float* C, __global const float* A, __global const float* B, int M,
                      int N, int K, int batches, int ldc, int lda, int ldb, long stride_c,
                      long stride_a, long stride_b, float alpha, float beta) {
  const int j = (int)get_global_id(0);
  const int i = (int)get_global_id(1);
  const int b = (int)get_global_id(2);
  if (i >= M || j >= N || b >= batches) return;
  const __global float* a = A + b * stride_a + (long)i * lda;
  const __global float* bbase = B + b * stride_b;
  float sum = 0.f;
  for (int p = 0; p < K; ++p) sum += a[p] * bbase[(long)p * ldb + j];
  __global float* c = C + b * stride_c + (long)i * ldc + j;
  *c = (beta == 0.f) ? alpha * sum : alpha * sum + beta * (*c);
}

// C[b,i,j] = alpha * sum_p A[b,p,i] * B[b,p,j] + beta * C
__kernel void gemm_tn(__global float* C, __global const float* A, __global const float* B, int M,
                      int N, int K, int batches, int ldc, int lda, int ldb, long stride_c,
                      long stride_a, long stride_b, float alpha, float beta) {
  const int j = (int)get_global_id(0);
  const int i = (int)get_global_id(1);
  const int b = (int)get_global_id(2);
  if (i >= M || j >= N || b >= batches) return;
  const __global float* abase = A + b * stride_a;
  const __global float* bbase = B + b * stride_b;
  float sum = 0.f;
  for (int p = 0; p < K; ++p) sum += abase[(long)p * lda + i] * bbase[(long)p * ldb + j];
  __global float* c = C + b * stride_c + (long)i * ldc + j;
  *c = (beta == 0.f) ? alpha * sum : alpha * sum + beta * (*c);
}

__kernel void embedding_gather(__global float* out, __global const float* weight,
                               __global const long* indices, long rows, int dim) {
  const long n = rows * (long)dim;
  const long i = (long)get_global_id(0);
  if (i >= n) return;
  const long row = i / dim;
  const int col = (int)(i - row * dim);
  const long id = indices[row];
  out[i] = weight[id * (long)dim + col];
}

__kernel void embedding_scatter(__global float* grad_weight, __global const float* grad_output,
                                __global const long* indices, long rows, int dim) {
  const long n = rows * (long)dim;
  const long i = (long)get_global_id(0);
  if (i >= n) return;
  const long row = i / dim;
  const int col = (int)(i - row * dim);
  const long id = indices[row];
  atomic_add_f(grad_weight + id * (long)dim + col, grad_output[i]);
}

__kernel void layer_norm_forward(__global float* y, __global float* mean_out,
                                 __global float* inv_std_out, __global const float* x,
                                 __global const float* gamma, __global const float* beta, int cols,
                                 float eps, __local float* buffer) {
  const int row = (int)get_group_id(0);
  const long base = (long)row * cols;
  const __global float* row_x = x + base;
  __global float* row_y = y + base;
  float sum = 0.f;
  for (int col = (int)get_local_id(0); col < cols; col += (int)get_local_size(0)) sum += row_x[col];
  const float mean = divf(block_sum(sum, buffer), (float)cols);
  float var_sum = 0.f;
  for (int col = (int)get_local_id(0); col < cols; col += (int)get_local_size(0)) {
    const float diff = row_x[col] - mean;
    var_sum += diff * diff;
  }
  var_sum = block_sum(var_sum, buffer);
  const float inv_std = rsqrt(divf(var_sum, (float)cols) + eps);
  if (get_local_id(0) == 0) {
    mean_out[row] = mean;
    inv_std_out[row] = inv_std;
  }
  for (int col = (int)get_local_id(0); col < cols; col += (int)get_local_size(0)) {
    const float xhat = (row_x[col] - mean) * inv_std;
    row_y[col] = xhat * gamma[col] + beta[col];
  }
}

__kernel void layer_norm_backward_input(__global float* dx, __global const float* dy,
                                        __global const float* x, __global const float* mean,
                                        __global const float* inv_std, __global const float* gamma,
                                        int cols, __local float* buffer) {
  const int row = (int)get_group_id(0);
  const long base = (long)row * cols;
  const float mean_v = mean[row];
  const float inv = inv_std[row];
  float sum_dhat = 0.f;
  float sum_dhat_xhat = 0.f;
  for (int col = (int)get_local_id(0); col < cols; col += (int)get_local_size(0)) {
    const float xhat = (x[base + col] - mean_v) * inv;
    const float dhat = dy[base + col] * gamma[col];
    sum_dhat += dhat;
    sum_dhat_xhat += dhat * xhat;
  }
  sum_dhat = block_sum(sum_dhat, buffer);
  sum_dhat_xhat = block_sum(sum_dhat_xhat, buffer);
  const float mean_dhat = divf(sum_dhat, (float)cols);
  const float mean_dhat_xhat = divf(sum_dhat_xhat, (float)cols);
  for (int col = (int)get_local_id(0); col < cols; col += (int)get_local_size(0)) {
    const float xhat = (x[base + col] - mean_v) * inv;
    const float dhat = dy[base + col] * gamma[col];
    dx[base + col] = inv * (dhat - mean_dhat - xhat * mean_dhat_xhat);
  }
}

__kernel void layer_norm_param_grad(__global const float* dy, __global const float* x,
                                    __global const float* mean, __global const float* inv_std,
                                    __global float* grad_gamma, __global float* grad_beta, int rows,
                                    int cols, __local float* buffer) {
  const int col = (int)get_group_id(0);
  float sum_gamma = 0.f;
  float sum_beta = 0.f;
  for (int row = (int)get_local_id(0); row < rows; row += (int)get_local_size(0)) {
    const long index = (long)row * cols + col;
    const float xhat = (x[index] - mean[row]) * inv_std[row];
    const float dy_v = dy[index];
    sum_gamma += dy_v * xhat;
    sum_beta += dy_v;
  }
  sum_gamma = block_sum(sum_gamma, buffer);
  sum_beta = block_sum(sum_beta, buffer);
  if (get_local_id(0) == 0) {
    grad_gamma[col] += sum_gamma;
    grad_beta[col] += sum_beta;
  }
}

__kernel void rms_norm_forward(__global float* y, __global float* inv_rms_out,
                               __global const float* x, __global const float* weight, int cols,
                               float eps, __local float* buffer) {
  const int row = (int)get_group_id(0);
  const long base = (long)row * cols;
  const __global float* row_x = x + base;
  __global float* row_y = y + base;
  float sq = 0.f;
  for (int col = (int)get_local_id(0); col < cols; col += (int)get_local_size(0)) {
    const float value = row_x[col];
    sq += value * value;
  }
  sq = block_sum(sq, buffer);
  const float inv_rms = rsqrt(divf(sq, (float)cols) + eps);
  if (get_local_id(0) == 0) inv_rms_out[row] = inv_rms;
  for (int col = (int)get_local_id(0); col < cols; col += (int)get_local_size(0)) {
    row_y[col] = row_x[col] * inv_rms * weight[col];
  }
}

__kernel void rms_norm_backward_input(__global float* dx, __global const float* dy,
                                      __global const float* x, __global const float* inv_rms,
                                      __global const float* weight, int cols, __local float* buffer) {
  const int row = (int)get_group_id(0);
  const long base = (long)row * cols;
  const float inv = inv_rms[row];
  float sum_dhat_xhat = 0.f;
  for (int col = (int)get_local_id(0); col < cols; col += (int)get_local_size(0)) {
    const float xhat = x[base + col] * inv;
    const float dhat = dy[base + col] * weight[col];
    sum_dhat_xhat += dhat * xhat;
  }
  const float mean_dhat_xhat = divf(block_sum(sum_dhat_xhat, buffer), (float)cols);
  for (int col = (int)get_local_id(0); col < cols; col += (int)get_local_size(0)) {
    const float xhat = x[base + col] * inv;
    const float dhat = dy[base + col] * weight[col];
    dx[base + col] = inv * (dhat - xhat * mean_dhat_xhat);
  }
}

__kernel void rms_norm_param_grad(__global const float* dy, __global const float* x,
                                  __global const float* inv_rms, __global float* grad_weight,
                                  int rows, int cols, __local float* buffer) {
  const int col = (int)get_group_id(0);
  float sum = 0.f;
  for (int row = (int)get_local_id(0); row < rows; row += (int)get_local_size(0)) {
    const long index = (long)row * cols + col;
    const float xhat = x[index] * inv_rms[row];
    sum += dy[index] * xhat;
  }
  sum = block_sum(sum, buffer);
  if (get_local_id(0) == 0) grad_weight[col] += sum;
}

__kernel void split_heads(__global float* dst, __global const float* src, int time, int heads,
                          int dh, long n) {
  const long i = (long)get_global_id(0);
  if (i >= n) return;
  const long d = i % dh;
  const long h = (i / dh) % heads;
  const long t = (i / ((long)dh * heads)) % time;
  const long batch = i / ((long)dh * heads * time);
  const long o = ((batch * heads + h) * time + t) * dh + d;
  dst[o] = src[i];
}

__kernel void merge_heads(__global float* dst, __global const float* src, int time, int heads,
                          int dh, long n) {
  const long i = (long)get_global_id(0);
  if (i >= n) return;
  const long d = i % dh;
  const long h = (i / dh) % heads;
  const long t = (i / ((long)dh * heads)) % time;
  const long batch = i / ((long)dh * heads * time);
  const long s = ((batch * heads + h) * time + t) * dh + d;
  dst[i] = src[s];
}

__kernel void rope_qk(__global float* q, __global float* k, int time, int dh, int pos0, long npairs,
                      int inverse) {
  const long pair_index = (long)get_global_id(0);
  if (pair_index >= npairs) return;
  const int pairs = dh / 2;
  const int pair = (int)(pair_index % pairs);
  const long vec = pair_index / pairs;
  const int t = (int)(vec % time);
  const long base = vec * dh + (long)pair * 2;
  const float theta = (float)(pos0 + t) *
                      exp(-log(10000.f) * (2.f * divf((float)pair, (float)dh)));
  const float c = cos(theta);
  const float s = sin(theta);
  const float q0 = q[base];
  const float q1 = q[base + 1];
  const float k0 = k[base];
  const float k1 = k[base + 1];
  if (inverse) {
    q[base] = q0 * c + q1 * s;
    q[base + 1] = -q0 * s + q1 * c;
    k[base] = k0 * c + k1 * s;
    k[base + 1] = -k0 * s + k1 * c;
  } else {
    q[base] = q0 * c - q1 * s;
    q[base + 1] = q0 * s + q1 * c;
    k[base] = k0 * c - k1 * s;
    k[base + 1] = k0 * s + k1 * c;
  }
}

__kernel void causal_mask(__global float* scores, int time, long n) {
  const long i = (long)get_global_id(0);
  if (i >= n) return;
  const int col = (int)(i % time);
  const int row = (int)((i / time) % time);
  if (col > row) scores[i] = -1.0e9f;
}

__kernel void write_kv(__global float* dst, __global const float* src, int t, int t_max, int dh,
                       long n) {
  const long i = (long)get_global_id(0);
  if (i >= n) return;
  const int d = (int)(i % dh);
  const long head = i / dh;
  dst[(head * (long)t_max + t) * dh + d] = src[i];
}

__kernel void softmax_rows(__global float* out, __global const float* in, int cols,
                           __local float* buffer) {
  const int row = (int)get_group_id(0);
  const long base = (long)row * cols;
  const __global float* row_in = in + base;
  __global float* row_out = out + base;
  float max_v = -INFINITY;
  for (int col = (int)get_local_id(0); col < cols; col += (int)get_local_size(0)) {
    max_v = fmax(max_v, row_in[col]);
  }
  max_v = block_max(max_v, buffer);
  float sum = 0.f;
  for (int col = (int)get_local_id(0); col < cols; col += (int)get_local_size(0)) {
    sum += exp(row_in[col] - max_v);
  }
  sum = block_sum(sum, buffer);
  const float inv = divf(1.f, sum);
  for (int col = (int)get_local_id(0); col < cols; col += (int)get_local_size(0)) {
    row_out[col] = exp(row_in[col] - max_v) * inv;
  }
}

__kernel void softmax_backward(__global float* ds, __global const float* probs,
                               __global const float* dp, int cols, int causal, __local float* buffer) {
  const int row = (int)get_group_id(0);
  const int query = row % cols;
  const long base = (long)row * cols;
  float dot = 0.f;
  for (int col = (int)get_local_id(0); col < cols; col += (int)get_local_size(0)) {
    dot += dp[base + col] * probs[base + col];
  }
  dot = block_sum(dot, buffer);
  for (int col = (int)get_local_id(0); col < cols; col += (int)get_local_size(0)) {
    float grad = probs[base + col] * (dp[base + col] - dot);
    if (causal && col > query) grad = 0.f;
    ds[base + col] = grad;
  }
}

__kernel void softmax_ce(__global const float* logits, __global const long* labels,
                         __global float* grad, __global float* per_sample, int n, int classes,
                         __global int* bad, __local float* scratch) {
  const int row = (int)get_group_id(0);
  if (row >= n) return;
  const __global float* row_logits = logits + (long)row * classes;
  __global float* row_grad = grad + (long)row * classes;
  float thread_max = -1e30f;
  for (int c = (int)get_local_id(0); c < classes; c += (int)get_local_size(0)) {
    thread_max = fmax(thread_max, row_logits[c]);
  }
  const float max_logit = block_max(thread_max, scratch);
  float thread_sum = 0.f;
  for (int c = (int)get_local_id(0); c < classes; c += (int)get_local_size(0)) {
    thread_sum += exp(row_logits[c] - max_logit);
  }
  const float sum = block_sum(thread_sum, scratch);
  const long label = labels[row];
  if (label < 0 || label >= classes) {
    if (get_local_id(0) == 0) atomic_cmpxchg(bad, 0, 1);
    return;
  }
  const float inv_sum = divf(1.f, sum);
  const float inv_n = divf(1.f, (float)n);
  for (int c = (int)get_local_id(0); c < classes; c += (int)get_local_size(0)) {
    const float softmax = exp(row_logits[c] - max_logit) * inv_sum;
    const float indicator = ((long)c == label) ? 1.f : 0.f;
    row_grad[c] = (softmax - indicator) * inv_n;
  }
  if (get_local_id(0) == 0) per_sample[row] = -(row_logits[label] - (max_logit + log(sum)));
}

__kernel void sequence_label_count(__global const long* labels, int rows, int classes,
                                   long ignore_index, __global int* n_valid, __global int* bad) {
  const long row = (long)get_global_id(0);
  if (row >= rows) return;
  const long label = labels[row];
  if (label == ignore_index) return;
  if (label < 0 || label >= (long)classes) {
    atomic_cmpxchg(bad, 0, 1);
    return;
  }
  atomic_add(n_valid, 1);
}

__kernel void sequence_ce(__global const float* logits, __global const long* labels,
                          __global float* grad, __global float* per_sample, int rows, int classes,
                          long ignore_index, int n_valid, __global int* bad, __local float* scratch) {
  const int row = (int)get_group_id(0);
  if (row >= rows) return;
  const __global float* row_logits = logits + (long)row * classes;
  __global float* row_grad = grad + (long)row * classes;
  const long label = labels[row];
  if (label == ignore_index) {
    for (int c = (int)get_local_id(0); c < classes; c += (int)get_local_size(0)) row_grad[c] = 0.f;
    if (get_local_id(0) == 0) per_sample[row] = 0.f;
    return;
  }
  if (label < 0 || label >= (long)classes) {
    for (int c = (int)get_local_id(0); c < classes; c += (int)get_local_size(0)) row_grad[c] = 0.f;
    if (get_local_id(0) == 0) {
      atomic_cmpxchg(bad, 0, 1);
      per_sample[row] = 0.f;
    }
    return;
  }
  float thread_max = -1e30f;
  for (int c = (int)get_local_id(0); c < classes; c += (int)get_local_size(0)) {
    thread_max = fmax(thread_max, row_logits[c]);
  }
  const float max_logit = block_max(thread_max, scratch);
  float thread_sum = 0.f;
  for (int c = (int)get_local_id(0); c < classes; c += (int)get_local_size(0)) {
    thread_sum += exp(row_logits[c] - max_logit);
  }
  const float sum = block_sum(thread_sum, scratch);
  const float inv_sum = divf(1.f, sum);
  const float inv_n = divf(1.f, (float)n_valid);
  for (int c = (int)get_local_id(0); c < classes; c += (int)get_local_size(0)) {
    const float softmax = exp(row_logits[c] - max_logit) * inv_sum;
    const float indicator = ((long)c == label) ? 1.f : 0.f;
    row_grad[c] = (softmax - indicator) * inv_n;
  }
  if (get_local_id(0) == 0) per_sample[row] = -(row_logits[label] - (max_logit + log(sum)));
}

int in_spatial(int v, int limit) { return v >= 0 && v < limit; }

__kernel void conv2d_forward(__global float* y, __global const float* x, __global const float* w,
                             __global const float* bias, int has_bias, int N, int IC, int OC, int H,
                             int W, int K, int OH, int OW, int stride, int pad) {
  const long total = (long)N * OC * OH * OW;
  const long idx = (long)get_global_id(0);
  if (idx >= total) return;
  const int ow = (int)(idx % OW);
  const int oh = (int)((idx / OW) % OH);
  const int oc = (int)((idx / ((long)OW * OH)) % OC);
  const int n = (int)(idx / ((long)OW * OH * OC));
  float sum = has_bias ? bias[oc] : 0.f;
  const int h0 = oh * stride - pad;
  const int w0 = ow * stride - pad;
  for (int ic = 0; ic < IC; ++ic) {
    for (int kh = 0; kh < K; ++kh) {
      for (int kw = 0; kw < K; ++kw) {
        const int ih = h0 + kh;
        const int iw = w0 + kw;
        if (!in_spatial(ih, H) || !in_spatial(iw, W)) continue;
        const float xv = x[(((long)n * IC + ic) * H + ih) * W + iw];
        const float wv = w[(((long)oc * IC + ic) * K + kh) * K + kw];
        sum += xv * wv;
      }
    }
  }
  y[idx] = sum;
}

__kernel void conv2d_backward_data(__global float* dx, __global const float* dy,
                                   __global const float* w, int N, int IC, int OC, int H, int W,
                                   int K, int OH, int OW, int stride, int pad) {
  const long total = (long)N * IC * H * W;
  const long idx = (long)get_global_id(0);
  if (idx >= total) return;
  const int iw = (int)(idx % W);
  const int ih = (int)((idx / W) % H);
  const int ic = (int)((idx / ((long)W * H)) % IC);
  const int n = (int)(idx / ((long)W * H * IC));
  float sum = 0.f;
  for (int oc = 0; oc < OC; ++oc) {
    for (int kh = 0; kh < K; ++kh) {
      const int h_tmp = ih + pad - kh;
      if (h_tmp % stride != 0) continue;
      const int oh = h_tmp / stride;
      if (!in_spatial(oh, OH)) continue;
      for (int kw = 0; kw < K; ++kw) {
        const int w_tmp = iw + pad - kw;
        if (w_tmp % stride != 0) continue;
        const int ow = w_tmp / stride;
        if (!in_spatial(ow, OW)) continue;
        const float g = dy[(((long)n * OC + oc) * OH + oh) * OW + ow];
        const float wv = w[(((long)oc * IC + ic) * K + kh) * K + kw];
        sum += g * wv;
      }
    }
  }
  dx[idx] = sum;
}

__kernel void conv2d_backward_filter(__global float* dw, __global const float* x,
                                     __global const float* dy, int N, int IC, int OC, int H, int W,
                                     int K, int OH, int OW, int stride, int pad) {
  const long total = (long)OC * IC * K * K;
  const long idx = (long)get_global_id(0);
  if (idx >= total) return;
  const int kw = (int)(idx % K);
  const int kh = (int)((idx / K) % K);
  const int ic = (int)((idx / ((long)K * K)) % IC);
  const int oc = (int)(idx / ((long)K * K * IC));
  float sum = 0.f;
  for (int n = 0; n < N; ++n) {
    for (int oh = 0; oh < OH; ++oh) {
      const int ih = oh * stride + kh - pad;
      if (!in_spatial(ih, H)) continue;
      for (int ow = 0; ow < OW; ++ow) {
        const int iw = ow * stride + kw - pad;
        if (!in_spatial(iw, W)) continue;
        const float xv = x[(((long)n * IC + ic) * H + ih) * W + iw];
        const float g = dy[(((long)n * OC + oc) * OH + oh) * OW + ow];
        sum += xv * g;
      }
    }
  }
  dw[idx] += sum;
}

__kernel void conv_bias_grad(__global const float* dy, __global float* grad_bias, int N, int OC,
                             int OH, int OW, __local float* buffer) {
  const int oc = (int)get_group_id(0);
  if (oc >= OC) return;
  const int spatial = OH * OW;
  float sum = 0.f;
  const int threads = (int)get_local_size(0);
  for (int i = (int)get_local_id(0); i < N * spatial; i += threads) {
    const int n = i / spatial;
    const int hw = i - n * spatial;
    sum += dy[(((long)n * OC + oc) * OH * OW) + hw];
  }
  sum = block_sum(sum, buffer);
  if (get_local_id(0) == 0) grad_bias[oc] += sum;
}

__kernel void max_pool_forward(__global float* y, __global int* index, __global const float* x,
                               int N, int C, int H, int W, int K, int OH, int OW, int stride) {
  const long total = (long)N * C * OH * OW;
  const long idx = (long)get_global_id(0);
  if (idx >= total) return;
  const int ow = (int)(idx % OW);
  const int oh = (int)((idx / OW) % OH);
  const int c = (int)((idx / ((long)OW * OH)) % C);
  const int n = (int)(idx / ((long)OW * OH * C));
  const int h0 = oh * stride;
  const int w0 = ow * stride;
  float best = -INFINITY;
  int best_i = 0;
  for (int kh = 0; kh < K; ++kh) {
    for (int kw = 0; kw < K; ++kw) {
      const int ih = h0 + kh;
      const int iw = w0 + kw;
      if (!in_spatial(ih, H) || !in_spatial(iw, W)) continue;
      const int flat = ((ih * W) + iw);
      const float value = x[(((long)n * C + c) * H + ih) * W + iw];
      if (value > best) {
        best = value;
        best_i = flat;
      }
    }
  }
  y[idx] = best;
  index[idx] = best_i;
}

__kernel void max_pool_backward(__global float* dx, __global const float* dy,
                                __global const int* index, int N, int C, int H, int W, int OH,
                                int OW) {
  const long total = (long)N * C * OH * OW;
  const long idx = (long)get_global_id(0);
  if (idx >= total) return;
  const int c = (int)((idx / ((long)OW * OH)) % C);
  const int n = (int)(idx / ((long)OW * OH * C));
  const int flat = index[idx];
  const int ih = flat / W;
  const int iw = flat - ih * W;
  atomic_add_f(dx + (((long)n * C + c) * H + ih) * W + iw, dy[idx]);
}

__kernel void global_avg_forward(__global float* y, __global const float* x, int N, int C, int H,
                                 int W, __local float* buffer) {
  const int nc = (int)get_group_id(0);
  if (nc >= N * C) return;
  const int c = nc % C;
  const int n = nc / C;
  const int spatial = H * W;
  float sum = 0.f;
  for (int i = (int)get_local_id(0); i < spatial; i += (int)get_local_size(0)) {
    sum += x[(((long)n * C + c) * H * W) + i];
  }
  sum = block_sum(sum, buffer);
  if (get_local_id(0) == 0) y[nc] = divf(sum, (float)spatial);
}

__kernel void global_avg_backward(__global float* dx, __global const float* dy, int N, int C, int H,
                                  int W) {
  const long spatial = (long)H * W;
  const long total = (long)N * C * spatial;
  const long idx = (long)get_global_id(0);
  if (idx >= total) return;
  const int hw = (int)(idx % spatial);
  const int c = (int)((idx / spatial) % C);
  const int n = (int)(idx / (spatial * C));
  (void)hw;
  dx[idx] = divf(dy[(long)n * C + c], (float)spatial);
}

__kernel void batch_norm_train(__global float* y, __global const float* x, __global const float* gamma,
                               __global const float* beta, __global float* running_mean,
                               __global float* running_var, __global float* saved_mean,
                               __global float* saved_inv, int N, int C, int H, int W, float eps,
                               float factor, __local float* buffer) {
  const int c = (int)get_group_id(0);
  if (c >= C) return;
  const int spatial = N * H * W;
  const int hw_plane = H * W;
  float sum = 0.f;
  for (int s = (int)get_local_id(0); s < spatial; s += (int)get_local_size(0)) {
    const int n = s / hw_plane;
    const int hw = s - n * hw_plane;
    sum += x[((long)n * C + c) * hw_plane + hw];
  }
  sum = block_sum(sum, buffer);
  const float mean = divf(sum, (float)spatial);
  float var_sum = 0.f;
  for (int s = (int)get_local_id(0); s < spatial; s += (int)get_local_size(0)) {
    const int n = s / hw_plane;
    const int hw = s - n * hw_plane;
    const float diff = x[((long)n * C + c) * hw_plane + hw] - mean;
    var_sum += diff * diff;
  }
  var_sum = block_sum(var_sum, buffer);
  const float var = divf(var_sum, (float)spatial);
  const float inv = rsqrt(var + eps);
  if (get_local_id(0) == 0) {
    saved_mean[c] = mean;
    saved_inv[c] = inv;
    running_mean[c] = (1.f - factor) * running_mean[c] + factor * mean;
    const float unbiased = var * divf((float)spatial, (float)(spatial - 1));
    running_var[c] = (1.f - factor) * running_var[c] + factor * unbiased;
  }
  barrier(CLK_GLOBAL_MEM_FENCE);
  for (int s = (int)get_local_id(0); s < spatial; s += (int)get_local_size(0)) {
    const int n = s / hw_plane;
    const int hw = s - n * hw_plane;
    const long index = ((long)n * C + c) * hw_plane + hw;
    const float xhat = (x[index] - mean) * inv;
    y[index] = xhat * gamma[c] + beta[c];
  }
}

__kernel void batch_norm_eval(__global float* y, __global const float* x, __global const float* gamma,
                              __global const float* beta, __global const float* running_mean,
                              __global const float* running_var, int N, int C, int H, int W,
                              float eps) {
  const long spatial = (long)H * W;
  const long total = (long)N * C * spatial;
  const long idx = (long)get_global_id(0);
  if (idx >= total) return;
  const int hw = (int)(idx % spatial);
  const int c = (int)((idx / spatial) % C);
  const int n = (int)(idx / (spatial * C));
  (void)hw;
  (void)n;
  const float inv = rsqrt(running_var[c] + eps);
  const float xhat = (x[idx] - running_mean[c]) * inv;
  y[idx] = xhat * gamma[c] + beta[c];
}

__kernel void batch_norm_backward(__global float* dx, __global const float* dy,
                                  __global const float* x, __global const float* gamma,
                                  __global const float* saved_mean, __global const float* saved_inv,
                                  __global float* grad_gamma, __global float* grad_beta, int N, int C,
                                  int H, int W, __local float* buffer) {
  const int c = (int)get_group_id(0);
  if (c >= C) return;
  const int spatial = N * H * W;
  const int hw_plane = H * W;
  const float mean = saved_mean[c];
  const float inv = saved_inv[c];
  const float scale = gamma[c];
  float sum_dhat = 0.f;
  float sum_dhat_xhat = 0.f;
  for (int s = (int)get_local_id(0); s < spatial; s += (int)get_local_size(0)) {
    const int n = s / hw_plane;
    const int hw = s - n * hw_plane;
    const long index = ((long)n * C + c) * hw_plane + hw;
    const float xhat = (x[index] - mean) * inv;
    const float dhat = dy[index] * scale;
    sum_dhat += dhat;
    sum_dhat_xhat += dhat * xhat;
  }
  sum_dhat = block_sum(sum_dhat, buffer);
  sum_dhat_xhat = block_sum(sum_dhat_xhat, buffer);
  const float mean_dhat = divf(sum_dhat, (float)spatial);
  const float mean_dhat_xhat = divf(sum_dhat_xhat, (float)spatial);
  if (get_local_id(0) == 0) {
    float g_sum = 0.f;
    float b_sum = 0.f;
    for (int s = 0; s < spatial; ++s) {
      const int n = s / hw_plane;
      const int hw = s - n * hw_plane;
      const long index = ((long)n * C + c) * hw_plane + hw;
      const float xhat = (x[index] - mean) * inv;
      g_sum += dy[index] * xhat;
      b_sum += dy[index];
    }
    grad_gamma[c] += g_sum;
    grad_beta[c] += b_sum;
  }
  for (int s = (int)get_local_id(0); s < spatial; s += (int)get_local_size(0)) {
    const int n = s / hw_plane;
    const int hw = s - n * hw_plane;
    const long index = ((long)n * C + c) * hw_plane + hw;
    const float xhat = (x[index] - mean) * inv;
    const float dhat = dy[index] * scale;
    dx[index] = inv * (dhat - mean_dhat - xhat * mean_dhat_xhat);
  }
}
