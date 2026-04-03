#include <math.h>
#include <torch/extension.h>
#include <cstdio>
#include <sstream>
#include <iostream>
#include <tuple>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <memory>
#include "name.h"
#include <fstream>
#include <string>
#include <functional>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda_fp16.h> 
#include <mma.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
using namespace nvcuda;
namespace cg = cooperative_groups;
#define CUDA_KERNEL_LOOP(i, n)                          \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;   \
      i < (n);                                          \
      i += blockDim.x * gridDim.x)

const int CUDA_NUM_THREADS = 1024;
inline int GET_BLOCKS(const int NN)
{
  return (NN + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}



/*
block size 32 16,  wrap size 32, tile size 64 * 64
*/
// grid_size (N + tile_x - 1) / tile_x, (M + tile_x - 1) / tile_x, B;  block size 32 * 16;  wrap size 32, tile size 64 * 64

__device__ int MAPPING(int y, int x, int group){
    int x_offset = x % group; 
    int x_g = x / group; 
    return (x_g^(y % 16)) * group + x_offset;
}

__global__ void gemm_cuda(
    const float* weight, const float* columns,
    float* output, const float* bias, const int M, const int K, const int N
)
{
    int bx =  blockIdx.x;
    int by =  blockIdx.y;
    int tx = threadIdx.x; // 0-31 thread idx per wrap
    int ty = threadIdx.y; // 0-15 wrap parallel
    int b_idx = blockIdx.z;
    // tile grid coord for output size
    uint2 tile_grid = {bx * TILE, by * TILE};
    // 16*16 tile per wrap， 16 wrap parallel
    uint2 wrap_group_idx = {ty % 4, ty / 4};

    float* out_im_ptr = output + b_idx * M * N;//  C_out ,  * H_out * W_out    
    const float* col_ptr = columns + b_idx * K * N; //  C * k_w * k_h ,  * H_out * W_out 
    // weight  C_out, C * k_w * k_h  
    __shared__ half MA[TILE][TILE + 8];
    __shared__ half MB[TILE][TILE + 8]; 
    __shared__ float MC[TILE][TILE];

    int round = (K + TILE - 1) / TILE;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);
    
    for (int k=0; k < round; ++k){
        // 将全局内存数据读入到共享内存中 uint2 mma_xy -- 4*8 4行坐标上连续 8列坐标上不连续 -- 为了避免 bank conflict
        for (int load_time = 0; load_time < 2; ++load_time) { // 0 1
            int mx = wrap_group_idx.x * 16 + (tx % 8) * 2 + load_time + k * TILE;
            // int HW_out_idx = tile_grid.x + wrap_group_idx.x * 16 + tx % 16;
            #pragma unroll
            for (int wrap_id = 0; wrap_id < 4; ++wrap_id) {// 0 - 15
                int C_out_idx = tile_grid.y +  wrap_group_idx.y * 16 + wrap_id * 4 + tx / 8;
                MA[wrap_group_idx.y * 16 + wrap_id * 4 + tx / 8][wrap_group_idx.x * 16 + (tx % 8) * 2 + load_time]= (mx < K && C_out_idx < M) ?  __float2half(weight[C_out_idx * K +mx]) : __float2half(0.0f);
                // 将全局内存数据读入到共享内存中 uint2 mma_xy -- 4 * 8 4行坐标不连续，8列坐标连续 -- 为了避免 bank conflict
                int my = wrap_group_idx.y * 16 + (wrap_id / 2) * 8 + (tx / 8) * 2 + load_time + k * TILE;
                int HW_out_idx = tile_grid.x + wrap_group_idx.x * 16 + (wrap_id % 2) * 8  + tx % 8;
                MB[wrap_group_idx.x * 16 + (wrap_id % 2) * 8  + tx % 8][wrap_group_idx.y * 16 + (wrap_id / 2) * 8 + (tx / 8) * 2 + load_time]= (HW_out_idx < N && my < K) ?  __float2half(col_ptr[HW_out_idx + my * N]) : __float2half(0.0f);
            }
        }
        __syncthreads();

        for (int km = 0; km < (TILE / 16); ++km) {
            wmma::load_matrix_sync(a_frag, &MA[wrap_group_idx.y * 16][km * 16], TILE + 8) ;
            wmma::load_matrix_sync(b_frag, &MB[wrap_group_idx.x * 16][km * 16], TILE + 8);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }

        __syncthreads();
    }
    wmma::store_matrix_sync(
        &MC[wrap_group_idx.y * 16][wrap_group_idx.x * 16], 
        c_frag, TILE, wmma::mem_row_major);

    
    #pragma unroll
    for (int wrap_id = 0; wrap_id < 8; ++wrap_id) {
        int row_y = wrap_id * 2 + tx/16;
        int C_out_idx = tile_grid.y + wrap_group_idx.y * 16 + row_y ;
        int col_x = tx % 16;
        col_x = MAPPING(row_y, col_x, 1);
        int HW_out_idx = tile_grid.x + wrap_group_idx.x * 16 + col_x;
        if (HW_out_idx < N && C_out_idx < M) {
            out_im_ptr[C_out_idx * N + HW_out_idx] = MC[wrap_group_idx.y * 16 + wrap_id * 2 + tx/16][wrap_group_idx.x * 16 + col_x];
            out_im_ptr[C_out_idx * N + HW_out_idx] += bias[C_out_idx];
        }
    }
}

__device__ float dcn_bilinear_cuda(const float *bottom_data, const int data_width,
                                      const int height, const int width, float h, float w)
{
    int h_low = floor(h);
    int w_low = floor(w);
    int h_high = h_low + 1;
    int w_high = w_low + 1;

    float lh = h - h_low;
    float lw = w - w_low;
    float hh = 1 - lh, hw = 1 - lw;

    float v1 = 0;
    if (h_low >= 0 && w_low >= 0)
        v1 = bottom_data[h_low * data_width + w_low];
    float v2 = 0;
    if (h_low >= 0 && w_high <= width - 1)
        v2 = bottom_data[h_low * data_width + w_high];
    float v3 = 0;
    if (h_high <= height - 1 && w_low >= 0)
        v3 = bottom_data[h_high * data_width + w_low];
    float v4 = 0;
    if (h_high <= height - 1 && w_high <= width - 1)
        v4 = bottom_data[h_high * data_width + w_high];

    float w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;

    float val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
    return val;
}


__global__ void im2colcuda(
        const float *input, 
        const float *mask,
        const float *offset,
        const int H, const int W, const int k_h, const int k_w,
        const int pad_h, const int pad_w,
        const int stride_h, const int stride_w,
        const int B, const int C,
        const int H_out, const int W_out,
        float *columns // B,   C * k_h * k_w(y),   1 * H_out * W_out(x)
    )
{
    int global_x = threadIdx.x + blockIdx.x * TILE_X;
    int global_y = threadIdx.y + blockIdx.y * TILE_Y;
    int B_idx = blockIdx.z;
    if((B_idx >= B) || (global_x >= (H_out * W_out)) || (global_y >= (C * k_h * k_w))) {
        return;
    }
    int C_idx = global_y / (k_h * k_w);
    int k_idx =  global_y % (k_h * k_w);
    int kh = k_idx / k_w;
    int kw = k_idx % k_w;

    int h_idx = global_x / W_out; 
    int w_idx = global_x % W_out; 
    int hin_idx = h_idx * stride_h - pad_h;
    int win_idx = w_idx * stride_w - pad_w;

    const float* data_im_ptr = input + (B_idx * C + C_idx) * H * W;
    const float *data_offset_ptr = offset + B_idx * 2 * k_h * k_w * H_out * W_out + 2 * k_idx * H_out * W_out + h_idx * W_out + w_idx;
    const float mask_val = mask[B_idx * k_h * k_w * H_out * W_out + k_idx * H_out * W_out + h_idx * W_out + w_idx];
    const float offset_h = data_offset_ptr[0];
    const float offset_w = data_offset_ptr[H_out * W_out];
    const float him = hin_idx + kh + offset_h ;
    const float wim = win_idx + kw + offset_w ;

    bool cts = (him > -1 && wim > -1 && him < H && wim < W);
    float val = 0.0f;
    if (cts) {
        val = dcn_bilinear_cuda(data_im_ptr, W, H, W, him, wim);
    }
    
    int offset_col = B_idx * C * k_w * k_h * H_out * W_out + (C_idx * k_w * k_h + k_idx) * H_out * W_out + h_idx * W_out + w_idx;
    columns[offset_col] = val * mask_val;
}


at::Tensor MyDCN_forward(
        const at::Tensor& input,
        const at::Tensor& weight,
        const at::Tensor& mask,
        const at::Tensor& offset,
        const at::Tensor& bias,
        const int stride_h,
        const int stride_w,
        const int pad_h,
        const int pad_w)
{
    AT_ASSERTM(input.is_cuda(), "input must be a CUDA tensor");
    AT_ASSERTM(weight.is_cuda(), "weight must be a CUDA tensor");
    AT_ASSERTM(bias.is_cuda(), "bias must be a CUDA tensor");
    AT_ASSERTM(offset.is_cuda(), "offset must be a CUDA tensor");
    AT_ASSERTM(mask.is_cuda(), "mask must be a CUDA tensor");
    // torch::nn::functional::pad()
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    const int B = input.size(0);
    const int C = input.size(1);
    const int H = input.size(2);
    const int W = input.size(3);

    const int C_out = weight.size(0);
    // const int channels_kernel = weight.size(1); -- C
    const int k_h = weight.size(2);
    const int k_w = weight.size(3);
    // bias.shape = C_out 
    const int H_out = (H - k_h + 2 * pad_h) / stride_h + 1;
    const int W_out = (W - k_w + 2 * pad_w) / stride_w + 1;

    // 将bias先加进去
    // auto ones = at::ones({B, bias.sizes()[0], H_out, W_out}, input.options());
    auto output = at::zeros({B, C_out, H_out, W_out}, input.options());

    // 构建输入的im2col矩阵
    auto columns = at::empty({B, C * k_h * k_w, 1 * H_out * W_out}, input.options());
    dim3 Grid0((H_out * W_out + TILE_X -1)/ TILE_X, (C * k_h * k_w + TILE_Y -1)/ TILE_Y, B); 
    dim3 Block0(TILE_X, TILE_Y);
    int K = C * k_w * k_h;
    int N = H_out * W_out;
    int M = C_out;
    dim3 Grid1((N + TILE -1)/ TILE, (M + TILE -1)/ TILE, B); 
    dim3 Block1(BLOCK_SIZE, 16);

    im2colcuda<<<Grid0, Block0, 0, stream>>>(
        input.contiguous().data_ptr<float>(), 
        mask.contiguous().data_ptr<float>(), 
        offset.contiguous().data_ptr<float>(), 
        H, W, k_h, k_w,
        pad_h, pad_w,
        stride_h, stride_w,
        B, C,
        H_out, W_out,
        columns.contiguous().data_ptr<float>()
    );
        // 进行GEMM 标准矩阵乘法
    gemm_cuda<<<Grid1, Block1, 0, stream>>> (
        weight.contiguous().data_ptr<float>(),// Cout C * k_h * k_w
        columns.contiguous().data_ptr<float>(),//B  C * k_h * k_w H_out * W_out
        output.contiguous().data_ptr<float>(),
        bias.contiguous().data_ptr<float>(),
        M,
        K,
        N
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Launch Error: " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return output;

    // 预热：执行几次 kernel 以消除初始化影响
    // for (int i = 0; i < 5; ++i) {
    //     im2colcuda<<<Grid0, Block0, 0, stream>>>(
    //         input.contiguous().data_ptr<float>(), 
    //         mask.contiguous().data_ptr<float>(), 
    //         offset.contiguous().data_ptr<float>(), 
    //         H, W, k_h, k_w,
    //         pad_h, pad_w,
    //         stride_h, stride_w,
    //         B, C,
    //         H_out, W_out,
    //         columns.contiguous().data_ptr<float>()
    //     );
    //     // 进行GEMM 标准矩阵乘法
    //     gemm_cuda<<<Grid1, Block1, 0, stream >>> (
    //         weight.contiguous().data_ptr<float>(),// Cout C * k_h * k_w
    //         columns.contiguous().data_ptr<float>(),//B  C * k_h * k_w H_out * W_out
    //         output.contiguous().data_ptr<float>(),
    //         bias.contiguous().data_ptr<float>(),
    //         M,
    //         K,
    //         N
    //     );

    //     // libtorch matrix multiply
    //     // auto ones = at::ones({B, bias.sizes()[0], H_out, W_out}, input.options());
    //     // auto ones_T = at::transpose(ones.contiguous(), 3, 1);
    //     // ones_T = at::mul(ones_T, bias.contiguous());
    //     // ones_T = at::transpose(ones_T, 3, 1);
    //     // output = at::zeros({B, C_out, H_out, W_out}, input.options());
    //     // output = at::add(output, ones_T);
    //     // auto weight_flat = weight.view({C_out, C * k_h * k_w});
    //     // auto product = at::matmul(weight_flat, columns);
    //     // output = at::add(output, product.view({B, C_out, H_out, W_out}));
    // }
    // cudaDeviceSynchronize();

    // output = at::zeros({B, C_out, H_out, W_out}, input.options());
    // columns = at::empty({B, C * k_h * k_w, 1 * H_out * W_out}, input.options());

    // // 创建 CUDA 事件用于计时
    // cudaEvent_t start, stop;
    // cudaEventCreate(&start);
    // cudaEventCreate(&stop);

    // // 开始计时
    // float num_iterations = 100;
    // cudaEventRecord(start);
    // for (int i = 0; i < num_iterations; ++i) {
    //     im2colcuda<<<Grid0, Block0, 0, stream>>>(
    //         input.contiguous().data_ptr<float>(), 
    //         mask.contiguous().data_ptr<float>(), 
    //         offset.contiguous().data_ptr<float>(), 
    //         H, W, k_h, k_w,
    //         pad_h, pad_w,
    //         stride_h, stride_w,
    //         B, C,
    //         H_out, W_out,
    //         columns.contiguous().data_ptr<float>()
    //     );
    //     // 进行GEMM 标准矩阵乘法
    //     gemm_cuda<<<Grid1, Block1, 0, stream >>> (
    //         weight.contiguous().data_ptr<float>(),// Cout C * k_h * k_w
    //         columns.contiguous().data_ptr<float>(),//B  C * k_h * k_w H_out * W_out
    //         output.contiguous().data_ptr<float>(),
    //         bias.contiguous().data_ptr<float>(),
    //         M,
    //         K,
    //         N
    //     );

    //     // auto ones = at::ones({B, bias.sizes()[0], H_out, W_out}, input.options());
    //     // auto ones_T = at::transpose(ones.contiguous(), 3, 1);
    //     // ones_T = at::mul(ones_T, bias.contiguous());
    //     // ones_T = at::transpose(ones_T, 3, 1);
    //     // output = at::zeros({B, C_out, H_out, W_out}, input.options());
    //     // output = at::add(output, ones_T);
    //     // auto weight_flat = weight.view({C_out, C * k_h * k_w});
    //     // auto product = at::matmul(weight_flat, columns);
    //     // output = at::add(output, product.view({B, C_out, H_out, W_out}));
    // }


    // cudaEventRecord(stop);
    // cudaEventSynchronize(stop);

    // // 计算平均执行时间（毫秒）
    // float elapsed_ms;
    // cudaEventElapsedTime(&elapsed_ms, start, stop);
    // double avg_time_s = (elapsed_ms / 1000) / num_iterations;

    // // 计算一次卷积的浮点操作数（乘加各一次，共2次）
    // double flops_per_kernel = B * H_out * W_out * C_out * (10.0 * k_h * k_w * C + 2.0 *  k_h * k_w);

    // // GFLOPS = 浮点操作数 / 时间(秒) / 1e9
    // double gflops = flops_per_kernel / (avg_time_s * 1e9);

    // // 打印结果
    // std::cout << "Convolution: B=" << B << ", C=" << C << ", H=" << H << ", W=" << W
    //         << ", C_out=" << C_out << ", kernel=" << k_h << "x" << k_w
    //         << ", stride=" << stride_h << "x" << stride_w
    //         << ", H_out=" << H_out << ", W_out=" << W_out << std::endl;
    // std::cout << "Time per kernel: " << avg_time_s * 1000.0 << " ms, "
    //         << "GFLOPS: " << gflops << std::endl;

    // // 清理事件
    // cudaEventDestroy(start);
    // cudaEventDestroy(stop);
    
    // return output;
}

// ------------------------- dcnv2 backward ---------------------- //

__device__ float dmcn_get_coordinate_weight_cuda(float argmax_h, float argmax_w,
                                            const int height, const int width, const float *im_data,
                                            const int data_width, const int bp_dir)
{
  if (argmax_h <= -1 || argmax_h >= height || argmax_w <= -1 || argmax_w >= width)
  {
    //empty
    return 0;
  }

  int argmax_h_low = floor(argmax_h);
  int argmax_w_low = floor(argmax_w);
  int argmax_h_high = argmax_h_low + 1;
  int argmax_w_high = argmax_w_low + 1;

  float weight = 0;

  if (bp_dir == 0)
  {
    if (argmax_h_low >= 0 && argmax_w_low >= 0)
      weight += -1 * (argmax_w_low + 1 - argmax_w) * im_data[argmax_h_low * data_width + argmax_w_low];
    if (argmax_h_low >= 0 && argmax_w_high <= width - 1)
      weight += -1 * (argmax_w - argmax_w_low) * im_data[argmax_h_low * data_width + argmax_w_high];
    if (argmax_h_high <= height - 1 && argmax_w_low >= 0)
      weight += (argmax_w_low + 1 - argmax_w) * im_data[argmax_h_high * data_width + argmax_w_low];
    if (argmax_h_high <= height - 1 && argmax_w_high <= width - 1)
      weight += (argmax_w - argmax_w_low) * im_data[argmax_h_high * data_width + argmax_w_high];
  }
  else if (bp_dir == 1)
  {
    if (argmax_h_low >= 0 && argmax_w_low >= 0)
      weight += -1 * (argmax_h_low + 1 - argmax_h) * im_data[argmax_h_low * data_width + argmax_w_low];
    if (argmax_h_low >= 0 && argmax_w_high <= width - 1)
      weight += (argmax_h_low + 1 - argmax_h) * im_data[argmax_h_low * data_width + argmax_w_high];
    if (argmax_h_high <= height - 1 && argmax_w_low >= 0)
      weight += -1 * (argmax_h - argmax_h_low) * im_data[argmax_h_high * data_width + argmax_w_low];
    if (argmax_h_high <= height - 1 && argmax_w_high <= width - 1)
      weight += (argmax_h - argmax_h_low) * im_data[argmax_h_high * data_width + argmax_w_high];
  }

  return weight;
}

__global__ void modulated_deformable_col2im_coord_gpu_kernel(const int n,
                                                             const float *data_col, const float *data_im,
                                                             const float *data_offset, const float *data_mask,
                                                             const int channels, const int height, const int width,
                                                             const int kernel_h, const int kernel_w,
                                                             const int pad_h, const int pad_w,
                                                             const int stride_h, const int stride_w,
                                                             const int channel_per_deformable_group,
                                                             const int batch_size, const int offset_channels,
                                                             const int height_col, const int width_col,
                                                             float *grad_offset, float *grad_mask)
{
  CUDA_KERNEL_LOOP(index, n)
  {
    float val = 0, mval = 0;
    int w = index % width_col;
    int h = (index / width_col) % height_col;
    int c = (index / width_col / height_col) % offset_channels;
    int b = (index / width_col / height_col) / offset_channels;
    // compute the start and end of the output

    const int deformable_group_index = c / (2 * kernel_h * kernel_w);
    const int col_step = kernel_h * kernel_w;
    int cnt = 0;
    const float *data_col_ptr = data_col + deformable_group_index * channel_per_deformable_group * batch_size * width_col * height_col;
    const float *data_im_ptr = data_im + (b + deformable_group_index) * channel_per_deformable_group / kernel_h / kernel_w * height * width;
    const float *data_offset_ptr = data_offset + (b  + deformable_group_index) * 2 * kernel_h * kernel_w * height_col * width_col;
    const float *data_mask_ptr = data_mask + (b  + deformable_group_index) * kernel_h * kernel_w * height_col * width_col;

    const int offset_c = c - deformable_group_index * 2 * kernel_h * kernel_w;

    for (int col_c = (offset_c / 2); col_c < channel_per_deformable_group; col_c += col_step)
    {
      const int col_pos = (((col_c * batch_size + b) * height_col) + h) * width_col + w;
      const int bp_dir = offset_c % 2;

      int j = (col_pos / width_col / height_col / batch_size) % kernel_w;
      int i = (col_pos / width_col / height_col / batch_size / kernel_w) % kernel_h;
      int w_out = col_pos % width_col;
      int h_out = (col_pos / width_col) % height_col;
      int w_in = w_out * stride_w - pad_w;
      int h_in = h_out * stride_h - pad_h;
      const int data_offset_h_ptr = (((2 * (i * kernel_w + j)) * height_col + h_out) * width_col + w_out);
      const int data_offset_w_ptr = (((2 * (i * kernel_w + j) + 1) * height_col + h_out) * width_col + w_out);
      const int data_mask_hw_ptr = (((i * kernel_w + j) * height_col + h_out) * width_col + w_out);
      const float offset_h = data_offset_ptr[data_offset_h_ptr];
      const float offset_w = data_offset_ptr[data_offset_w_ptr];
      const float mask = data_mask_ptr[data_mask_hw_ptr];
      float inv_h = h_in + i + offset_h;
      float inv_w = w_in + j + offset_w;
      if (inv_h <= -1 || inv_w <= -1 || inv_h >= height || inv_w >= width)
      {
        inv_h = inv_w = -2;
      }
      else
      {
        mval += data_col_ptr[col_pos] * dcn_bilinear_cuda(data_im_ptr + cnt * height * width, width, height, width, inv_h, inv_w);
      }
      const float weight = dmcn_get_coordinate_weight_cuda(
          inv_h, inv_w,
          height, width, data_im_ptr + cnt * height * width, width, bp_dir);
      val += weight * data_col_ptr[col_pos] * mask;
      cnt += 1;
    }
    // KERNEL_ASSIGN(grad_offset[index], offset_req, val);
    grad_offset[index] = val;
    if (offset_c % 2 == 0)
      // KERNEL_ASSIGN(grad_mask[(((b * deformable_group + deformable_group_index) * kernel_h * kernel_w + offset_c / 2) * height_col + h) * width_col + w], mask_req, mval);
      grad_mask[(((b + deformable_group_index) * kernel_h * kernel_w + offset_c / 2) * height_col + h) * width_col + w] = mval;
  }
}

void modulated_deformable_col2im_coord_cuda(cudaStream_t stream,
  const float* data_col, const float* data_im, const float* data_offset, const float* data_mask,
  const int batch_size, const int channels, const int height_im, const int width_im, 
  const int height_col, const int width_col, const int kernel_h, const int kernel_w,
  const int pad_h, const int pad_w, const int stride_h, const int stride_w, 
  float* grad_offset, float* grad_mask) {
  const int num_kernels = batch_size * height_col * width_col * 2 * kernel_h * kernel_w;
  const int channel_per_deformable_group = channels * kernel_h * kernel_w;
  modulated_deformable_col2im_coord_gpu_kernel
      <<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS,
        0, stream>>>(
        num_kernels, data_col, data_im, data_offset, data_mask, channels, height_im, width_im,
        kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w,
        channel_per_deformable_group,
        batch_size, 2 * kernel_h * kernel_w, height_col, width_col, 
        grad_offset, grad_mask);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    printf("error in modulated_deformable_col2im_coord_cuda: %s\n", cudaGetErrorString(err));
  }
}

__device__ float dmcn_get_gradient_weight_cuda(float argmax_h, float argmax_w,
                                          const int h, const int w, const int height, const int width)
{
  if (argmax_h <= -1 || argmax_h >= height || argmax_w <= -1 || argmax_w >= width)
  {
    //empty
    return 0;
  }

  int argmax_h_low = floor(argmax_h);
  int argmax_w_low = floor(argmax_w);
  int argmax_h_high = argmax_h_low + 1;
  int argmax_w_high = argmax_w_low + 1;

  float weight = 0;
  if (h == argmax_h_low && w == argmax_w_low)
    weight = (h + 1 - argmax_h) * (w + 1 - argmax_w);
  if (h == argmax_h_low && w == argmax_w_high)
    weight = (h + 1 - argmax_h) * (argmax_w + 1 - w);
  if (h == argmax_h_high && w == argmax_w_low)
    weight = (argmax_h + 1 - h) * (w + 1 - argmax_w);
  if (h == argmax_h_high && w == argmax_w_high)
    weight = (argmax_h + 1 - h) * (argmax_w + 1 - w);
  return weight;
}


__global__ void modulated_deformable_col2im_gpu_kernel(const int n,
                                                       const float *data_col, const float *data_offset, const float *data_mask,
                                                       const int channels, const int height, const int width,
                                                       const int kernel_h, const int kernel_w,
                                                       const int pad_h, const int pad_w,
                                                       const int stride_h, const int stride_w,
                                                       const int channel_per_deformable_group,
                                                       const int batch_size,
                                                       const int height_col, const int width_col,
                                                       float *grad_im)
{
  CUDA_KERNEL_LOOP(index, n)
  {
    const int j = (index / width_col / height_col / batch_size) % kernel_w;
    const int i = (index / width_col / height_col / batch_size / kernel_w) % kernel_h;
    const int c = index / width_col / height_col / batch_size / kernel_w / kernel_h;
    // compute the start and end of the output

    const int deformable_group_index = c / channel_per_deformable_group;

    int w_out = index % width_col;
    int h_out = (index / width_col) % height_col;
    int b = (index / width_col / height_col) % batch_size;
    int w_in = w_out * stride_w - pad_w;
    int h_in = h_out * stride_h - pad_h;

    const float *data_offset_ptr = data_offset + (b  + deformable_group_index) * 2 * kernel_h * kernel_w * height_col * width_col;
    const float *data_mask_ptr = data_mask + (b  + deformable_group_index) * kernel_h * kernel_w * height_col * width_col;
    const int data_offset_h_ptr = ((2 * (i * kernel_w + j)) * height_col + h_out) * width_col + w_out;
    const int data_offset_w_ptr = ((2 * (i * kernel_w + j) + 1) * height_col + h_out) * width_col + w_out;
    const int data_mask_hw_ptr = ((i * kernel_w + j) * height_col + h_out) * width_col + w_out;
    const float offset_h = data_offset_ptr[data_offset_h_ptr];
    const float offset_w = data_offset_ptr[data_offset_w_ptr];
    const float mask = data_mask_ptr[data_mask_hw_ptr];
    const float cur_inv_h_data = h_in + i + offset_h;
    const float cur_inv_w_data = w_in + j + offset_w;

    const float cur_top_grad = data_col[index] * mask;
    const int cur_h = (int)cur_inv_h_data;
    const int cur_w = (int)cur_inv_w_data;
    for (int dy = -2; dy <= 2; ++dy)
    {
      for (int dx = -2; dx <= 2; ++dx)
      {
        if (cur_h + dy >= 0 && cur_h + dy < height &&
            cur_w + dx >= 0 && cur_w + dx < width &&
            abs(cur_inv_h_data - (cur_h + dy)) < 1 &&
            abs(cur_inv_w_data - (cur_w + dx)) < 1)
        {
          int cur_bottom_grad_pos = ((b * channels + c) * height + cur_h + dy) * width + cur_w + dx;
          float weight = dmcn_get_gradient_weight_cuda(cur_inv_h_data, cur_inv_w_data, cur_h + dy, cur_w + dx, height, width);
          atomicAdd(grad_im + cur_bottom_grad_pos, weight * cur_top_grad);
        }
      }
    }
  }
}

void modulated_deformable_col2im_cuda(cudaStream_t stream,
  const float* data_col, const float* data_offset, const float* data_mask,
  const int batch_size, const int channels, const int height_im, const int width_im, 
  const int height_col, const int width_col, const int kernel_h, const int kernel_w,
  const int pad_h, const int pad_w, const int stride_h, const int stride_w, 
  float* grad_im)
{

  const int channel_per_deformable_group = channels;
  const int num_kernels = channels * kernel_h * kernel_w * batch_size * height_col * width_col;
  modulated_deformable_col2im_gpu_kernel
      <<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS,
          0, stream>>>(
        num_kernels, data_col, data_offset, data_mask, channels, height_im, width_im,
        kernel_h, kernel_w, pad_h, pad_h, stride_h, stride_w,
        channel_per_deformable_group,
        batch_size, height_col, width_col, grad_im);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    printf("error in modulated_deformable_col2im_cuda: %s\n", cudaGetErrorString(err));
  }

}


__global__ void modulated_deformable_im2col_gpu_kernel(const int n,
                                                       const float *data_im, const float *data_offset, const float *data_mask,
                                                       const int height, const int width, const int kernel_h, const int kernel_w,
                                                       const int pad_h, const int pad_w,
                                                       const int stride_h, const int stride_w,
                                                       const int channel_per_deformable_group,
                                                       const int batch_size, const int num_channels,
                                                       const int height_col, const int width_col,
                                                       float *data_col)
{
  // launch channels * batch_size * height_col * width_col cores
  CUDA_KERNEL_LOOP(index, n)
  {
    // NOTE(CharlesShang): different from Dai Jifeng's MXNet implementation, col_buffer is of shape (c*kw*kh, N, oh, ow)
    // here columns is of shape (N, c*kw*kh, oh * ow), need to adapt axis

    // index index of output matrix
    const int w_col = index % width_col;
    const int h_col = (index / width_col) % height_col;
    // const int b_col = (index / width_col / height_col) % batch_size;
    const int b_col = (index / width_col / height_col / num_channels) % batch_size;
    // const int c_im = (index / width_col / height_col) / batch_size;
    const int c_im = (index / width_col / height_col) % num_channels;
    // const int c_col = c_im * kernel_h * kernel_w;
    const int c_col = c_im * kernel_h * kernel_w;

    // compute deformable group index
    const int deformable_group_index = c_im / channel_per_deformable_group;

    const int h_in = h_col * stride_h - pad_h;
    const int w_in = w_col * stride_w - pad_w;

    //  float *data_col_ptr = data_col + ((c_col * batch_size + b_col) * height_col + h_col) * width_col + w_col;
    float *data_col_ptr = data_col + ((b_col * num_channels * kernel_w * kernel_h + c_col) * height_col + h_col) * width_col + w_col;
    //const float* data_im_ptr = data_im + ((b_col * num_channels + c_im) * height + h_in) * width + w_in;
    const float *data_im_ptr = data_im + (b_col * num_channels + c_im) * height * width;
    const float *data_offset_ptr = data_offset + (b_col  + deformable_group_index) * 2 * kernel_h * kernel_w * height_col * width_col;

    const float *data_mask_ptr = data_mask + (b_col  + deformable_group_index) * kernel_h * kernel_w * height_col * width_col;

    for (int i = 0; i < kernel_h; ++i)
    {
      for (int j = 0; j < kernel_w; ++j)
      {
        const int data_offset_h_ptr = ((2 * (i * kernel_w + j)) * height_col + h_col) * width_col + w_col;
        const int data_offset_w_ptr = ((2 * (i * kernel_w + j) + 1) * height_col + h_col) * width_col + w_col;
        const int data_mask_hw_ptr = ((i * kernel_w + j) * height_col + h_col) * width_col + w_col;
        const float offset_h = data_offset_ptr[data_offset_h_ptr];
        const float offset_w = data_offset_ptr[data_offset_w_ptr];
        const float mask = data_mask_ptr[data_mask_hw_ptr];
        float val = static_cast<float>(0);
        const float h_im = h_in + i + offset_h;
        const float w_im = w_in + j + offset_w;
        //if (h_im >= 0 && w_im >= 0 && h_im < height && w_im < width) {
        if (h_im > -1 && w_im > -1 && h_im < height && w_im < width)
        {
          //const float map_h = i  + offset_h;
          //const float map_w = j  + offset_w;
          //const int cur_height = height - h_in;
          //const int cur_width = width - w_in;
          //val = dcn_bilinear_cuda(data_im_ptr, width, cur_height, cur_width, map_h, map_w);
          val = dcn_bilinear_cuda(data_im_ptr, width, height, width, h_im, w_im);
        }
        *data_col_ptr = val * mask;
        // data_col_ptr += batch_size * height_col * width_col;
        data_col_ptr += height_col * width_col;
      }
    }
  }
}

void modulated_deformable_im2col_cuda(cudaStream_t stream,
  const float* data_im, const float* data_offset, const float* data_mask,
  const int batch_size, const int channels, const int height_im, const int width_im, 
  const int height_col, const int width_col, const int kernel_h, const int kernel_w,
  const int pad_h, const int pad_w, const int stride_h, const int stride_w, 
  float* data_col) {
  // num_axes should be smaller than block size
  const int channel_per_deformable_group = channels;
  const int num_kernels = channels * batch_size * height_col * width_col;
  modulated_deformable_im2col_gpu_kernel
      <<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS,
          0, stream>>>(
      num_kernels, data_im, data_offset, data_mask, height_im, width_im, kernel_h, kernel_w,
      pad_h, pad_w, stride_h, stride_w, channel_per_deformable_group,
      batch_size, channels, height_col, width_col, data_col);
  
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    printf("error in modulated_deformable_im2col_cuda: %s\n", cudaGetErrorString(err));
  }

}


std::vector<at::Tensor> MyDCN_backward(const at::Tensor &input,
                                             const at::Tensor &weight,
                                             const at::Tensor &mask,
                                             const at::Tensor &offset,
                                             const at::Tensor &bias,
                                             const at::Tensor &grad_output,
                                             const int stride_h, const int stride_w,
                                             const int pad_h, const int pad_w
                                            )
{
    AT_ASSERTM(input.is_cuda(), "input must be a CUDA tensor");
    AT_ASSERTM(weight.is_cuda(), "weight must be a CUDA tensor");
    AT_ASSERTM(bias.is_cuda(), "bias must be a CUDA tensor");
    AT_ASSERTM(offset.is_cuda(), "offset must be a CUDA tensor");
    AT_ASSERTM(mask.is_cuda(), "mask must be a CUDA tensor");
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    const int batch = input.size(0);
    const int channels = input.size(1);
    const int height = input.size(2);
    const int width = input.size(3);

    const int channels_out = weight.size(0);
    const int channels_kernel = weight.size(1);
    const int kernel_h = weight.size(2);
    const int kernel_w = weight.size(3);

    AT_ASSERTM(channels == channels_kernel,
               "Input shape and kernel channels wont match: (%d vs %d).", channels, channels_kernel);

    const int height_out = (height - kernel_h + 2 * pad_h) / stride_h + 1;
    const int width_out = (width - kernel_w + 2 * pad_w) / stride_w + 1;

    auto ones = at::ones({height_out, width_out}, input.options());
    auto columns = at::empty({channels * kernel_h * kernel_w, 1 * height_out * width_out}, input.options());
    auto output = at::empty({batch, channels_out, height_out, width_out}, input.options());

    auto grad_input = at::zeros_like(input);
    auto grad_weight = at::zeros_like(weight);
    auto grad_bias = at::zeros_like(bias);
    auto grad_offset = at::zeros_like(offset);
    auto grad_mask = at::zeros_like(mask);

    using scalar_t = float;

    for (int b = 0; b < batch; ++b)
    {
        auto input_n = input.select(0, b);
        auto offset_n = offset.select(0, b);
        auto mask_n = mask.select(0, b);
        auto grad_output_n = grad_output.select(0, b);
        auto grad_input_n = grad_input.select(0, b);
        auto grad_offset_n = grad_offset.select(0, b);
        auto grad_mask_n = grad_mask.select(0, b);

        // Torch implementation
        auto weight_flat = weight.view({channels_out, channels*kernel_h*kernel_w});
        weight_flat = at::transpose(weight_flat, 1, 0);

        auto grad_output_n_flat = grad_output_n.view({channels_out, height_out*width_out});
        columns = at::matmul(weight_flat, grad_output_n_flat);

        // gradient w.r.t. input coordinate data
        modulated_deformable_col2im_coord_cuda(stream,
                                               columns.data_ptr<scalar_t>(),
                                               input_n.data_ptr<scalar_t>(),
                                               offset_n.data_ptr<scalar_t>(),
                                               mask_n.data_ptr<scalar_t>(),
                                               1, channels, height, width,
                                               height_out, width_out, kernel_h, kernel_w,
                                               pad_h, pad_w, stride_h, stride_w,
                                               grad_offset_n.data_ptr<scalar_t>(),
                                               grad_mask_n.data_ptr<scalar_t>());
        // cudaStreamSynchronize(stream);
        // gradient w.r.t. input data
        modulated_deformable_col2im_cuda(stream,
                                         columns.data_ptr<scalar_t>(),
                                         offset_n.data_ptr<scalar_t>(),
                                         mask_n.data_ptr<scalar_t>(),
                                         1, channels, height, width,
                                         height_out, width_out, kernel_h, kernel_w,
                                         pad_h, pad_w, stride_h, stride_w,
                                         grad_input_n.data_ptr<scalar_t>());

        // gradient w.r.t. weight, dWeight should accumulate across the batch and group
        modulated_deformable_im2col_cuda(stream,
                                         input_n.data_ptr<scalar_t>(),
                                         offset_n.data_ptr<scalar_t>(),
                                         mask_n.data_ptr<scalar_t>(),
                                         1, channels, height, width,
                                         height_out, width_out, kernel_h, kernel_w,
                                         pad_h, pad_w, stride_h, stride_w,
                                         columns.data_ptr<scalar_t>());


       // Torch implementation
        auto product = at::matmul(grad_output_n_flat, at::transpose(columns, 1, 0));
        grad_weight = at::add(grad_weight, product.view({channels_out, channels, kernel_h, kernel_w}));

        // Torch implementation
        auto ones_flat = ones.view({height_out*width_out});
        product = at::matmul(grad_output_n_flat, ones_flat);
        grad_bias = at::add(grad_bias, product);

    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Launch Error: " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return {
        grad_input, grad_offset, grad_mask, grad_weight, grad_bias
    };
}