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
/*
block size 32 16,  wrap size 32, tile size 64 * 64
*/
// grid_size (N + tile_x - 1) / tile_x, (M + tile_x - 1) / tile_x, B;  block size 32 * 16;  wrap size 32, tile size 64 * 64


__global__ void gemm_weight_grad_cuda(
    const half* grad_output, const half* columns,
    float* grad_weight, const int M, const int K, const int N
)
{
    int bx =  blockIdx.x;
    int by =  blockIdx.y;
    int tx = threadIdx.x; // 0-31 thread idx per wrap
    int ty = threadIdx.y; // 0-15 wrap parallel
    int b_idx = blockIdx.z;
    // tile grid coord for output size
    int2 tile_grid = {bx * TILE, by * TILE};
  
    int2 wrap_group_idx = {ty % 4, ty / 4};

    const half* weight = grad_output + b_idx * M * K;//    co,  ho * wo
    float* out_im_ptr = grad_weight + b_idx * M * N;//     co,  c * k1 * k2    
    const half* col_ptr = columns + b_idx * K * N; // ho * wo,  c * k1 * k2 

    extern __shared__ uint8_t smem [];
    half* MA = reinterpret_cast<half*>(smem);
    half* MB = reinterpret_cast<half*>(smem + 2 * TILE * (TILE + 8) * sizeof(half));
    float* MC = reinterpret_cast<float*>(smem);

    int round = (K + TILE - 1) / TILE;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    int read_state = 0;
    int write_state = 0;

    int k = 0;
    {
        for (int load_time = 0; load_time < 2; ++load_time) { // 0 1
            int mx = wrap_group_idx.x * 16 + (tx % 8) * 2 + load_time + k * TILE;
            #pragma unroll
            for (int wrap_id = 0; wrap_id < 4; ++wrap_id) {// 0 - 15
                int C_out_idx = tile_grid.y +  wrap_group_idx.y * 16 + wrap_id * 4 + tx / 8;
                MA[write_state * TILE * (TILE + 8) + (wrap_group_idx.y * 16 + wrap_id * 4 + tx / 8) * (TILE + 8) + 
                    wrap_group_idx.x * 16 + (tx % 8) * 2 + load_time]= 
                    (mx < K && C_out_idx < M) ?  weight[C_out_idx * K +mx] : __float2half(0.0f);
                int my = wrap_group_idx.y * 16 + (wrap_id / 2) * 8 + (tx / 8) * 2 + load_time + k * TILE;
                int HW_out_idx = tile_grid.x + wrap_group_idx.x * 16 + (wrap_id % 2) * 8  + tx % 8;
                MB[write_state * TILE * (TILE + 8) + (wrap_group_idx.x * 16 + (wrap_id % 2) * 8  + tx % 8) * (TILE + 8) + 
                    wrap_group_idx.y * 16 + (wrap_id / 2) * 8 + (tx / 8) * 2 + load_time]= 
                    (HW_out_idx < N && my < K) ?  col_ptr[HW_out_idx + my * N] : __float2half(0.0f);
            }
        }
        
    }
    __syncthreads();
    
    for (int k=0; k < round; ++k){
        read_state = write_state;
        write_state ^= 1;
        if ( k + 1 < round) {
            int next_k = k + 1;
            for (int load_time = 0; load_time < 2; ++load_time) { // 0 1
                int mx = wrap_group_idx.x * 16 + (tx % 8) * 2 + load_time + next_k * TILE;
                // int HW_out_idx = tile_grid.x + wrap_group_idx.x * 16 + tx % 16;
                #pragma unroll
                for (int wrap_id = 0; wrap_id < 4; ++wrap_id) {// 0 - 15
                    int C_out_idx = tile_grid.y +  wrap_group_idx.y * 16 + wrap_id * 4 + tx / 8;
                    MA[write_state * TILE * (TILE + 8) + (wrap_group_idx.y * 16 + wrap_id * 4 + tx / 8) * (TILE + 8) + 
                        wrap_group_idx.x * 16 + (tx % 8) * 2 + load_time]= 
                        (mx < K && C_out_idx < M) ?  weight[C_out_idx * K +mx] : __float2half(0.0f);
                    
                    int my = wrap_group_idx.y * 16 + (wrap_id / 2) * 8 + (tx / 8) * 2 + load_time + next_k * TILE;
                    int HW_out_idx = tile_grid.x + wrap_group_idx.x * 16 + (wrap_id % 2) * 8  + tx % 8;
                    MB[write_state * TILE * (TILE + 8) + (wrap_group_idx.x * 16 + (wrap_id % 2) * 8  + tx % 8) * (TILE + 8) + 
                        wrap_group_idx.y * 16 + (wrap_id / 2) * 8 + (tx / 8) * 2 + load_time]= 
                        (HW_out_idx < N && my < K) ? col_ptr[HW_out_idx + my * N] : __float2half(0.0f);
                }
            }
        }
 
        for (int km = 0; km < (TILE / 16); ++km) {
            wmma::load_matrix_sync(a_frag, MA + read_state * TILE * (TILE + 8) + wrap_group_idx.y * 16 * (TILE + 8) + km * 16, TILE + 8) ;
            wmma::load_matrix_sync(b_frag, MB + read_state * TILE * (TILE + 8) + wrap_group_idx.x * 16 * (TILE + 8) + km * 16, TILE + 8);

            // 6. 张量核乘加
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
        __syncthreads();
    }
    wmma::store_matrix_sync(
        MC + wrap_group_idx.y * 16 * TILE + wrap_group_idx.x * 16, 
        c_frag, TILE, wmma::mem_row_major);

    #pragma unroll
    for (int wrap_id = 0; wrap_id < 8; ++wrap_id) {
        int C_out_idx = tile_grid.y +  wrap_group_idx.y * 16 + wrap_id * 2 + tx/16;
        int HW_out_idx = tile_grid.x + wrap_group_idx.x * 16 + tx % 16;
        if (HW_out_idx < N && C_out_idx < M) {
            out_im_ptr[C_out_idx * N + HW_out_idx] = MC[(wrap_group_idx.y * 16 + wrap_id * 2 + tx/16) * TILE + 
              wrap_group_idx.x * 16 + tx % 16];
        }
    }
}

//  K M * b M N -> b K N
__global__ void gemm_column_grad_cuda(
    const half* weight, const half* grad_output,
    float* grad_columns_ptr, const int M, const int K, const int N
)
{
    int bx =  blockIdx.x;
    int by =  blockIdx.y;
    int tx = threadIdx.x; // 0-31 thread idx per wrap
    int ty = threadIdx.y; // 0-15 wrap parallel
    int b_idx = blockIdx.z;
    // tile grid coord for output size
    int2 tile_grid = {bx * TILE, by * TILE};
  
    int2 wrap_group_idx = {ty % 4, ty / 4};

    float* out_im_ptr = grad_columns_ptr + b_idx * M * N;//     co,  c * k1 * k2 -- C 
    const half* col_ptr = grad_output + b_idx * K * N; // ho * wo,  c * k1 * k2 -- B

    extern __shared__ uint8_t smem [];
    half* MA = reinterpret_cast<half*>(smem);
    half* MB = reinterpret_cast<half*>(smem + 2 * TILE * (TILE + 8) * sizeof(half));
    float* MC = reinterpret_cast<float*>(smem);

    int round = (K + TILE - 1) / TILE;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    int read_state = 0;
    int write_state = 0;

    int k = 0;
    {
        for (int load_time = 0; load_time < 2; ++load_time) { // 0 1
            int mx = wrap_group_idx.x * 16 + (tx % 8) * 2 + load_time + k * TILE;
            #pragma unroll
            for (int wrap_id = 0; wrap_id < 4; ++wrap_id) {// 0 - 15
                int C_out_idx = tile_grid.y +  wrap_group_idx.y * 16 + wrap_id * 4 + tx / 8;
                MA[write_state * TILE * (TILE + 8) + (wrap_group_idx.y * 16 + wrap_id * 4 + tx / 8) * (TILE + 8) + 
                    wrap_group_idx.x * 16 + (tx % 8) * 2 + load_time]= 
                    (mx < K && C_out_idx < M) ?  weight[C_out_idx * K +mx] : __float2half(0.0f);
                int my = wrap_group_idx.y * 16 + (wrap_id / 2) * 8 + (tx / 8) * 2 + load_time + k * TILE;
                int HW_out_idx = tile_grid.x + wrap_group_idx.x * 16 + (wrap_id % 2) * 8  + tx % 8;
                MB[write_state * TILE * (TILE + 8) + (wrap_group_idx.x * 16 + (wrap_id % 2) * 8  + tx % 8) * (TILE + 8) + 
                    wrap_group_idx.y * 16 + (wrap_id / 2) * 8 + (tx / 8) * 2 + load_time]= 
                    (HW_out_idx < N && my < K) ?  col_ptr[HW_out_idx + my * N] : __float2half(0.0f);
            }
        }
        
    }
    __syncthreads();
    
    for (int k=0; k < round; ++k){
        read_state = write_state;
        write_state ^= 1;
        if ( k + 1 < round) {
            int next_k = k + 1;
            for (int load_time = 0; load_time < 2; ++load_time) { // 0 1
                int mx = wrap_group_idx.x * 16 + (tx % 8) * 2 + load_time + next_k * TILE;
                // int HW_out_idx = tile_grid.x + wrap_group_idx.x * 16 + tx % 16;
                #pragma unroll
                for (int wrap_id = 0; wrap_id < 4; ++wrap_id) {// 0 - 15
                    int C_out_idx = tile_grid.y +  wrap_group_idx.y * 16 + wrap_id * 4 + tx / 8;
                    MA[write_state * TILE * (TILE + 8) + (wrap_group_idx.y * 16 + wrap_id * 4 + tx / 8) * (TILE + 8) + 
                        wrap_group_idx.x * 16 + (tx % 8) * 2 + load_time]= 
                        (mx < K && C_out_idx < M) ?  weight[C_out_idx * K +mx] : __float2half(0.0f);
                    
                    int my = wrap_group_idx.y * 16 + (wrap_id / 2) * 8 + (tx / 8) * 2 + load_time + next_k * TILE;
                    int HW_out_idx = tile_grid.x + wrap_group_idx.x * 16 + (wrap_id % 2) * 8  + tx % 8;
                    MB[write_state * TILE * (TILE + 8) + (wrap_group_idx.x * 16 + (wrap_id % 2) * 8  + tx % 8) * (TILE + 8) + 
                        wrap_group_idx.y * 16 + (wrap_id / 2) * 8 + (tx / 8) * 2 + load_time]= 
                        (HW_out_idx < N && my < K) ? col_ptr[HW_out_idx + my * N] : __float2half(0.0f);
                }
            }
        }
 
        for (int km = 0; km < (TILE / 16); ++km) {
            wmma::load_matrix_sync(a_frag, MA + read_state * TILE * (TILE + 8) + wrap_group_idx.y * 16 * (TILE + 8) + km * 16, TILE + 8) ;
            wmma::load_matrix_sync(b_frag, MB + read_state * TILE * (TILE + 8) + wrap_group_idx.x * 16 * (TILE + 8) + km * 16, TILE + 8);

            // 6. 张量核乘加
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
        __syncthreads();
    }
    wmma::store_matrix_sync(
        MC + wrap_group_idx.y * 16 * (TILE) + wrap_group_idx.x * 16, 
        c_frag, TILE, wmma::mem_row_major);

    #pragma unroll
    for (int wrap_id = 0; wrap_id < 8; ++wrap_id) {
        int C_out_idx = tile_grid.y +  wrap_group_idx.y * 16 + wrap_id * 2 + tx/16;
        int HW_out_idx = tile_grid.x + wrap_group_idx.x * 16 + tx % 16;
        if (HW_out_idx < N && C_out_idx < M) {
            out_im_ptr[C_out_idx * N + HW_out_idx] = MC[(wrap_group_idx.y * 16 + wrap_id * 2 + tx/16) * TILE + 
              wrap_group_idx.x * 16 + tx % 16];
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
        half *columns // B,   C * k_h * k_w(y),   1 * H_out * W_out(x)
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
    float him = hin_idx + kh + offset_h ;
    float wim = win_idx + kw + offset_w ;

    bool cts = (him > -1 && wim > -1 && him < H && wim < W);
    float val = 0.0f;
    if (cts) {
        val = dcn_bilinear_cuda(data_im_ptr, W, H, W, him, wim);
    }
    
    int offset_col = B_idx *(C * k_w * k_h) * (H_out * W_out) + 
                    (C_idx * k_w * k_h + k_idx) * (H_out * W_out) + h_idx * W_out + w_idx;
    columns[offset_col] = __float2half(val * mask_val);
}

// 计算偏置梯度
__global__ void compute_bias_gradient(
    const float* grad_output,
    float* grad_bias,
    const int B, const int M, const int N
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;// M维度索引
    int bidx = blockIdx.y; // batch维度索引
    if (idx >= M || bidx >= B) return;
    float sum = 0.0f;
    for (int n = 0; n < N; ++n) {
        sum += grad_output[bidx * M * N + idx * N + n];
    }
    
    grad_bias[bidx * M + idx] = sum; // 每个batch的偏置梯度
    
}


__device__ void dcn_bilinear_coordinate_gradient_cuda(
                                           const float *bottom_data, const int data_width,
                                           const int height, const int width, 
                                           float h, float w, float grad_output,
                                           float grad_h, float grad_w)
{
    /*
    输入 2D input
    w, h, w, him, wim
    ∂L/∂val, grad_input-- 2D
    grad_h, grad_w
    */
    if (h <= -1 || h >= height || w <= -1 || w >= width)
    {
        //empty
        return;
    }
    int h_low = floor(h);
    int w_low = floor(w);
    int h_high = h_low + 1;
    int w_high = w_low + 1;

    float lh = h - h_low;
    float lw = w - w_low;
    float hh = 1 - lh, hw = 1 - lw;

    // 计算相对于h和w的梯度
    float v1 = 0.0f, v2 = 0.0f, v3 = 0.0f, v4 = 0.0f;
    if (h_low >= 0 && w_low >= 0)
        v1 = bottom_data[h_low * data_width + w_low];
    if (h_low >= 0 && w_high <= width - 1)
        v2 = bottom_data[h_low * data_width + w_high];
    if (h_high <= height - 1 && w_low >= 0)
        v3 = bottom_data[h_high * data_width + w_low];
    if (h_high <= height - 1 && w_high <= width - 1)
        v4 = bottom_data[h_high * data_width + w_high];

    grad_h = (v3 - v1) * hw + (v4 - v2) * lw;
    grad_w = (v2 - v1) * hh + (v4 - v3) * lh;

    grad_h *= grad_output;
    grad_w *= grad_output;
}


// 反向传播：计算对偏移，掩码的梯度
__global__ void im2colcuda_backward_offset_mask(
    const float *grad_output, 
    const float *input,
    const float *mask,
    const float *offset,
    float *grad_mask,
    float *grad_offset,
    const int H, const int W, const int k_h, const int k_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int B, const int C,
    const int H_out, const int W_out
)
{
    float val = 0.0f;
    // float offval = 0.0f;
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
    const float grad_output_val = grad_output[B_idx * (C * k_h * k_w) * (H_out * W_out) + global_y * (H_out * W_out) + global_x];
    const float mask_val = mask[B_idx * k_h * k_w * H_out * W_out + k_idx * H_out * W_out + h_idx * W_out + w_idx];
    
    const float offset_h = data_offset_ptr[0];
    const float offset_w = data_offset_ptr[H_out * W_out];
    float him = hin_idx + kh + offset_h ;
    float wim = win_idx + kw + offset_w ;

    if (him <= -1 || wim <= -1 || him >= H || wim >= W)
    {
        him = wim = -2;
    } else {
        val = dcn_bilinear_cuda(data_im_ptr, W, H, W, him, wim) * grad_output_val;
    }

    float grad_h = 0.0f;
    float grad_w = 0.0f;
    dcn_bilinear_coordinate_gradient_cuda(
        data_im_ptr, W, H, W, him, wim, 
        grad_output_val * mask_val, 
        grad_h, grad_w);

    __syncthreads();
    
    // 更新偏移的梯度
    atomicAdd(&grad_offset[B_idx * 2 * k_h * k_w * H_out * W_out + 2 * k_idx * H_out * W_out + h_idx * W_out + w_idx], grad_h);
    atomicAdd(&grad_offset[B_idx * 2 * k_h * k_w * H_out * W_out + (2 * k_idx + 1) * H_out * W_out + h_idx * W_out + w_idx], grad_w);

    // 计算掩码梯度
    atomicAdd(&grad_mask[B_idx * k_h * k_w * H_out * W_out + k_idx * H_out * W_out + h_idx * W_out + w_idx], 
              val);
}


__device__ float dmcn_get_gradient_weight_cuda(
    float argmax_h, float argmax_w,
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

// 反向传播：计算对输入的梯度
__global__ void im2colcuda_backward_input(
        const float *grad_output, 
        const float *mask,
        const float *offset,
        float *grad_input,
        const int H, const int W, const int k_h, const int k_w,
        const int pad_h, const int pad_w,
        const int stride_h, const int stride_w,
        const int B, const int C,
        const int H_out, const int W_out
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
    // float *grad_im_ptr = grad_input + (B_idx * C + C_idx) * H * W;
    const float *data_offset_ptr = offset + B_idx * 2 * k_h * k_w * H_out * W_out + 2 * k_idx * H_out * W_out + h_idx * W_out + w_idx;
    const float mask_val = mask[B_idx * k_h * k_w * H_out * W_out + k_idx * H_out * W_out + h_idx * W_out + w_idx];
    const float grad_output_val = grad_output[B_idx * (C * k_h * k_w) * (H_out * W_out) + global_y * (H_out * W_out) + global_x];
    
    const float offset_h = data_offset_ptr[0];
    const float offset_w = data_offset_ptr[H_out * W_out];
    float him = hin_idx + kh + offset_h ;
    float wim = win_idx + kw + offset_w ;

    const int cur_h = (int)him;
    const int cur_w = (int)wim;
    for (int dy = -2; dy <= 2; dy++)
    {
        for (int dx = -2; dx <= 2; dx++)
        {
            if (cur_h + dy >= 0 && cur_h + dy < H &&
                cur_w + dx >= 0 && cur_w + dx < W &&
                abs(him - (cur_h + dy)) < 1 &&
                abs(wim - (cur_w + dx)) < 1)
            {
                int cur_bottom_grad_pos = ((B_idx * C + C_idx) * H + cur_h + dy) * W + cur_w + dx;
                float weight = dmcn_get_gradient_weight_cuda(him, wim, cur_h + dy, cur_w + dx, H, W);
                atomicAdd(grad_input + cur_bottom_grad_pos, weight * grad_output_val * mask_val);
            }
        }
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

    int K = channels * kernel_w * kernel_h;
    int N = height_out * width_out;
    int M = channels_out;

    auto columns = at::empty({batch, K, N}, input.options().dtype(at::kHalf));

    auto grad_input = at::zeros_like(input);
    auto grad_weight = at::zeros({batch, M, K}, input.options());
    auto grad_bias = at::zeros({batch, M},bias.options());
    auto grad_offset = at::zeros_like(offset);
    auto grad_mask = at::zeros_like(mask);
    
    // for all gemm
    size_t smem = 4 * TILE * (TILE + 8) * sizeof(half);

    // 创建临时列梯度
    auto grad_columns = at::zeros({batch, K, N}, input.options());
    // grad_output -- b co ho wo

    // 获取指针
    half* columns_ptr = reinterpret_cast<half*>(columns.data_ptr<at::Half>());

    // compute column -- col.T = ∂output / ∂weight
    dim3 Grid0((N + TILE_X - 1) / TILE_X, (K + TILE_Y - 1) / TILE_Y, batch);
    dim3 Block0(TILE_X, TILE_Y);
    im2colcuda<<<Grid0, Block0, 0, stream>>>(
        input.contiguous().data_ptr<float>(), 
        mask.contiguous().data_ptr<float>(), 
        offset.contiguous().data_ptr<float>(), 
        height, width, kernel_h, kernel_w,
        pad_h, pad_w,
        stride_h, stride_w,
        batch, channels,
        height_out, width_out,
        columns_ptr
    );
    // compute  col.T
    auto columns_T = at::transpose(columns, 2, 1);
    half* columns_T_ptr = reinterpret_cast<half*>(columns_T.contiguous().data_ptr<at::Half>());
    // compute grad_weight -- ∂L/∂weight = ∂L/∂output @ ∂output/∂weight = grad_output @ columns_T
    dim3 Grid1((K + TILE - 1) / TILE, (M + TILE - 1) / TILE, batch);
    dim3 Block1(BLOCK_SIZE, 16);
    auto grad_output_ = grad_output.to(at::kHalf).contiguous();
    half* grad_output_ptr = reinterpret_cast<half*>(grad_output_.data_ptr<at::Half>());
    gemm_weight_grad_cuda<<<Grid1, Block1, smem, stream>>>(
        grad_output_ptr, 
        columns_T_ptr,
        grad_weight.contiguous().data_ptr<float>(),
        M, N, K
    );
    auto grad_weight_batch = grad_weight.sum(0).view({channels_out, channels, kernel_h, kernel_w}); // sum across batch

    // compute grad_columns = grad_output @ W_T  grad_output -- b M N.  W_T -- K M
    //  K M * b M N -> b K N
    auto weight_ptr = weight.view({channels_out, channels * kernel_h * kernel_w});
    weight_ptr = at::transpose(weight_ptr, 1, 0).to(at::kHalf);
    half* weight_ptr_ = reinterpret_cast<half*>(
        weight_ptr.contiguous().data_ptr<at::Half>());
    dim3 Grid2((N + TILE - 1) / TILE, (K + TILE - 1) / TILE, batch);
    gemm_column_grad_cuda<<<Grid2, Block1, smem, stream>>>(
        weight_ptr_,
        grad_output_ptr, 
        grad_columns.contiguous().data_ptr<float>(),
        K, M, N
    );

    // compute grad_bias = ∂L/∂bias = ∂L/∂output @ ∂output/∂bias = grad_output @ 1
    dim3 Grid3((M + DLA - 1) / DLA, batch);
    dim3 Block3(DLA, 1);
    compute_bias_gradient<<<Grid3, Block3, 0, stream>>>(
        grad_output.contiguous().data_ptr<float>(), // b M N
        grad_bias.contiguous().data_ptr<float>(),// b M
        batch, M, N
    );
    auto grad_bias_batch = grad_bias.sum(0); // sum across batch

    // compute grad_offset and grad_mask 沿着广播C维度方向叠加梯度 利用原子加法操作
    im2colcuda_backward_offset_mask<<<Grid0, Block0, 0, stream>>>(
        grad_columns.contiguous().data_ptr<float>(), // b K N
        input.contiguous().data_ptr<float>(), // b c h w
        mask.contiguous().data_ptr<float>(), // b k_h k_w h_out w_out
        offset.contiguous().data_ptr<float>(),// b 2 k_h k_w h_out w_out
        grad_mask.contiguous().data_ptr<float>(), // b k_h k_w h_out w_out
        grad_offset.contiguous().data_ptr<float>(),
        height, width, kernel_h, kernel_w,
        pad_h, pad_w,
        stride_h, stride_w,
        batch, channels,
        height_out, width_out
    );

    // //compute grad_input  -- b 2 k_h k_w h_out w_out -- 仍然沿着广播C维度方向叠加梯度 利用原子加法操作
    im2colcuda_backward_input<<<Grid0, Block0, 0, stream>>>(
        grad_columns.contiguous().data_ptr<float>(), // b K N
        mask.contiguous().data_ptr<float>(), // b k_h k_w h_out w_out
        offset.contiguous().data_ptr<float>(),// b 2 k_h k_w h_out w_out
        grad_input.contiguous().data_ptr<float>(),
        height, width, kernel_h, kernel_w,
        pad_h, pad_w,
        stride_h, stride_w,
        batch, channels,
        height_out, width_out
    ); 
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Launch Error: " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return {
        grad_input, grad_offset, grad_mask, grad_weight_batch, grad_bias_batch
    };
}
