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

__device__ void cp_async(const half *A, int globalx_idx, int globaly_idx, int W,
    int2 wrap_group_idx, int write_state, half* MA) {
    int tx = threadIdx.x;
    void *ptr = 
        (void *)(MA + write_state * TILE * (TILE + 8) + (wrap_group_idx.y * 16 + tx / 2) * (TILE + 8) + 
                wrap_group_idx.x * 16 + (tx % 2) * 8);
    uint32_t smem_ptr;

    asm(
        "{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 %0, smem_ptr; }\n"
        : "=r"(smem_ptr)
        : "l"(ptr));

    asm volatile("cp.async.cg.shared.global [%0], [%1], %2;\n" ::"r"(smem_ptr),
                    "l"(&A[globaly_idx * W + globalx_idx]),
                    "n"(16));
}

__global__ void gemm_cuda(
    const half* weight, const half* columns,
    float* output, const float* bias, const int M, const int K, const int N
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
    float* out_im_ptr = output + b_idx * M * N;//  C_out ,  * H_out * W_out    
    const half* col_ptr = columns + b_idx * K * N; //  C * k_w * k_h ,  * H_out * W_out 

    extern __shared__ uint8_t smem [];
    half* MA = reinterpret_cast<half*>(smem);
    half* MB = reinterpret_cast<half*>(smem + 2 * TILE * (TILE + 8) * sizeof(half));

    int round = (K + TILE - 1) / TILE;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    int read_state = 0;
    int write_state = 0;

    {
        int mx = wrap_group_idx.x * 16 + (tx % 2) * 8 + 0 * TILE; // 8 vector
        int C_out_idx = tile_grid.y +  wrap_group_idx.y * 16 + tx / 2;
        cp_async(weight, mx, C_out_idx, K, wrap_group_idx, write_state, MA);
        int my = wrap_group_idx.y * 16 + tx / 2 + 0 * TILE;
        int HW_out_idx = tile_grid.x + wrap_group_idx.x * 16 + (tx % 2) * 8 ;
        cp_async(col_ptr, HW_out_idx, my, N, wrap_group_idx, write_state, MB);
        asm volatile("cp.async.commit_group;\n");
    }
    
    for (int k=0; k < round; ++k){
        asm volatile("cp.async.wait_group %0;\n"::"n"(0));
        __syncthreads();
        read_state = k % 2;
        write_state = (k + 1) % 2;
        // 将全局内存数据读入到共享内存中 uint2 mma_xy -- 4*8 4行坐标上连续 8列坐标上不连续 -- 为了避免 bank conflict
        if ( k + 1 < round) {
            int next_k = k + 1;
            int mx = wrap_group_idx.x * 16 + (tx % 2) * 8 + next_k * TILE;
            int C_out_idx = tile_grid.y +  wrap_group_idx.y * 16 + tx / 2;
            cp_async(weight, mx, C_out_idx, K, wrap_group_idx, write_state, MA);
            int my = wrap_group_idx.y * 16 + tx / 2 + next_k * TILE;
            int HW_out_idx = tile_grid.x + wrap_group_idx.x * 16 + (tx % 2) * 8 ;
            cp_async(col_ptr, HW_out_idx, my, N, wrap_group_idx, write_state, MB);
            asm volatile("cp.async.commit_group;\n");
        }
        
        for (int km = 0; km < (TILE / 16); ++km) {
            wmma::load_matrix_sync(a_frag, MA + read_state * TILE * (TILE + 8) + wrap_group_idx.y * 16 * (TILE + 8) + km * 16, TILE + 8);
            wmma::load_matrix_sync(b_frag, MB + read_state * TILE * (TILE + 8) + km * 16 * (TILE + 8) + wrap_group_idx.x * 16, TILE + 8);

            // 6. 张量核乘加
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
    }
    int C_out_idx = tile_grid.y +  wrap_group_idx.y * 16;
    int HW_out_idx = tile_grid.x + wrap_group_idx.x * 16;
    wmma::store_matrix_sync(
            out_im_ptr+C_out_idx * N + HW_out_idx, 
            c_frag, N, wmma::mem_row_major);

    #pragma unroll
    for (int wrap_id = 0; wrap_id < 8; ++wrap_id) {
        int y = C_out_idx + wrap_id * 2 + tx/16;
        int x = HW_out_idx + tx % 16;
        if (x < N && y < M) {
            out_im_ptr[y * N + x] += bias[y];
        }
    }
}



__device__ float dcn_bilinear_cuda_forward(const float *bottom_data, const int data_width,
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


__global__ void im2colcuda_forward(
        const float *input, 
        const float *mask,
        const float *offset,
        const int H, const int W, const int k_h, const int k_w,
        const int pad_h, const int pad_w,
        const int stride_h, const int stride_w,
        const int B, const int C,
        const int H_out, const int W_out,
        const int col_ypad, const int col_xpad,
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
    const float him = hin_idx + kh + offset_h ;
    const float wim = win_idx + kw + offset_w ;

    bool cts = (him > -1 && wim > -1 && him < H && wim < W);
    float val = 0.0f;
    if (cts) {
        val = dcn_bilinear_cuda_forward(data_im_ptr, W, H, W, him, wim);
    }
    
    int offset_col = B_idx *(C * k_w * k_h + col_ypad) * (H_out * W_out +col_xpad) + 
                    (C_idx * k_w * k_h + k_idx) * (H_out * W_out +col_xpad) + h_idx * W_out + w_idx;
    columns[offset_col] = __float2half(val * mask_val);
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
    const int k_h = weight.size(2);
    const int k_w = weight.size(3);

    const int H_out = (H - k_h + 2 * pad_h) / stride_h + 1;
    const int W_out = (W - k_w + 2 * pad_w) / stride_w + 1;

    int K = C * k_w * k_h;
    int N = H_out * W_out;
    int M = C_out;

    dim3 Grid0((N + TILE_X -1)/ TILE_X, (K + TILE_Y -1)/ TILE_Y, B); 
    dim3 Block0(TILE_X, TILE_Y);

    // padding weight and column
    int M_padded = (M + TILE - 1) / TILE * TILE;
    int K_padded = (K + TILE - 1) / TILE * TILE;
    int N_padded = (N + TILE - 1) / TILE * TILE;
    
    // padding weight
    auto weight_pad = weight.view({M, K});
    weight_pad = torch::nn::functional::pad(
      weight_pad, torch::nn::functional::PadFuncOptions(
        {0, (TILE - K % TILE) % TILE, 0, (TILE - M % TILE) % TILE})).contiguous();

    // padding column
    auto columns = at::empty({B, K_padded, N_padded}, input.options());
    int col_ypad = (TILE - K % TILE) % TILE;
    int col_xpad = (TILE - N % TILE) % TILE;
    AT_ASSERTM((col_ypad == (K_padded - K) && col_xpad == (N_padded - N)), "column pad size would be error !");

    // pad output 
    auto output = torch::empty({B, M_padded, N_padded}, input.options().dtype(torch::kFloat32)).contiguous();

    dim3 Grid1((N_padded + TILE -1)/ TILE, (M_padded + TILE -1)/ TILE, B); 
    dim3 Block1(BLOCK_SIZE, 16);

    if (weight_pad.dtype() != torch::kHalf) {
        weight_pad = weight_pad.to(torch::kHalf);
    }
    if (columns.dtype() != torch::kHalf) {
        columns = columns.to(torch::kHalf);
    }

    half* weight_ptr = reinterpret_cast<half*>(weight_pad.data_ptr<at::Half>());
    half* columns_ptr = reinterpret_cast<half*>(columns.data_ptr<at::Half>());
    
    size_t smem = 4 * TILE * (TILE + 8) * sizeof(half);
 
    im2colcuda_forward<<<Grid0, Block0, 0, stream>>>(
        input.contiguous().data_ptr<float>(), 
        mask.contiguous().data_ptr<float>(), 
        offset.contiguous().data_ptr<float>(), 
        H, W, k_h, k_w,
        pad_h, pad_w,
        stride_h, stride_w,
        B, C,
        H_out, W_out,
        col_ypad, col_xpad,
        columns_ptr
    );
    // 进行GEMM 标准矩阵乘法
    gemm_cuda<<<Grid1, Block1, smem, stream>>> (
        weight_ptr,// Cout C * k_h * k_w
        columns_ptr,//B  C * k_h * k_w H_out * W_out
        output.data_ptr<float>(),
        bias.contiguous().data_ptr<float>(),
        M_padded,
        K_padded,
        N_padded
    );
  
    output = output.slice(1, 0, M).slice(2, 0, N).view({B, M, H_out, W_out});
    return output;
}
