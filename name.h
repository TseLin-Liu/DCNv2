#ifndef DCN_V2_IM2COL_CUDA
#define DCN_V2_IM2COL_CUDA
#include <torch/extension.h>
#include <cstdio>
#include <tuple>
#include <string>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#define BLOCK_SIZE 32
#define TILE_Y 16
#define TILE_X 16
#define TILE 64


at::Tensor MyDCN_forward(
        const at::Tensor& input,
        const at::Tensor& weight,
        const at::Tensor& mask,
        const at::Tensor& offset,
        const at::Tensor& bias,
        const int stride_h,
        const int stride_w,
        const int pad_h,
        const int pad_w);

std::vector<at::Tensor> MyDCN_backward(
        const at::Tensor& input,
        const at::Tensor& weight,
        const at::Tensor& mask,
        const at::Tensor& offset,
        const at::Tensor& bias,
        const at::Tensor &grad_output,
        const int stride_h,
        const int stride_w,
        const int pad_h,
        const int pad_w);

#endif
