#include <exception>
#include <optional>

#include <iostream>

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

#include "kernels.cuh"

#define CHECK_CUDA_ERROR(func)                                                 \
  {                                                                            \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
      printf("CUDA API failed at line %d with error: %s (%d)\n", __LINE__,     \
             cudaGetErrorString(status), status);                              \
      throw std::exception();                                                  \
    }                                                                          \
  }

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

using idx_t = long;  // int

torch::Tensor spconv2d(
    const torch::Tensor &input,
    const torch::Tensor &weight_values,
    const torch::Tensor &weight_col_idx,
    const torch::Tensor &weight_row_nnz,
    int f,
    int c_in_over_groups,
    int r,
    int s,
    const torch::Tensor &bias,
    int stride_x,
    int stride_y,
    int dilation_x,
    int dilation_y
) {
    CHECK_INPUT(input);
    CHECK_INPUT(weight_values);
    CHECK_INPUT(weight_col_idx);
    CHECK_INPUT(weight_row_nnz);
    CHECK_INPUT(bias);

    // Keep constant for now, but support these as arguments later
    int groups = 1;

    idx_t n;  // Batch size
    idx_t c;  // Channels
    idx_t h;  // Image height
    idx_t w;  // Image width
    /*
    p -> vertical spatial loc
    q -> horizontal spatial loc
    */

    if (input.dim() == 3) {
        n = 1;
        c = input.sizes()[0];
        h = input.sizes()[1];
        w = input.sizes()[2];
    } else if (input.dim() == 4) {
        n = input.sizes()[0];
        c = input.sizes()[1];
        h = input.sizes()[2];
        w = input.sizes()[3];
    } else {
        throw torch::ValueError("Input must have 3 or 4 dimensions");
    }

    int c_out = f * groups;
    int h_out = ((h - dilation_y * (r - 1) - 1) / stride_y) + 1;
    int w_out = ((w - dilation_x * (s - 1) - 1) / stride_x) + 1;
    torch::Tensor output = torch::empty({n, c_out, h_out, w_out}, torch::TensorOptions().device(torch::kCUDA, 0));
    // output = at::expand_copy(bias.view({1, -1, 1, 1}), {n, c_out, h_out, w_out});

    // Parallelize along output channels
    constexpr int block_size = 64;
    int grid_size = (c_out + block_size - 1) / block_size;

    // Leftover saved
    // auto grid_dim = dim(
    //     (w_out + block_size - 1) / block_size,
    //     (h_out + block_size - 1) / block_size
    // );

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES(input.type(), "spconv2d", ([&] {
        sp_conv2d_kernel<scalar_t, idx_t><<<grid_size, block_size, 0, stream>>>(
            input.data_ptr<scalar_t>(),
            weight_values.data_ptr<scalar_t>(),
            weight_col_idx.data_ptr<idx_t>(),
            weight_row_nnz.data_ptr<idx_t>(),
            weight_values.sizes()[1],
            bias.data_ptr<scalar_t>(),
            n,
            c,
            h,
            w,
            r,
            s,
            c_out,
            h_out,
            w_out,
            stride_x,
            stride_y,
            dilation_x,
            dilation_y,
            output.data_ptr<scalar_t>()
        );
    }));

    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    return output.view({c_out, n, h_out, w_out}).transpose(0, 1);
}

torch::Tensor spconv2d_v2(
    const torch::Tensor &input,
    const torch::Tensor &weight_values,
    const torch::Tensor &weight_col_idx,
    const torch::Tensor &weight_row_nnz,
    int f,
    int c_in_over_groups,
    int r,
    int s,
    const torch::Tensor &bias,
    int stride_x,
    int stride_y,
    int dilation_x,
    int dilation_y
) {
    CHECK_INPUT(input);
    CHECK_INPUT(weight_values);
    CHECK_INPUT(weight_col_idx);
    CHECK_INPUT(weight_row_nnz);
    CHECK_INPUT(bias);

    // Keep constant for now, but support these as arguments later
    int groups = 1;

    idx_t n;  // Batch size
    idx_t c;  // Channels
    idx_t h;  // Image height
    idx_t w;  // Image width
    /*
    p -> vertical spatial loc
    q -> horizontal spatial loc
    */

    if (input.dim() == 3) {
        n = 1;
        c = input.sizes()[0];
        h = input.sizes()[1];
        w = input.sizes()[2];
    } else if (input.dim() == 4) {
        n = input.sizes()[0];
        c = input.sizes()[1];
        h = input.sizes()[2];
        w = input.sizes()[3];
    } else {
        throw torch::ValueError("Input must have 3 or 4 dimensions");
    }

    int c_out = f * groups;
    int h_out = ((h - dilation_y * (r - 1) - 1) / stride_y) + 1;
    int w_out = ((w - dilation_x * (s - 1) - 1) / stride_x) + 1;
    torch::Tensor output = torch::empty({c_out, n, h_out, w_out}, torch::TensorOptions().device(torch::kCUDA, 0));
    // output = at::expand_copy(bias.view({1, -1, 1, 1}), {n, c_out, h_out, w_out});

    // Parallelize along output channels
    constexpr int block_size = 128;
    // int grid_size = (c_out + block_size - 1) / block_size;
    idx_t out_ncols = n * h_out * w_out;
    auto grid_size = dim3((out_ncols + block_size - 1) / block_size, c_out);

    // Leftover saved
    // auto grid_dim = dim(
    //     (w_out + block_size - 1) / block_size,
    //     (h_out + block_size - 1) / block_size
    // );

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES(input.type(), "spconv2d_v2", ([&] {
        sp_conv2d_kernel_v2<scalar_t, idx_t><<<grid_size, block_size, 0, stream>>>(
            input.data_ptr<scalar_t>(),
            weight_values.data_ptr<scalar_t>(),
            weight_col_idx.data_ptr<idx_t>(),
            weight_row_nnz.data_ptr<idx_t>(),
            weight_values.sizes()[1],
            bias.data_ptr<scalar_t>(),
            n,
            c,
            h,
            w,
            r,
            s,
            c_out,
            h_out,
            w_out,
            stride_x,
            stride_y,
            dilation_x,
            dilation_y,
            output.data_ptr<scalar_t>()
        );
    }));

    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    return output.view({c_out, n, h_out, w_out}).transpose(0, 1);
}

torch::Tensor spconv2d_v3(
    const torch::Tensor &input,
    const torch::Tensor &weight_values,
    const torch::Tensor &weight_col_idx,
    const torch::Tensor &weight_row_nnz,
    int f,
    int c_in_over_groups,
    int r,
    int s,
    const torch::Tensor &bias,
    int stride_x,
    int stride_y,
    int dilation_x,
    int dilation_y
) {
    CHECK_INPUT(input);
    CHECK_INPUT(weight_values);
    CHECK_INPUT(weight_col_idx);
    CHECK_INPUT(weight_row_nnz);
    CHECK_INPUT(bias);

    // Keep constant for now, but support these as arguments later
    int groups = 1;

    idx_t n;  // Batch size
    idx_t c;  // Channels
    idx_t h;  // Image height
    idx_t w;  // Image width
    /*
    p -> vertical spatial loc
    q -> horizontal spatial loc
    */

    if (input.dim() == 3) {
        n = 1;
        c = input.sizes()[0];
        h = input.sizes()[1];
        w = input.sizes()[2];
    } else if (input.dim() == 4) {
        n = input.sizes()[0];
        c = input.sizes()[1];
        h = input.sizes()[2];
        w = input.sizes()[3];
    } else {
        throw torch::ValueError("Input must have 3 or 4 dimensions");
    }

    idx_t c_out = f * groups;
    idx_t h_out = ((h - dilation_y * (r - 1) - 1) / stride_y) + 1;
    idx_t w_out = ((w - dilation_x * (s - 1) - 1) / stride_x) + 1;
    idx_t out_ncols = n * h_out * w_out;
    torch::Tensor output = torch::empty({c_out, out_ncols}, torch::TensorOptions().device(torch::kCUDA, 0));

    // Parallelize along output channels
    constexpr int block_size = 128;
    auto grid_size = dim3((h_out * w_out + block_size - 1) / block_size, c_out);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES(input.type(), "spconv2d_v3", ([&] {
        sp_conv2d_kernel_v3<scalar_t, idx_t><<<grid_size, block_size, 0, stream>>>(
            input.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            weight_values.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            weight_col_idx.packed_accessor32<idx_t, 2, torch::RestrictPtrTraits>(),
            weight_row_nnz.data_ptr<idx_t>(),
            weight_values.sizes()[1],
            bias.data_ptr<scalar_t>(),
            n,
            c,
            h,
            w,
            r,
            s,
            c_out,
            h_out,
            w_out,
            stride_x,
            stride_y,
            dilation_x,
            dilation_y,
            output.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>()
        );
    }));

    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    return output.view({c_out, n, h_out, w_out}).transpose(0, 1);
}

torch::Tensor spconv2d_v4(
    const torch::Tensor &input,
    const torch::Tensor &weight_values,
    const torch::Tensor &weight_col_idx,
    const torch::Tensor &weight_row_nnz,
    int f,
    int c_in_over_groups,
    int r,
    int s,
    const torch::Tensor &bias,
    int stride_x,
    int stride_y,
    int dilation_x,
    int dilation_y
) {
    CHECK_INPUT(input);
    CHECK_INPUT(weight_values);
    CHECK_INPUT(weight_col_idx);
    CHECK_INPUT(weight_row_nnz);
    CHECK_INPUT(bias);

    // Keep constant for now, but support these as arguments later
    int groups = 1;

    idx_t n;  // Batch size
    idx_t c;  // Channels
    idx_t h;  // Image height
    idx_t w;  // Image width
    /*
    p -> vertical spatial loc
    q -> horizontal spatial loc
    */

    if (input.dim() == 3) {
        n = 1;
        c = input.sizes()[0];
        h = input.sizes()[1];
        w = input.sizes()[2];
    } else if (input.dim() == 4) {
        n = input.sizes()[0];
        c = input.sizes()[1];
        h = input.sizes()[2];
        w = input.sizes()[3];
    } else {
        throw torch::ValueError("Input must have 3 or 4 dimensions");
    }

    idx_t c_out = f * groups;
    idx_t h_out = ((h - dilation_y * (r - 1) - 1) / stride_y) + 1;
    idx_t w_out = ((w - dilation_x * (s - 1) - 1) / stride_x) + 1;
    idx_t out_ncols = n * h_out * w_out;
    torch::Tensor output = torch::empty({c_out, out_ncols}, torch::TensorOptions().device(torch::kCUDA, 0));

    // Parallelize along output channels
    constexpr int block_size = 128;
    auto grid_size = dim3((h_out * w_out + block_size - 1) / block_size, c_out);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES(input.type(), "spconv2d_v4", ([&] {
        size_t shmem_size = 2 * weight_values.sizes()[1] * sizeof(scalar_t);  // Shared memory space for sparse matrix row
        sp_conv2d_kernel_v3<scalar_t, idx_t><<<grid_size, block_size, shmem_size, stream>>>(
            input.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            weight_values.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            weight_col_idx.packed_accessor32<idx_t, 2, torch::RestrictPtrTraits>(),
            weight_row_nnz.data_ptr<idx_t>(),
            weight_values.sizes()[1],
            bias.data_ptr<scalar_t>(),
            n,
            c,
            h,
            w,
            r,
            s,
            c_out,
            h_out,
            w_out,
            stride_x,
            stride_y,
            dilation_x,
            dilation_y,
            output.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>()
        );
    }));

    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    return output.view({c_out, n, h_out, w_out}).transpose(0, 1);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "spconv2d",
        &spconv2d,
        "spconv2d"
    );
    m.def(
        "spconv2d_v2",
        &spconv2d_v2,
        "spconv2d_v2"
    );
    m.def(
        "spconv2d_v3",
        &spconv2d_v3,
        "spconv2d_v3"
    );
    m.def(
        "spconv2d_v4",
        &spconv2d_v4,
        "spconv2d_v4"
    );
}
