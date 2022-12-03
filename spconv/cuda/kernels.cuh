#pragma once

#include <torch/extension.h>

template <typename value_t, typename idx_t>
__global__ void sp_conv2d_kernel(
    const value_t *input,
    const value_t *weight_values,
    const idx_t *weight_col_idx,
    const idx_t *weight_row_nnz,
    idx_t weight_max_row_nnz,
    const value_t *bias,
    idx_t n,
    idx_t c,
    idx_t h,
    idx_t w,
    idx_t r,
    idx_t s,
    idx_t c_out,
    idx_t h_out,
    idx_t w_out,
    idx_t stride_x,
    idx_t stride_y,
    idx_t dilation_x,
    idx_t dilation_y,
    value_t *output
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= c_out) {
        return;
    }

    // extern __shared__ value_t input_buf[];

    value_t bias_val = bias[tid];

    idx_t filter_range_y = (r - 1) * dilation_y + 1;
    idx_t filter_range_x = (s - 1) * dilation_x + 1;

    idx_t filter_above = filter_range_y / 2;
    idx_t filter_below = (filter_range_y - 1) / 2;
    idx_t filter_left = filter_range_x / 2;
    idx_t filter_right = (filter_range_x - 1) / 2;

    // Loop over batch
    //   Loop over spatial locations
    //      Loop over NZs of thread's filter and sum

    for (idx_t n_i = 0; n_i < n; n_i++) {
        idx_t filter_loc_flat = 0;  // Store results in successive locations of the output, starting from 0

        for (idx_t i = filter_above; i + filter_below < h; i += stride_y) {
            for (idx_t j = filter_left; j + filter_right < w; j += stride_x) {
                value_t dot_prod = bias_val;

                for (idx_t nz_i = 0; nz_i < weight_row_nnz[tid]; nz_i++) {
                    idx_t k = weight_col_idx[(tid * weight_max_row_nnz) + (nz_i)];
                    idx_t c_i = k / (r * s);  // Channel number
                    idx_t r_i = (k - c_i * r * s) / r;
                    idx_t s_i = k % s;

                    // Image indices that current filter NZ position lines up with
                    idx_t img_loc_i = i - filter_above + r_i * dilation_y;
                    idx_t img_loc_j = j - filter_left + s_i * dilation_x;

                    value_t inp_val = input[(n_i * c * h * w) + (c_i * h * w) + (img_loc_i * w) + (img_loc_j)];
                    dot_prod += weight_values[(tid * weight_max_row_nnz) + (nz_i)] * inp_val;
                }

                output[(tid * n * h_out * w_out) + (n_i * h_out * w_out) + filter_loc_flat] = dot_prod;
                filter_loc_flat++;
            }
        }
    }
}

template <typename value_t, typename idx_t>
__global__ void sp_conv2d_kernel_v2(
    const value_t *input,
    const value_t *weight_values,
    const idx_t *weight_col_idx,
    const idx_t *weight_row_nnz,
    idx_t weight_max_row_nnz,
    const value_t *bias,
    idx_t n,
    idx_t c,
    idx_t h,
    idx_t w,
    idx_t r,
    idx_t s,
    idx_t c_out,
    idx_t h_out,
    idx_t w_out,
    idx_t stride_x,
    idx_t stride_y,
    idx_t dilation_x,
    idx_t dilation_y,
    value_t *output
) {
    int out_row = blockIdx.y;
    int out_col = blockIdx.x * blockDim.x + threadIdx.x;

    idx_t out_cols = n * h_out * w_out;

    if (out_row >= c_out || out_col >= out_cols) {
        return;
    }

    value_t bias_val = bias[out_row];

    idx_t filter_range_y = (r - 1) * dilation_y + 1;
    idx_t filter_range_x = (s - 1) * dilation_x + 1;

    idx_t filter_above = filter_range_y / 2;
    idx_t filter_below = (filter_range_y - 1) / 2;
    idx_t filter_left = filter_range_x / 2;
    idx_t filter_right = (filter_range_x - 1) / 2;

    // Loop over batch
    //     Loop over NZs of thread's output location and sum

    for (idx_t n_i = 0; n_i < n; n_i++) {  // Grid stride loop over batch dimension
        idx_t batch_invar_col = out_col % (h_out * w_out);  // Want the column number irrespective of the batch
        idx_t out_i = batch_invar_col / w_out;
        idx_t out_j = batch_invar_col % w_out;
        idx_t i = out_i * stride_y + filter_above;
        idx_t j = out_j * stride_x + filter_left;

        value_t dot_prod = bias_val;

        for (idx_t nz_i = 0; nz_i < weight_row_nnz[out_row]; nz_i++) {
            idx_t nz_idx = (out_row * weight_max_row_nnz) + (nz_i);
            idx_t k = weight_col_idx[nz_idx];
            idx_t c_i = k / (r * s);  // Channel number
            idx_t r_i = (k - c_i * r * s) / r;
            idx_t s_i = k % s;

            // Image indices that current filter NZ position lines up with
            // Indirect im2col indexing logic
            idx_t img_loc_i = i - filter_above + r_i * dilation_y;
            idx_t img_loc_j = j - filter_left + s_i * dilation_x;

            value_t inp_val = input[(n_i * c * h * w) + (c_i * h * w) + (img_loc_i * w) + (img_loc_j)];
            dot_prod += weight_values[nz_idx] * inp_val;
        }

        output[(out_row * n * h_out * w_out) + (n_i * h_out * w_out) + out_col] = dot_prod;
    }
}

template <typename value_t, typename idx_t>
__global__ void sp_conv2d_kernel_v3(
    const torch::PackedTensorAccessor32<value_t, 4, torch::RestrictPtrTraits> input,
    const torch::PackedTensorAccessor32<value_t, 2, torch::RestrictPtrTraits> weight_values,
    const torch::PackedTensorAccessor32<idx_t, 2, torch::RestrictPtrTraits> weight_col_idx,
    const idx_t *weight_row_nnz,
    idx_t weight_max_row_nnz,
    const value_t *bias,
    idx_t n,
    idx_t c,
    idx_t h,
    idx_t w,
    idx_t r,
    idx_t s,
    idx_t c_out,
    idx_t h_out,
    idx_t w_out,
    idx_t stride_x,
    idx_t stride_y,
    idx_t dilation_x,
    idx_t dilation_y,
    torch::PackedTensorAccessor32<value_t, 2, torch::RestrictPtrTraits> output
) {
    int out_row = blockIdx.y;
    int out_flat_sp_loc = blockIdx.x * blockDim.x + threadIdx.x;

    idx_t out_cols = n * h_out * w_out;

    if (out_row >= c_out) {
        return;
    }

    value_t bias_val = bias[out_row];

    idx_t filter_range_y = (r - 1) * dilation_y + 1;
    idx_t filter_range_x = (s - 1) * dilation_x + 1;

    idx_t filter_above = filter_range_y / 2;
    idx_t filter_below = (filter_range_y - 1) / 2;
    idx_t filter_left = filter_range_x / 2;
    idx_t filter_right = (filter_range_x - 1) / 2;

    // Loop over batch
    //     Loop over NZs of thread's output location and sum

    for (idx_t n_i = 0; n_i < n; n_i++) {  // Grid stride loop over batch dimension
        idx_t out_col = (n_i * h_out * w_out) + out_flat_sp_loc;

        if (out_col >= out_cols) {
            return;
        }

        idx_t out_i = out_flat_sp_loc / w_out;
        idx_t out_j = out_flat_sp_loc % w_out;
        idx_t i = out_i * stride_y;
        idx_t j = out_j * stride_x;

        value_t dot_prod = bias_val;

        for (idx_t nz_i = 0; nz_i < weight_row_nnz[out_row]; nz_i++) {
            // Get corresponding value from
            idx_t k = weight_col_idx[out_row][nz_i];

            // Index location within filter
            idx_t c_i = k / (r * s);
            idx_t r_i = (k - c_i * r * s) / r;
            idx_t s_i = k % s;

            // Image indices that current filter NZ position lines up with
            // Indirect im2col indexing logic
            idx_t img_loc_i = i + r_i * dilation_y;
            idx_t img_loc_j = j + s_i * dilation_x;

            value_t inp_val = input[n_i][c_i][img_loc_i][img_loc_j];
            dot_prod += weight_values[out_row][nz_i] * inp_val;
        }

        output[out_row][out_col] = dot_prod;
    }
}

template <typename value_t, typename idx_t>
__global__ void sp_conv2d_kernel_v4(
    const torch::PackedTensorAccessor32<value_t, 4, torch::RestrictPtrTraits> input,
    const torch::PackedTensorAccessor32<value_t, 2, torch::RestrictPtrTraits> weight_values,
    const torch::PackedTensorAccessor32<idx_t, 2, torch::RestrictPtrTraits> weight_col_idx,
    const idx_t *weight_row_nnz,
    idx_t weight_max_row_nnz,
    const value_t *bias,
    idx_t n,
    idx_t c,
    idx_t h,
    idx_t w,
    idx_t r,
    idx_t s,
    idx_t c_out,
    idx_t h_out,
    idx_t w_out,
    idx_t stride_x,
    idx_t stride_y,
    idx_t dilation_x,
    idx_t dilation_y,
    torch::PackedTensorAccessor32<value_t, 2, torch::RestrictPtrTraits> output
) {
    int out_row = blockIdx.y;
    int out_flat_sp_loc = blockIdx.x * blockDim.x + threadIdx.x;

    idx_t out_cols = n * h_out * w_out;

    if (out_row >= c_out) {
        return;
    }

    idx_t row_nnz = weight_row_nnz[out_row];

    __shared__ void *shared_mem;
    value_t *weight_values_local = (value_t*)shared_mem;
    idx_t *weight_col_idx_local = (idx_t*)&weight_values_local[weight_max_row_nnz];

    int tid = threadIdx.x;  // Rename for readability (compiler should optimize this out)
    int threads = blockDim.x;

    // Thread block stride over nonzeros
    for (idx_t chunk = 0; chunk < (row_nnz + threads - 1) / threads; chunk++) {
        int i = chunk * threads + tid;

        if (i < row_nnz) {
            weight_values_local[i] = weight_values[i];
            weight_col_idx_local[i] = weight_col_idx[i];
        }
    }

    __syncthreads();  // Syncthreads here is more readable but it can be moved down

    value_t bias_val = bias[out_row];

    idx_t filter_range_y = (r - 1) * dilation_y + 1;
    idx_t filter_range_x = (s - 1) * dilation_x + 1;

    idx_t filter_above = filter_range_y / 2;
    idx_t filter_below = (filter_range_y - 1) / 2;
    idx_t filter_left = filter_range_x / 2;
    idx_t filter_right = (filter_range_x - 1) / 2;

    // Loop over batch
    //     Loop over NZs of thread's output location and sum

    for (idx_t n_i = 0; n_i < n; n_i++) {  // Grid stride loop over batch dimension
        idx_t out_col = (n_i * h_out * w_out) + out_flat_sp_loc;

        if (out_col >= out_cols) {
            return;
        }

        idx_t out_i = out_flat_sp_loc / w_out;
        idx_t out_j = out_flat_sp_loc % w_out;
        idx_t i = out_i * stride_y;
        idx_t j = out_j * stride_x;

        value_t dot_prod = bias_val;

        for (idx_t nz_i = 0; nz_i < row_nnz; nz_i++) {
            // Get corresponding value from
            idx_t k = weight_col_idx_local[out_row][nz_i];

            // Index location within filter
            idx_t c_i = k / (r * s);
            idx_t r_i = (k - c_i * r * s) / r;
            idx_t s_i = k % s;

            // Image indices that current filter NZ position lines up with
            // Indirect im2col indexing logic
            idx_t img_loc_i = i + r_i * dilation_y;
            idx_t img_loc_j = j + s_i * dilation_x;

            value_t inp_val = input[n_i][c_i][img_loc_i][img_loc_j];
            dot_prod += weight_values_local[out_row][nz_i] * inp_val;
        }

        output[out_row][out_col] = dot_prod;
    }
}
