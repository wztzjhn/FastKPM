#include <cuComplex.h>

namespace fkpm {
    __global__ void outer_product_kernel(int n_rows, int n_cols, float alpha, cuFloatComplex *A, cuFloatComplex *B,
                                         int D_nnz, int *D_row_idx, int *D_col_idx, cuFloatComplex *D_val) {
        int idx = blockIdx.x*blockDim.x + threadIdx.x;
        while (idx < D_nnz) {
            int i = D_row_idx[idx];
            int j = D_col_idx[idx];
            double acc_re = 0;
            double acc_im = 0;
            for (int k = 0; k < n_cols; k++) {
                cuFloatComplex a = A[k*n_rows+i];
                cuFloatComplex b = B[k*n_rows+j];
                // acc += A_ik * conj(B_jk)
                acc_re += a.x * b.x + a.y * b.y;
                acc_im += a.y * b.x - a.x * b.y;
            }
            // D_ij += alpha * acc
            D_val[idx] = cuCaddf(D_val[idx], make_cuFloatComplex(alpha*acc_re, alpha*acc_im));
            idx += gridDim.x*blockDim.x;
        }
    }
    
    // D_ij += alpha \sum_k A_ik conj(B_jk)
    void outer_product(int n_rows, int n_cols, float alpha, cuFloatComplex *A, cuFloatComplex *B,
                       int D_nnz, int *D_row_idx, int *D_col_idx, cuFloatComplex *D_val) {
        int block_size = 64;
        int grid_size = min(max(D_nnz / block_size, 1), 256);
        outer_product_kernel<<<grid_size, block_size>>>(n_rows, n_cols, alpha, A, B, D_nnz, D_row_idx, D_col_idx, D_val);
    }
}
