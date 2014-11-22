#include <cuComplex.h>

namespace fkpm {
    
    // --- Real --------------------------------------------------------------
    
    template <typename T>
    __global__ void outer_product_kernel_re(int n_rows, int n_cols, T alpha, T *A, T *B,
                                            int D_nnz, int *D_row_idx, int *D_col_idx, T *D_val) {
        int idx = blockIdx.x*blockDim.x + threadIdx.x;
        while (idx < D_nnz) {
            int i = D_row_idx[idx];
            int j = D_col_idx[idx];
            T acc = 0;
            for (int k = 0; k < n_cols; k++) {
                T a = A[k*n_rows+i];
                T b = B[k*n_rows+j];
                acc += a * b;
            }
            D_val[idx] += alpha*acc;
            idx += gridDim.x*blockDim.x;
        }
    }
    template <typename T>
    void outer_product_re(int n_rows, int n_cols, T alpha, T *A, T *B,
    int D_nnz, int *D_row_idx, int *D_col_idx, T *D_val) {
        int block_size = 64;
        int grid_size = min(max(D_nnz / block_size, 1), 256);
        outer_product_kernel_re<<<grid_size, block_size>>>(n_rows, n_cols, alpha, A, B, D_nnz, D_row_idx, D_col_idx, D_val);
    }
    
    
    // --- Complex --------------------------------------------------------------
    
    template <typename T, typename T_re>
    __global__ void outer_product_kernel_cx(int n_rows, int n_cols, T_re alpha, T *A, T *B,
                                         int D_nnz, int *D_row_idx, int *D_col_idx, T *D_val) {
        int idx = blockIdx.x*blockDim.x + threadIdx.x;
        while (idx < D_nnz) {
            int i = D_row_idx[idx];
            int j = D_col_idx[idx];
            T_re acc_re = 0;
            T_re acc_im = 0;
            for (int k = 0; k < n_cols; k++) {
                T a = A[k*n_rows+i];
                T b = B[k*n_rows+j];
                // acc += A_ik * conj(B_jk)
                acc_re += a.x * b.x + a.y * b.y;
                acc_im += a.y * b.x - a.x * b.y;
            }
            // D_ij += alpha * acc
            D_val[idx].x += alpha*acc_re;
            D_val[idx].y += alpha*acc_im;
            idx += gridDim.x*blockDim.x;
        }
    }
    template <typename T, typename T_re>
    void outer_product_cx(int n_rows, int n_cols, T_re alpha, T *A, T *B,
                       int D_nnz, int *D_row_idx, int *D_col_idx, T *D_val) {
        int block_size = 64;
        int grid_size = min(max(D_nnz / block_size, 1), 256);
        outer_product_kernel_cx<<<grid_size, block_size>>>(n_rows, n_cols, alpha, A, B, D_nnz, D_row_idx, D_col_idx, D_val);
    }
    
    
    // --- Instances --------------------------------------------------------------
    
    template <typename T, typename T_re>
    void outer_product(int n_rows, int n_cols, T_re alpha, T *A, T *B,
                       int D_nnz, int *D_row_idx, int *D_col_idx, T *D_val);
    template <> // float
    void outer_product(int n_rows, int n_cols, float alpha, float *A, float *B,
                       int D_nnz, int *D_row_idx, int *D_col_idx, float *D_val) {
        outer_product_re(n_rows, n_cols, alpha, A, B, D_nnz, D_row_idx, D_col_idx, D_val);
    }
    template <> // double
    void outer_product(int n_rows, int n_cols, double alpha, double *A, double *B,
                       int D_nnz, int *D_row_idx, int *D_col_idx, double *D_val) {
        outer_product_re(n_rows, n_cols, alpha, A, B, D_nnz, D_row_idx, D_col_idx, D_val);
    }
    template <> // cx_float
    void outer_product(int n_rows, int n_cols, float alpha, cuFloatComplex *A, cuFloatComplex *B,
                       int D_nnz, int *D_row_idx, int *D_col_idx, cuFloatComplex *D_val) {
        outer_product_cx(n_rows, n_cols, alpha, A, B, D_nnz, D_row_idx, D_col_idx, D_val);
    }
    template <> // cx_double
    void outer_product(int n_rows, int n_cols, double alpha, cuDoubleComplex *A, cuDoubleComplex *B,
                       int D_nnz, int *D_row_idx, int *D_col_idx, cuDoubleComplex *D_val) {
        outer_product_cx(n_rows, n_cols, alpha, A, B, D_nnz, D_row_idx, D_col_idx, D_val);
    }
}
