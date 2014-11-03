#include <cuComplex.h>

namespace fkpm {
    __global__ void outer_product_kernel(int n_rows, int n_cols, cuFloatComplex *a, cuFloatComplex *b, float scal,
                                         int D_nnz, int *D_row_idx, int *D_col_idx, cuFloatComplex *D_val) {
        int idx = blockIdx.x*blockDim.x + threadIdx.x;
        while (idx < D_nnz) {
            int i = D_row_idx[idx];
            int j = D_col_idx[idx];
            float acc_re = 0;
            float acc_im = 0;
            for (int k = 0; k < n_cols; k++) {
                cuFloatComplex cb = b[k*n_rows+i];
                cuFloatComplex ca = a[k*n_rows+j];
                // acc += conj(b) * a
                acc_re += cuCrealf(cb) * cuCrealf(ca) + cuCimagf(cb) * cuCimagf(ca);
                acc_im += cuCrealf(cb) * cuCimagf(ca) - cuCimagf(cb) * cuCrealf(ca);
            }
            // D_val[idx] += scal * acc
            D_val[idx] = cuCaddf(D_val[idx], make_cuFloatComplex(scal * acc_re, scal * acc_im));
            idx += gridDim.x*blockDim.x;
        }
    }

    void outer_product(int n_rows, int n_cols, cuFloatComplex *a, cuFloatComplex *b, float scal,
                       int D_nnz, int *D_row_idx, int *D_col_idx, cuFloatComplex *D_val) {
        int block_size = 64;
        int grid_size = min(max(D_nnz / block_size, 1), 256);
        outer_product_kernel<<<grid_size, block_size>>>(n_rows, n_cols, a, b, scal, D_nnz, D_row_idx, D_col_idx, D_val);
    }
}
