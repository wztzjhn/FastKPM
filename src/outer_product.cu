#include <cuComplex.h>

namespace fkpm {
    
    template <typename T> T zero();
    template<> __device__ __inline__ float           zero()                         { return 0; }
    template<> __device__ __inline__ double          zero()                         { return 0; }
    template<> __device__ __inline__ cuFloatComplex  zero()                         { return make_cuFloatComplex(0, 0); }
    template<> __device__ __inline__ cuDoubleComplex zero()                         { return make_cuDoubleComplex(0, 0); }
    
    __device__ __inline__ float           add(float a, float b)                     { return a+b; }
    __device__ __inline__ double          add(double a, double b)                   { return a+b; }
    __device__ __inline__ cuFloatComplex  add(cuFloatComplex a,  cuFloatComplex b)  { return cuCaddf(a, b); }
    __device__ __inline__ cuDoubleComplex add(cuDoubleComplex a, cuDoubleComplex b) { return cuCadd(a, b); }
    
    __device__ __inline__ float           conj(float a)                             { return a; }
    __device__ __inline__ double          conj(double a)                            { return a; }
    __device__ __inline__ cuFloatComplex  conj(cuFloatComplex a)                    { return cuConjf(a); }
    __device__ __inline__ cuDoubleComplex conj(cuDoubleComplex a)                   { return cuConj(a); }
    
    __device__ __inline__ float           mul(float a, float b)                     { return a*b; }
    __device__ __inline__ double          mul(double a, double b)                   { return a*b; }
    __device__ __inline__ cuFloatComplex  mul(cuFloatComplex a,  cuFloatComplex b)  { return cuCmulf(a, b); }
    __device__ __inline__ cuDoubleComplex mul(cuDoubleComplex a, cuDoubleComplex b) { return cuCmul(a, b); }
    __device__ __inline__ cuFloatComplex  mul(float a,  cuFloatComplex b)           { return make_cuFloatComplex(a*b.x, a*b.y); }
    __device__ __inline__ cuDoubleComplex mul(double a, cuDoubleComplex b)          { return make_cuDoubleComplex(a*b.x, a*b.y); }
    
    
    template <typename T, typename T_re>
    __global__ void outer_product_kernel(int b_rows, int b_len, int n_cols, T_re alpha, T *A, T *B,
                                         int n_blocks, int *D_row_idx, int *D_col_idx, T *D_val) {
        // idx realizes each index into D_val
        for (int idx = blockIdx.x*blockDim.x + threadIdx.x;
             idx < b_len*b_len*n_blocks;
             idx += gridDim.x*blockDim.x) {
            
            int k = idx / (b_len*b_len);
            int i = D_row_idx[k];
            int j = D_col_idx[k];
            int bj = (idx / b_len) % b_len;
            int bi = idx % b_len;
            
            T acc = zero<T>();
            for (int l = 0; l < n_cols; l++) {
                T a = A[b_len*b_rows*l + b_len*i + bi];
                T b = B[b_len*b_rows*l + b_len*j + bj];
                acc = add(acc, mul(a, conj(b)));
            }
            D_val[idx] = add(D_val[idx], mul(alpha, acc));
        }
    }
    
    template <typename T, typename T_re>
    void outer_product(int b_rows, int b_len, int n_cols, T_re alpha, T *A, T *B,
                       int n_blocks, int *D_row_idx, int *D_col_idx, T *D_val) {
        int block_size = 64;
        int grid_size = min(max(b_len*b_len*n_blocks / block_size, 1), 256);
        outer_product_kernel<<<grid_size, block_size>>>(b_rows, b_len, n_cols, alpha, A, B, n_blocks, D_row_idx, D_col_idx, D_val);
    }
    
    template void outer_product(int b_rows, int b_len, int n_cols, float alpha, float *A, float *B,
                                int n_blocks, int *D_row_idx, int *D_col_idx, float *D_val);
    
    template void outer_product(int b_rows, int b_len, int n_cols, double alpha, double *A, double *B,
                                int n_blocks, int *D_row_idx, int *D_col_idx, double *D_val);
    
    template void outer_product<cuFloatComplex,float>(int b_rows, int b_len, int n_cols, float alpha, cuFloatComplex *A, cuFloatComplex *B,
                                int n_blocks, int *D_row_idx, int *D_col_idx, cuFloatComplex *D_val);
    
    template void outer_product(int b_rows, int b_len, int n_cols, double alpha, cuDoubleComplex *A, cuDoubleComplex *B,
                                int n_blocks, int *D_row_idx, int *D_col_idx, cuDoubleComplex *D_val);
}
