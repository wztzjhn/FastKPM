#include "fastkpm.h"

#ifndef WITH_CUDA

namespace fkpm {
    template <typename T>
    std::shared_ptr<Engine<T>> mk_engine_cuSPARSE(Vec<int> devices) {
        return nullptr;
    }
    template std::shared_ptr<Engine<float>> mk_engine_cuSPARSE(Vec<int> devices);
    template std::shared_ptr<Engine<double>> mk_engine_cuSPARSE(Vec<int> devices);
    template std::shared_ptr<Engine<cx_float>> mk_engine_cuSPARSE(Vec<int> devices);
    template std::shared_ptr<Engine<cx_double>> mk_engine_cuSPARSE(Vec<int> devices);
}

#else

#include <cstdlib>
#include <cassert>
#include <thread>
#include <cuda_runtime.h>
#include <cusparse_v2.h>
#include <cublas_v2.h>

#define TRY(x) gen_cuda_test_error((x), __FILE__, __LINE__, #x)

namespace fkpm {
    inline const char *gen_cuda_error_string(cudaError_t stat) {
        return cudaGetErrorString(stat);
    }
    inline const char *gen_cuda_error_string(cublasStatus_t stat) {
        switch (stat) {
            case CUBLAS_STATUS_SUCCESS:             return "CUBLAS_STATUS_SUCCESS";
            case CUBLAS_STATUS_NOT_INITIALIZED:     return "CUBLAS_STATUS_NOT_INITIALIZED";
            case CUBLAS_STATUS_ALLOC_FAILED:        return "CUBLAS_STATUS_ALLOC_FAILED";
            case CUBLAS_STATUS_INVALID_VALUE:       return "CUBLAS_STATUS_INVALID_VALUE";
            case CUBLAS_STATUS_ARCH_MISMATCH:       return "CUBLAS_STATUS_ARCH_MISMATCH";
            case CUBLAS_STATUS_MAPPING_ERROR:       return "CUBLAS_STATUS_MAPPING_ERROR";
            case CUBLAS_STATUS_EXECUTION_FAILED:    return "CUBLAS_STATUS_EXECUTION_FAILED";
            case CUBLAS_STATUS_INTERNAL_ERROR:      return "CUBLAS_STATUS_INTERNAL_ERROR";
            case CUBLAS_STATUS_NOT_SUPPORTED:       return "CUBLAS_STATUS_NOT_SUPPORTED";
            default:                                return "<unknown cublas error>";
        }
    }
    inline const char *gen_cuda_error_string(cusparseStatus_t stat) {
        switch (stat) {
            case CUSPARSE_STATUS_SUCCESS:           return "CUSPARSE_STATUS_SUCCESS";
            case CUSPARSE_STATUS_NOT_INITIALIZED:   return "CUSPARSE_STATUS_NOT_INITIALIZED";
            case CUSPARSE_STATUS_ALLOC_FAILED:      return "CUSPARSE_STATUS_ALLOC_FAILED";
            case CUSPARSE_STATUS_INVALID_VALUE:     return "CUSPARSE_STATUS_INVALID_VALUE";
            case CUSPARSE_STATUS_ARCH_MISMATCH:     return "CUSPARSE_STATUS_ARCH_MISMATCH";
            case CUSPARSE_STATUS_MAPPING_ERROR:     return "CUSPARSE_STATUS_MAPPING_ERROR";
            case CUSPARSE_STATUS_EXECUTION_FAILED:  return "CUSPARSE_STATUS_EXECUTION_FAILED";
            case CUSPARSE_STATUS_INTERNAL_ERROR:    return "CUSPARSE_STATUS_INTERNAL_ERROR";
            case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED: return "CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
            case CUSPARSE_STATUS_ZERO_PIVOT:        return "CUSPARSE_STATUS_ZERO_PIVOT";
            default:                                return "<unknown cusparse error>";
        }
    }
    template <typename T>
    void gen_cuda_test_error(T stat, char const* file, int line, char const* code) {
        if (stat) {
            std::cerr << file << ":" << line <<  ", " << code << ", Error: " << gen_cuda_error_string(stat) << std::endl;
            std::abort();
        }
    }
    
    inline float           cuda_cast(float  x)    { return x; };
    inline double          cuda_cast(double x)    { return x; };
    inline cuFloatComplex  cuda_cast(cx_float x)  { return make_cuFloatComplex(x.real(), x.imag()); };
    inline cuDoubleComplex cuda_cast(cx_double x) { return make_cuDoubleComplex(x.real(), x.imag()); };
    
    inline float  cuda_real(float  x)          { return x; };
    inline double cuda_real(double x)          { return x; };
    inline float  cuda_real(cuFloatComplex x)  { return x.x; };
    inline double cuda_real(cuDoubleComplex x) { return x.x; };
    inline cx_double cuda_cmplx(float x)           {return cx_double(x, 0.0); };
    inline cx_double cuda_cmplx(double x)          {return cx_double(x, 0.0); };
    inline cx_double cuda_cmplx(cuFloatComplex x)  {return cx_double(x.x, x.y); };
    inline cx_double cuda_cmplx(cuDoubleComplex x) {return cx_double(x.x, x.y); };

    // -- GEAM (matrix-matrix addition/transposition) --
    inline // float
    cublasStatus_t gen_geam(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, const float *alpha,
                            const float *A, int lda, const float *beta, const float *B, int ldb, float *C, int ldc) {
        return cublasSgeam(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
    }
    inline // double
    cublasStatus_t gen_geam(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, const double *alpha,
                            const double *A, int lda, const double *beta, const double *B, int ldb, double *C, int ldc) {
        return cublasDgeam(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
    }
    inline // cx_float
    cublasStatus_t gen_geam(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, const cuFloatComplex *alpha,
                            const cuFloatComplex *A, int lda, const cuFloatComplex *beta, const cuFloatComplex *B, int ldb, cuFloatComplex *C, int ldc) {
        return cublasCgeam(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
    }
    inline // cx_double
    cublasStatus_t gen_geam(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, const cuDoubleComplex *alpha, 
                            const cuDoubleComplex *A, int lda, const cuDoubleComplex *beta, const cuDoubleComplex *B, int ldb, cuDoubleComplex *C, int ldc) {
        return cublasZgeam(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
    }
    
    // -- BSRMV (block sparse matrix-vector multiplication) --
    inline // float
    cusparseStatus_t gen_bsrmv(cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA,
                               int mb, int nb, int nnzb, const float *alpha, const cusparseMatDescr_t descrA,
                               const float *bsrValA, const int *bsrRowPtrA, const int *bsrColIndA, const int blockDim,
                               const float *x, const float *beta, float *y) {
        return cusparseSbsrmv(handle, dirA, transA, mb, nb, nnzb, alpha, descrA, bsrValA, bsrRowPtrA, bsrColIndA, blockDim, x, beta, y);
    }
    inline // double
    cusparseStatus_t gen_bsrmv(cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA,
                               int mb, int nb, int nnzb, const double *alpha, const cusparseMatDescr_t descrA,
                               const double *bsrValA, const int *bsrRowPtrA, const int *bsrColIndA, const int blockDim,
                               const double *x, const double *beta, double *y) {
        return cusparseDbsrmv(handle, dirA, transA, mb, nb, nnzb, alpha, descrA, bsrValA, bsrRowPtrA, bsrColIndA, blockDim, x, beta, y);
    }
    inline // cx_float
    cusparseStatus_t gen_bsrmv(cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA,
                               int mb, int nb, int nnzb, const cuFloatComplex *alpha, const cusparseMatDescr_t descrA,
                               const cuFloatComplex *bsrValA, const int *bsrRowPtrA, const int *bsrColIndA, const int blockDim,
                               const cuFloatComplex *x, const cuFloatComplex *beta, cuFloatComplex *y) {
        return cusparseCbsrmv(handle, dirA, transA, mb, nb, nnzb, alpha, descrA, bsrValA, bsrRowPtrA, bsrColIndA, blockDim, x, beta, y);
    }
    inline // cx_double
    cusparseStatus_t gen_bsrmv(cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA,
                               int mb, int nb, int nnzb, const cuDoubleComplex *alpha, const cusparseMatDescr_t descrA,
                               const cuDoubleComplex *bsrValA, const int *bsrRowPtrA, const int *bsrColIndA, const int blockDim,
                               const cuDoubleComplex *x,const cuDoubleComplex *beta, cuDoubleComplex *y) {
        return cusparseZbsrmv(handle, dirA, transA, mb, nb, nnzb, alpha, descrA, bsrValA, bsrRowPtrA, bsrColIndA, blockDim, x, beta, y);
    }
    
    // -- CSRMM (sparse-dense matrix multiplication) --
    inline // float
    cusparseStatus_t gen_csrmm(cusparseHandle_t handle, cusparseOperation_t transA, int m, int n, int k, int nnz, const float *alpha,
                               const cusparseMatDescr_t descrA, const float  *csrValA, const int *csrRowPtrA, const int *csrColIndA,
                               const float *B, int ldb, const float *beta, float *C, int ldc) {
        return cusparseScsrmm(handle, transA, m, n, k, nnz, alpha, descrA, csrValA, csrRowPtrA, csrColIndA, B, ldb, beta, C, ldc);
    }
    inline // double
    cusparseStatus_t gen_csrmm(cusparseHandle_t handle, cusparseOperation_t transA, int m, int n, int k, int nnz, const double *alpha,
                               const cusparseMatDescr_t descrA, const double  *csrValA, const int *csrRowPtrA, const int *csrColIndA,
                               const double *B, int ldb, const double *beta, double *C, int ldc) {
        return cusparseDcsrmm(handle, transA, m, n, k, nnz, alpha, descrA, csrValA, csrRowPtrA, csrColIndA, B, ldb, beta, C, ldc);
    }
    inline // cx_float
    cusparseStatus_t gen_csrmm(cusparseHandle_t handle, cusparseOperation_t transA, int m, int n, int k, int nnz, const cuFloatComplex *alpha,
                               const cusparseMatDescr_t descrA, const cuFloatComplex  *csrValA, const int *csrRowPtrA, const int *csrColIndA,
                               const cuFloatComplex *B, int ldb, const cuFloatComplex *beta, cuFloatComplex *C, int ldc) {
        return cusparseCcsrmm(handle, transA, m, n, k, nnz, alpha, descrA, csrValA, csrRowPtrA, csrColIndA, B, ldb, beta, C, ldc);
    }
    inline // cx_double
    cusparseStatus_t gen_csrmm(cusparseHandle_t handle, cusparseOperation_t transA, int m, int n, int k, int nnz, const cuDoubleComplex *alpha,
                               const cusparseMatDescr_t descrA, const cuDoubleComplex  *csrValA, const int *csrRowPtrA, const int *csrColIndA,
                               const cuDoubleComplex *B, int ldb, const cuDoubleComplex *beta, cuDoubleComplex *C, int ldc) {
        return cusparseZcsrmm(handle, transA, m, n, k, nnz, alpha, descrA, csrValA, csrRowPtrA, csrColIndA, B, ldb, beta, C, ldc);
    }
    
    // -- BSRMM (block sparse-dense matrix multiplication) --
    inline // float
    cusparseStatus_t gen_bsrmm(cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA, cusparseOperation_t transB,
                               int mb, int n, int kb, int nnzb, const float *alpha, const cusparseMatDescr_t descrA,
                               const float *bsrValA, const int *bsrRowPtrA, const int *bsrColIndA, const int blockDim,
                               const float *B, const int ldb, const float *beta, float *C, int ldc) {
        return cusparseSbsrmm(handle, dirA, transA, transB, mb, n, kb, nnzb, alpha, descrA, bsrValA, bsrRowPtrA, bsrColIndA, blockDim, B, ldb, beta, C, ldc);
    }
    inline // double
    cusparseStatus_t gen_bsrmm(cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA, cusparseOperation_t transB,
                               int mb, int n, int kb, int nnzb, const double *alpha, const cusparseMatDescr_t descrA,
                               const double *bsrValA, const int *bsrRowPtrA, const int *bsrColIndA, const int blockDim,
                               const double *B, const int ldb, const double *beta, double *C, int ldc) {
        return cusparseDbsrmm(handle, dirA, transA, transB, mb, n, kb, nnzb, alpha, descrA, bsrValA, bsrRowPtrA, bsrColIndA, blockDim, B, ldb, beta, C, ldc);
    }
    inline // cx_float
    cusparseStatus_t gen_bsrmm(cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA, cusparseOperation_t transB,
                               int mb, int n, int kb, int nnzb, const cuFloatComplex *alpha, const cusparseMatDescr_t descrA,
                               const cuFloatComplex *bsrValA, const int *bsrRowPtrA, const int *bsrColIndA, const int blockDim,
                               const cuFloatComplex *B, const int ldb, const cuFloatComplex *beta, cuFloatComplex *C, int ldc) {
        return cusparseCbsrmm(handle, dirA, transA, transB, mb, n, kb, nnzb, alpha, descrA, bsrValA, bsrRowPtrA, bsrColIndA, blockDim, B, ldb, beta, C, ldc);
    }
    inline // cx_double
    cusparseStatus_t gen_bsrmm(cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA, cusparseOperation_t transB,
                               int mb, int n, int kb, int nnzb, const cuDoubleComplex *alpha, const cusparseMatDescr_t descrA,
                               const cuDoubleComplex *bsrValA, const int *bsrRowPtrA, const int *bsrColIndA, const int blockDim,
                               const cuDoubleComplex *B, const int ldb, const cuDoubleComplex *beta, cuDoubleComplex *C, int ldc) {
        return cusparseZbsrmm(handle, dirA, transA, transB, mb, n, kb, nnzb, alpha, descrA, bsrValA, bsrRowPtrA, bsrColIndA, blockDim, B, ldb, beta, C, ldc);
    }
    
    // -- DOTC (complex-conjugated dot product) --
    inline // float
    cublasStatus_t gen_dotc(cublasHandle_t handle, int n, const float *x, int incx, const float *y, int incy, float *result) {
        return cublasSdot(handle, n, x, incx, y, incy, result);
    }
    inline // double
    cublasStatus_t gen_dotc(cublasHandle_t handle, int n, const double *x, int incx, const double *y, int incy, double *result) {
        return cublasDdot(handle, n, x, incx, y, incy, result);
    }
    inline // cx_float
    cublasStatus_t gen_dotc(cublasHandle_t handle, int n, const cuFloatComplex *x, int incx, const cuFloatComplex *y, int incy, cuFloatComplex *result) {
        return cublasCdotc(handle, n, x, incx, y, incy, result);
    }
    inline // cx_double
    cublasStatus_t gen_dotc(cublasHandle_t handle, int n, const cuDoubleComplex *x, int incx, const cuDoubleComplex *y, int incy, cuDoubleComplex *result) {
        return cublasZdotc(handle, n, x, incx, y, incy, result);
    }
    
    // -- AXPY (y = alpha*x + y) --
    inline // float
    cublasStatus_t gen_axpy(cublasHandle_t handle, int n, const float *alpha, const float *x, int incx, float *y, int incy) {
        return cublasSaxpy(handle, n, alpha, x, incx, y, incy);
    }
    inline // double
    cublasStatus_t gen_axpy(cublasHandle_t handle, int n, const double *alpha, const double *x, int incx, double *y, int incy) {
        return cublasDaxpy(handle, n, alpha, x, incx, y, incy);
    }
    inline // cx_float
    cublasStatus_t gen_axpy(cublasHandle_t handle, int n, const cuFloatComplex *alpha, const cuFloatComplex *x, int incx, cuFloatComplex *y, int incy) {
        return cublasCaxpy(handle, n, alpha, x, incx, y, incy);
    }
    inline // cx_double
    cublasStatus_t gen_axpy(cublasHandle_t handle, int n, const cuDoubleComplex *alpha, const cuDoubleComplex *x, int incx, cuDoubleComplex *y, int incy) {
        return cublasZaxpy(handle, n, alpha, x, incx, y, incy);
    }
    
    // -- SCAL (x = alpha*x) --
    inline // float*Vec<float>
    cublasStatus_t gen_scal(cublasHandle_t handle, int n, const float *alpha, float *x, int incx) {
        return cublasSscal(handle, n, alpha, x, incx);
    }
    inline // double*Vec<double>
    cublasStatus_t gen_scal(cublasHandle_t handle, int n, const double *alpha, double *x, int incx) {
        return cublasDscal(handle, n, alpha, x, incx);
    }
    inline // cx_float*Vec<cx_float>
    cublasStatus_t gen_scal(cublasHandle_t handle, int n, const cuComplex *alpha, cuComplex *x, int incx) {
        return cublasCscal(handle, n, alpha, x, incx);
    }
    inline // float*Vec<cx_float>
    cublasStatus_t gen_scal(cublasHandle_t handle, int n, const float *alpha, cuComplex *x, int incx) {
        return cublasCsscal(handle, n, alpha, x, incx);
    }
    inline // cx_double*Vec<cx_double>
    cublasStatus_t gen_scal(cublasHandle_t handle, int n, const cuDoubleComplex *alpha, cuDoubleComplex *x, int incx) {
        return cublasZscal(handle, n, alpha, x, incx);
    }
    inline // double*Vec<cx_double>
    cublasStatus_t gen_scal(cublasHandle_t handle, int n, const double *alpha, cuDoubleComplex *x, int incx) {
        return cublasZdscal(handle, n, alpha, x, incx);
    }
    
    template <typename T, typename T_re>
    void outer_product(int b_rows, int b_len, int n_cols, T_re alpha, const T *A, const T *B,
                       int n_blocks, const int *D_row_idx, const int *D_col_idx, T *D_val);
    
    template <typename T>
    class CuVec {
    public:
        int size = 0;
        int capacity = 0;
        T* ptr = 0;
        void resize(int size) {
            this->size = size;
            if (size > capacity) {
                capacity = (3*size)/2;
                TRY(cudaFree(ptr));
                TRY(cudaMalloc(&ptr, capacity*sizeof(T)));
            }
        }
        void memset(int value) {
            TRY(cudaMemset(ptr, value, size*sizeof(T)));
        }
        void from_host(int size, T const* src) {
            resize(size);
            TRY(cudaMemcpy(ptr, src, size*sizeof(T), cudaMemcpyHostToDevice));
        }
        void from_device(CuVec<T> const& that) {
            resize(that.size);
            TRY(cudaMemcpy(ptr, that.ptr, size*sizeof(T), cudaMemcpyDeviceToDevice));
        }
        void to_host(T* dst) const {
            TRY(cudaMemcpy(dst, ptr, size*sizeof(T), cudaMemcpyDeviceToHost));
        }
        void deallocate() {
            TRY(cudaFree(ptr));
            size = capacity = 0;
            ptr = 0;
        }
    };
    
    template <typename T>
    class Engine_cuSPARSE: public Engine<T> {
    public:
        typedef decltype(std::real(T(0))) T_re;
        typedef decltype(cuda_cast(T(0))) T_cu;
        const T_cu zero_cu = cuda_cast(T(0));
        const T_cu one_cu  = cuda_cast(T(1));
        
        int device = 0;
        EnergyScale es{0, 0};
        int n_rows = 0;
        int b_len = 0;
        int n_blocks = 0;
        Vec<T> Hs_val;
        
        cublasHandle_t bl_handle;
        cusparseHandle_t cs_handle;
        cusparseMatDescr_t cs_mat_descr;
        
        CuVec<T> R_d, xi_d, t_d;
        CuVec<T> a_d[3];
        CuVec<T> b_d[3];
        
        CuVec<int> HColIndex_d, HRowPtr_d;
        CuVec<T> HVal_d;
        
        CuVec<int> DRowIndex_d, DColIndex_d;
        CuVec<T> DVal_d;
        
        Engine_cuSPARSE(int device) {
            this->device = device;
            TRY(cudaSetDevice(device));
            
            TRY(cublasCreate(&bl_handle));
            TRY(cusparseCreate(&cs_handle));
            TRY(cusparseCreateMatDescr(&cs_mat_descr));
            TRY(cusparseSetMatType(cs_mat_descr, CUSPARSE_MATRIX_TYPE_GENERAL));
            TRY(cusparseSetMatIndexBase(cs_mat_descr, CUSPARSE_INDEX_BASE_ZERO));
        }
        
        ~Engine_cuSPARSE() {
            TRY(cudaSetDevice(device));
            
            R_d.deallocate();
            xi_d.deallocate();
            t_d.deallocate();
            for (int i = 0; i < 3; i++) {
                a_d[i].deallocate();
                b_d[i].deallocate();
            }
            
            HRowPtr_d.deallocate();
            HColIndex_d.deallocate();
            HVal_d.deallocate();
            
            DRowIndex_d.deallocate();
            DColIndex_d.deallocate();
            DVal_d.deallocate();
            
            TRY(cusparseDestroyMatDescr(cs_mat_descr));
            TRY(cusparseDestroy(cs_handle));
            TRY(cublasDestroy(bl_handle));
        }
        
        EnergyScale energy_scale(SpMatBsr<T> const& H, double extend, int iters) {
            TRY(cudaSetDevice(device));
            assert(H.n_rows == H.n_cols);
            
            // Storage for sparse matrix H
            int n_rows = H.n_rows;
            int b_len = H.b_len;
            int n_blocks = H.n_blocks();
            CuVec<int> HColIndex_d, HRowPtr_d;
            CuVec<T> HVal_d;
            HRowPtr_d.from_host(H.row_ptr.size(), H.row_ptr.data());
            HColIndex_d.from_host(H.col_idx.size(), H.col_idx.data());
            HVal_d.from_host(H.val.size(), H.val.data());
            
            // Storage for dense vectors
            CuVec<T> v0_d, v1_d, w_d;
            v0_d.resize(n_rows);
            v0_d.memset(0);
            arma::Col<T> v1(n_rows);
            v1.randn();
            v1 /= std::sqrt(std::real(arma::cdot(v1, v1)));
            v1_d.from_host(n_rows, v1.memptr());
            w_d.resize(n_rows);
            
            // Storage for tridiagonal matrix elements
            Vec<double> alpha(iters), beta(iters);
            beta[0] = 0;
            
            for (int j = 1; j < iters; j++) {
                // w = H * v1
                TRY(gen_bsrmv(cs_handle, CUSPARSE_DIRECTION_COLUMN, CUSPARSE_OPERATION_NON_TRANSPOSE,
                              n_rows, n_rows, n_blocks, &one_cu, cs_mat_descr,
                              (T_cu *)HVal_d.ptr, HRowPtr_d.ptr, HColIndex_d.ptr, b_len,
                              (T_cu *)v1_d.ptr, &zero_cu, (T_cu *)w_d.ptr));
                
                // alpha[j-1] = real(w dot v1);
                T_cu cdot_res;
                TRY(gen_dotc(bl_handle, n_rows, (T_cu *)w_d.ptr, 1, (T_cu *)v1_d.ptr, 1, &cdot_res));
                alpha[j-1] = cuda_real(cdot_res);
                
                // w = w - alpha[j-1] * v1 - beta[j-1] * v0;
                T_cu scal = cuda_cast(T(-alpha[j-1]));
                TRY(gen_axpy(bl_handle, n_rows, &scal, (T_cu *)v1_d.ptr, 1, (T_cu *)w_d.ptr, 1));
                scal = cuda_cast(T(-beta[j-1]));
                TRY(gen_axpy(bl_handle, n_rows, &scal, (T_cu *)v0_d.ptr, 1, (T_cu *)w_d.ptr, 1));
                
                // beta[j] = sqrt(real(w dot w))
                TRY(gen_dotc(bl_handle, n_rows, (T_cu *)w_d.ptr, 1, (T_cu *)w_d.ptr, 1, &cdot_res));
                beta[j] = std::sqrt(cuda_real(cdot_res));
                
                // v0 = v1;
                v0_d.from_device(v1_d);
                
                // v1 = w / beta[j];
                v1_d.from_device(w_d);
                scal = cuda_cast(T(1.0 / beta[j]));
                TRY(gen_scal(bl_handle, n_rows, &scal, (T_cu *)v1_d.ptr, 1));
            }
            
            // w = H * v1;
            TRY(gen_bsrmv(cs_handle, CUSPARSE_DIRECTION_COLUMN, CUSPARSE_OPERATION_NON_TRANSPOSE,
                          n_rows, n_rows, n_blocks, &one_cu, cs_mat_descr,
                          (T_cu *)HVal_d.ptr, HRowPtr_d.ptr, HColIndex_d.ptr, b_len,
                          (T_cu *)v1_d.ptr, &zero_cu, (T_cu *)w_d.ptr));
            
            // alpha[iters-1] = real(w dot v1)
            T_cu cdot_res;
            TRY(gen_dotc(bl_handle, n_rows, (T_cu *)w_d.ptr, 1, (T_cu *)v1_d.ptr, 1, &cdot_res));
            alpha[iters-1] = cuda_real(cdot_res);
            
            // Deallocate all CUDA storage
            v0_d.deallocate();
            v1_d.deallocate();
            w_d.deallocate();
            HRowPtr_d.deallocate();
            HColIndex_d.deallocate();
            HVal_d.deallocate();
            
            // Find eigenvalues of tridiagonal matrix
            arma::mat tri(iters, iters);
            tri.zeros();
            tri(0, 0) = alpha[0];
            for (int j = 1; j < iters; j++) {
                tri(j, j-1) = beta[j];
                tri(j-1, j) = beta[j];
                tri(j, j) = alpha[j];
            }
            arma::vec evals = arma::eig_sym(tri);
            
            // Stretch energy scale by amount "extend" on each side
            double eig_min = *std::min_element(evals.begin(), evals.end());
            double eig_max = *std::max_element(evals.begin(), evals.end());
            double slack = extend * (eig_max - eig_min);
            return {eig_min-slack, eig_max+slack};
        }
        
        void transfer_R() {
            TRY(cudaSetDevice(device));
            
            if (this->R.n_cols == 0) {
                std::cerr << "Cannot handle empty R matrix! Device " << device << ". Exiting.\n";
                std::exit(EXIT_FAILURE);
            }
            
            int sz = this->R.size();
            xi_d.resize(sz);
            R_d.resize(sz);
            t_d.resize(sz);
            for (int i = 0; i < 3; i++) {
                a_d[i].resize(sz);
                b_d[i].resize(sz);
            }

            // t_d = R
            t_d.from_host(sz, this->R.memptr());
            // R_d = transpose(t_d)
            int n = this->R.n_rows;
            int s = this->R.n_cols;
            TRY(gen_geam(bl_handle, CUBLAS_OP_T, CUBLAS_OP_T, s, n, &one_cu, (T_cu *)t_d.ptr, n,
                         &zero_cu, (T_cu *)t_d.ptr, n, (T_cu *)R_d.ptr, s));
        }
        
        void set_H(SpMatBsr<T> const& H, EnergyScale const& es) {
            TRY(cudaSetDevice(device));
            assert(H.n_rows == H.n_cols);
            
            this->es = es;
            n_rows = H.n_rows;
            b_len = H.b_len;
            n_blocks = H.n_blocks();
            
            // Hs = H/es.mag()
            Hs_val.resize(H.val.size());
            for (int k = 0; k < H.val.size(); k++) {
                Hs_val[k] = H.val[k] / T_re(es.mag());
            }
            // Hs = Hs - es.avg()/es.mag()
            int diag_cnt = 0;
            T_re es_shift = es.avg() / es.mag();
            for (int k = 0; k < n_blocks; k++) {
                if (H.row_idx[k] == H.col_idx[k]) {
                    T* v = &Hs_val[b_len*b_len*k];
                    for (int bi = 0; bi < b_len; bi++) {
                        v[b_len*bi+bi] -= es_shift;
                    }
                    diag_cnt++;
                }
            }
            assert(diag_cnt == n_rows);
            
            HRowPtr_d.from_host(H.row_ptr.size(), H.row_ptr.data());
            HColIndex_d.from_host(H.col_idx.size(), H.col_idx.data());
            HVal_d.from_host(Hs_val.size(), Hs_val.data());
        }
        
        // C = (alpha H B^T)^T + beta C
        // H: n*n, B_d: s*n, C_d: s*n, t_d: n*s
        void cgemm_H(T alpha, CuVec<T> const& B_d, T beta, CuVec<T> const& C_d, int s = -1) {
            int n = this->R.n_rows;
            if (s < 1) s = this->R.n_cols;
            
            // t = H B^T
            TRY(gen_bsrmm(cs_handle, CUSPARSE_DIRECTION_COLUMN, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE,
                          n_rows, s, n_rows, n_blocks, // H # block rows, # element cols, # block cols, # nonzero blocks
                          &one_cu,
                          cs_mat_descr, (T_cu *)HVal_d.ptr, HRowPtr_d.ptr, HColIndex_d.ptr, b_len, // H matrix
                          (T_cu *)B_d.ptr, s,
                          &zero_cu,
                          (T_cu *)t_d.ptr, n));
            
            // C = alpha t^T + beta C
            T_cu alpha_cu = cuda_cast(alpha);
            T_cu beta_cu  = cuda_cast(beta);
            TRY(gen_geam(bl_handle, CUBLAS_OP_T, CUBLAS_OP_N, s, n, &alpha_cu, (T_cu *)t_d.ptr, n,
                         &beta_cu, (T_cu *)C_d.ptr, s, (T_cu *)C_d.ptr, s));
        }
        
        // y = alpha H x + beta y
        void cgemv_H(T alpha, CuVec<T> const& x_d, T beta, CuVec<T> const& y_d) {
            T_cu alpha_cu = cuda_cast(alpha);
            T_cu beta_cu  = cuda_cast(beta);
            TRY(gen_bsrmv(cs_handle, CUSPARSE_DIRECTION_COLUMN, CUSPARSE_OPERATION_NON_TRANSPOSE,
                          n_rows, n_rows, n_blocks, &alpha_cu, cs_mat_descr,
                          (T_cu *)HVal_d.ptr, HRowPtr_d.ptr, HColIndex_d.ptr, b_len,
                          (T_cu *)x_d.ptr, &beta_cu, (T_cu *)y_d.ptr));
        }
        
        Vec<double> moments(int M) {
            TRY(cudaSetDevice(device));
            assert(b_len*n_rows == this->R.n_rows);
            assert(M % 2 == 0);
            
            Vec<double> mu(M);
            
            a_d[0].from_device(R_d);            // a0 = \alpha_0 = R
            a_d[1].memset(0);
            cgemm_H(1, R_d, 0, a_d[1]);         // a1 = \alpha_1 = H R
            
            T_cu result1, result2;
            TRY(gen_dotc(bl_handle, this->R.size(), (T_cu *)a_d[0].ptr, 1, (T_cu *)a_d[0].ptr, 1, &result1));
            TRY(gen_dotc(bl_handle, this->R.size(), (T_cu *)a_d[1].ptr, 1, (T_cu *)a_d[0].ptr, 1, &result2));
            mu[0] = cuda_real(result1);
            mu[1] = cuda_real(result2);
            
            for (int m = 1; m < M/2; m++) {
                a_d[2].from_device(a_d[0]);
                cgemm_H(2, a_d[1], -1, a_d[2]); // a2 = 2 H a1 - a0
                
                auto temp = a_d[0];
                a_d[0] = a_d[1];
                a_d[1] = a_d[2];
                a_d[2] = temp;
                
                // 2 \alpha_m^\dagger \alpha_m - mu0
                T_cu result1, result2;
                TRY(gen_dotc(bl_handle, this->R.size(), (T_cu *)a_d[0].ptr, 1, (T_cu *)a_d[0].ptr, 1, &result1));
                TRY(gen_dotc(bl_handle, this->R.size(), (T_cu *)a_d[1].ptr, 1, (T_cu *)a_d[0].ptr, 1, &result2));
                mu[2*m+0] = 2 * cuda_real(result1) - mu[0];
                // 2 \alpha_{m+1}^\dagger \alpha_m - mu1
                mu[2*m+1] = 2 * cuda_real(result2) - mu[1];
            }
            
            return mu;
        }
        
        // memory usage in moments2_v1 ~ a_chunk_ncols * R_chunk_ncols (if not including R used outside)
        Vec<Vec<cx_double>> moments2_v1(int M, SpMatBsr<T> const& j1op, SpMatBsr<T> const& j2op,
                                        int a_chunk_ncols, int R_chunk_ncols) {
            Vec<Vec<cx_double>> mu(M);
            for (int i = 0; i < M; i++) mu[i].resize(M, 0);
            TRY(cudaSetDevice(device));
            size_t mem_free, mem_total;
            double mem_percent;
            TRY(cudaMemGetInfo(&mem_free,&mem_total));
            mem_percent = 1.0- (double) mem_free/mem_total;
            if(mem_percent > 0.7) std::cout << "Warning1: gpu mem used " << mem_percent << "!\n";
            
            CuVec<int> j1_RowPtr_d, j1_ColIdx_d, j2_RowPtr_d, j2_ColIdx_d;
            CuVec<T>   j1_Values_d, j2_Values_d;
            j1_RowPtr_d.from_host(j1op.row_ptr.size(), j1op.row_ptr.data());
            j1_ColIdx_d.from_host(j1op.col_idx.size(), j1op.col_idx.data());
            j1_Values_d.from_host(j1op.val.size(),     j1op.val.data());
            j2_RowPtr_d.from_host(j2op.row_ptr.size(), j2op.row_ptr.data());
            j2_ColIdx_d.from_host(j2op.col_idx.size(), j2op.col_idx.data());
            j2_Values_d.from_host(j2op.val.size(),     j2op.val.data());
            int n  = this->R.n_rows;
            assert(b_len * n_rows == n);
            assert(j1op.n_rows == n && j1op.n_cols == n);
            assert(j2op.n_rows == n && j2op.n_cols == n);
            assert(M % 2 == 0);
            if (a_chunk_ncols < 3) a_chunk_ncols = 10;
            if (R_chunk_ncols < 1) R_chunk_ncols = 64;
            if (a_chunk_ncols > M) a_chunk_ncols = M;
            if (R_chunk_ncols > this->R.n_cols) R_chunk_ncols = this->R.n_cols;

            Vec<CuVec<T>> alpha(a_chunk_ncols), atild(a_chunk_ncols);
            CuVec<T> atild0_d, Rchunk_d, temp_d;
            T_cu result_temp;
            int s  = R_chunk_ncols;
            int sz = n * s;
            for (int i = 0; i < a_chunk_ncols; i++) alpha[i].resize(sz);
            for (int i = 0; i < a_chunk_ncols; i++) atild[i].resize(sz);
            atild0_d.resize(sz);
            Rchunk_d.resize(sz);
            temp_d.resize(sz);
            Rchunk_d.memset(0);
            alpha[1].memset(0);
            atild[1].memset(0);
            t_d.memset(0);
            
            TRY(cudaMemGetInfo(&mem_free,&mem_total));
            mem_percent = 1.0- (double) mem_free/mem_total;
            if(mem_percent > 0.7) std::cout << "Warning2: gpu mem used " << mem_percent << "!\n";
            
            for (int k = 0; k < this->R.n_cols; k += R_chunk_ncols) {
                if (k + R_chunk_ncols > this->R.n_cols) {
                    s  = this->R.n_cols - k;
                    sz = n * s;
                    for (int i = 0; i < a_chunk_ncols; i++) alpha[i].resize(sz);
                    for (int i = 0; i < a_chunk_ncols; i++) atild[i].resize(sz);
                    atild0_d.resize(sz);
                    Rchunk_d.resize(sz);
                    temp_d.resize(sz);
                    Rchunk_d.memset(0);
                }
                temp_d.from_host(sz, this->R.colptr(k));
                TRY(gen_geam(bl_handle, CUBLAS_OP_T, CUBLAS_OP_N, s, n, &one_cu, (T_cu *)temp_d.ptr, n,
                             &zero_cu, (T_cu *)Rchunk_d.ptr, s, (T_cu *)Rchunk_d.ptr, s));  // transfer Rchunk_d
                TRY(gen_bsrmm(cs_handle, CUSPARSE_DIRECTION_COLUMN, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE,
                              n_rows, s, n_rows, n_blocks, &one_cu, cs_mat_descr,
                              (T_cu *)j1_Values_d.ptr, j1_RowPtr_d.ptr, j1_ColIdx_d.ptr, b_len,
                              (T_cu *)Rchunk_d.ptr, s, &zero_cu, (T_cu *)temp_d.ptr, n));
                TRY(gen_geam(bl_handle, CUBLAS_OP_T, CUBLAS_OP_N, s, n, &one_cu, (T_cu *)temp_d.ptr, n,
                             &zero_cu, (T_cu *)atild0_d.ptr, s, (T_cu *)atild0_d.ptr, s));  // atild0_d = (j1 * Rchunk_d^T)^T
                int alpha_begin = 0;
                int alpha_end   = a_chunk_ncols - 1;
                alpha[0].from_device(Rchunk_d);         // \alpha_0^T
                cgemm_H(1, alpha[0], 0, alpha[1], s);      // \alpha_1^T
                while (alpha_begin <= alpha_end) {
                    if (alpha_begin != 0) {
                        alpha[0].from_device(alpha[a_chunk_ncols-2]);
                        cgemm_H(2, alpha[a_chunk_ncols-1], -1, alpha[0], s);
                        alpha[1].from_device(alpha[a_chunk_ncols-1]);
                        cgemm_H(2, alpha[0], -1, alpha[1], s);
                    }
                    for (int m1 = 2; m1 <= alpha_end - alpha_begin; m1++) {
                        alpha[m1].from_device(alpha[m1-2]);
                        cgemm_H(2, alpha[m1-1], -1, alpha[m1], s);
                    }
                    int atild_begin = 0;
                    int atild_end   = a_chunk_ncols - 1;
                    atild[0].from_device(atild0_d);
                    cgemm_H(1, atild[0], 0, atild[1], s);
                    while (atild_begin <= atild_end) {
                        if (atild_begin != 0) {
                            atild[0].from_device(atild[a_chunk_ncols-2]);
                            cgemm_H(2, atild[a_chunk_ncols-1], -1, atild[0], s);
                            atild[1].from_device(atild[a_chunk_ncols-1]);
                            cgemm_H(2, atild[0], -1, atild[1], s);
                        }
                        for (int m2 = 2; m2 <= atild_end - atild_begin; m2++) {
                            atild[m2].from_device(atild[m2-2]);
                            cgemm_H(2, atild[m2-1], -1, atild[m2], s);
                        }
                        for (int m2 = atild_begin; m2 <= atild_end; m2++) {
                            TRY(gen_bsrmm(cs_handle, CUSPARSE_DIRECTION_COLUMN, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE,
                                          n_rows, s, n_rows, n_blocks, &one_cu, cs_mat_descr,
                                          (T_cu *)j2_Values_d.ptr, j2_RowPtr_d.ptr, j2_ColIdx_d.ptr, b_len,
                                          (T_cu *)atild[m2-atild_begin].ptr, s, &zero_cu, (T_cu *)t_d.ptr, n));
                            TRY(gen_geam(bl_handle, CUBLAS_OP_T, CUBLAS_OP_N, s, n, &one_cu, (T_cu *)t_d.ptr, n,
                                         &zero_cu, (T_cu *)temp_d.ptr, s, (T_cu *)temp_d.ptr, s));
                            for (int m1 = alpha_begin; m1 <= alpha_end; m1++) {
                                TRY(gen_dotc(bl_handle, sz, (T_cu *)alpha[m1-alpha_begin].ptr, 1, (T_cu *)temp_d.ptr, 1, &result_temp));
                                //std::cout << "k=" << k << ",m1=" << m1 << ",m2=" << m2 << ",res=" << cuda_cmplx(result_temp)
                                //<< std::endl;
                                mu[m1][m2] += cuda_cmplx(result_temp);
                            }
                        }
                        atild_begin = atild_end + 1;
                        atild_end   = std::min(M-1, atild_end + a_chunk_ncols);
                    }
                    alpha_begin = alpha_end + 1;
                    alpha_end   = std::min(M-1, alpha_end + a_chunk_ncols);
                }
            }
            temp_d.deallocate();
            Rchunk_d.deallocate();
            atild0_d.deallocate();
            j1_RowPtr_d.deallocate();
            j1_ColIdx_d.deallocate();
            j1_Values_d.deallocate();
            j2_RowPtr_d.deallocate();
            j2_ColIdx_d.deallocate();
            j2_Values_d.deallocate();
            for (int i = 0; i < alpha.size(); i++) alpha[i].deallocate();
            for (int i = 0; i < atild.size(); i++) atild[i].deallocate();
            return mu;
        }
        
        Vec<Vec<cx_double>> moments2_v2(int M, SpMatBsr<T> const& j1op, SpMatBsr<T> const& j2op, int a_chunk_ncols) {
            std::cerr << "moments2_v2 unimplemented on GPU\n";
            std::exit(EXIT_FAILURE);
        }
        
        void stoch_matrix(Vec<double> const& c, SpMatBsr<T>& D) {
            TRY(cudaSetDevice(device));
            assert(D.n_rows == n_rows && D.n_cols == n_rows && D.b_len == b_len);
            assert(b_len*n_rows == this->R.n_rows && b_len*n_rows >= this->R.n_cols);
            
            a_d[0].from_device(R_d);            // a0 = T_0[H] R = R
            cgemm_H(1, R_d, 0, a_d[1]);         // a1 = T_1[H] R = H R
            
            // xi = c0 a0 + c1 a1
            xi_d.memset(0);
            T_cu scal0 = cuda_cast(T(c[0]));
            T_cu scal1 = cuda_cast(T(c[1]));
            TRY(gen_axpy(bl_handle, this->R.size(), &scal0, (T_cu *)a_d[0].ptr, 1, (T_cu *)xi_d.ptr, 1));
            TRY(gen_axpy(bl_handle, this->R.size(), &scal1, (T_cu *)a_d[1].ptr, 1, (T_cu *)xi_d.ptr, 1));
            
            int M = c.size();
            for (int m = 2; m < M; m++) {
                a_d[2].from_device(a_d[0]);
                cgemm_H(2, a_d[1], -1, a_d[2]); // a2 = T_m[H] R = 2 H a1 - a0
                
                // xi += cm a2
                T_cu scal1 = cuda_cast(T(c[m]));
                TRY(gen_axpy(bl_handle, this->R.size(), &scal1, (T_cu *)a_d[2].ptr, 1, (T_cu *)xi_d.ptr, 1));
                
                auto temp = a_d[0];
                a_d[0] = a_d[1];
                a_d[1] = a_d[2];
                a_d[2] = temp;
            }
            
            // D_ij = (1/2) [R_ik conj(\xi_jk) + \xi_ik conj(R_jk)]
            DRowIndex_d.from_host(D.row_idx.size(), D.row_idx.data());
            DColIndex_d.from_host(D.col_idx.size(), D.col_idx.data());
            DVal_d.resize(D.val.size());
            DVal_d.memset(0);
            outer_product(n_rows, b_len, this->R.n_cols, T_re(0.5), (T_cu *)R_d.ptr, (T_cu *)xi_d.ptr,
                          D.n_blocks(), DRowIndex_d.ptr, DColIndex_d.ptr, (T_cu *)DVal_d.ptr);
            outer_product(n_rows, b_len, this->R.n_cols, T_re(0.5), (T_cu *)xi_d.ptr, (T_cu *)R_d.ptr,
                          D.n_blocks(), DRowIndex_d.ptr, DColIndex_d.ptr, (T_cu *)DVal_d.ptr);
            DVal_d.to_host(D.val.data());
            
            a_d[0].memset(0);
            a_d[1].memset(0);
        }
        
        void autodiff_matrix(Vec<double> const& c, SpMatBsr<T>& D) {
            TRY(cudaSetDevice(device));
            assert(D.n_rows == n_rows && D.n_cols == n_rows && D.b_len == b_len);
            assert(b_len*n_rows == this->R.n_rows && b_len*n_rows >= this->R.n_cols);
            
            int M = c.size();
            double diag = c[1];
            for (int m = 1; m < M/2; m++) {
                diag -= c[2*m+1];
            }
            
            D.zeros();
            DRowIndex_d.from_host(D.row_idx.size(), D.row_idx.data());
            DColIndex_d.from_host(D.col_idx.size(), D.col_idx.data());
            DVal_d.from_host(D.val.size(), D.val.data());
            
            // D += diag R R^\dagger
            outer_product(n_rows, b_len, this->R.n_cols, T_re(diag), (T_cu *)R_d.ptr, (T_cu *)R_d.ptr,
                          D.n_blocks(), DRowIndex_d.ptr, DColIndex_d.ptr, (T_cu *)DVal_d.ptr);
            TRY(cudaPeekAtLastError());
            TRY(cudaDeviceSynchronize());
            
            // b1 = \beta_{M/2+1}, b0 = \beta_{M/2}
            b_d[1].memset(0);
            if (M <= 2) {
                b_d[0].memset(0);
            }
            else {
                // b0 = 2 c[M-1] a1
                b_d[0].from_device(a_d[0]);
                T_re scal0 = 2*c[M-1];
                TRY(gen_scal(bl_handle, b_d[0].size, &scal0, (T_cu *)b_d[0].ptr, 1));
            }
            
            for (int m = M/2-1; m >= 1; m--) {
                // a0 = \alpha_m, b0 = \beta_{m+1}
                
                // D += 2 \alpha_m \beta_{m+1}^\dagger
                outer_product(n_rows, b_len, this->R.n_cols, T_re(2), (T_cu *)a_d[0].ptr, (T_cu *)b_d[0].ptr,
                              D.n_blocks(), DRowIndex_d.ptr, DColIndex_d.ptr, (T_cu *)DVal_d.ptr);
                TRY(cudaPeekAtLastError());
                TRY(cudaDeviceSynchronize());
                
                // (a0, a1, a2) <= (2 H a1 - a2, a0, a1)
                auto temp = a_d[2];
                a_d[2] = a_d[1];
                a_d[1] = a_d[0];
                a_d[0] = temp;
                a_d[0].from_device(a_d[2]);
                cgemm_H(2, a_d[1], -1, a_d[0]); // a0 = 2 H a1 - a2
                
                // (b0, b1, b2) <= (2 H b1 - b2, a0, a1)
                temp = b_d[2];
                b_d[2] = b_d[1];
                b_d[1] = b_d[0];
                b_d[0] = temp;
                b_d[0].from_device(b_d[2]);
                cgemm_H(2, b_d[1], -1, b_d[0]);
                
                // b0 += 4*c[2*m]*a1
                T_cu scal0 = cuda_cast(T(4*c[2*m]));
                TRY(gen_axpy(bl_handle, b_d[0].size, &scal0, (T_cu *)a_d[1].ptr, 1, (T_cu *)b_d[0].ptr, 1));
                // b0 += 2*c[2*m+1]*a2
                T_cu scal1 = cuda_cast(T(2*c[2*m+1]));
                TRY(gen_axpy(bl_handle, b_d[0].size, &scal1, (T_cu *)a_d[2].ptr, 1, (T_cu *)b_d[0].ptr, 1));
                // b0 += 2*c[2*m-1]*a0
                if (m > 1) {
                    T_cu scal2 = cuda_cast(T(2*c[2*m-1]));
                    TRY(gen_axpy(bl_handle, b_d[0].size, &scal2, (T_cu *)a_d[0].ptr, 1, (T_cu *)b_d[0].ptr, 1));
                }
            }
            
            // D += \alpha_0 \beta_1^\dagger
            outer_product(n_rows, b_len, this->R.n_cols, T_re(1), (T_cu *)a_d[0].ptr, (T_cu *)b_d[0].ptr,
                          D.n_blocks(), DRowIndex_d.ptr, DColIndex_d.ptr, (T_cu *)DVal_d.ptr);
            TRY(cudaPeekAtLastError());
            TRY(cudaDeviceSynchronize());
            
            DVal_d.to_host(D.val.data());
            D.symmetrize();
            D.scale(1.0/es.mag());
            
            a_d[0].memset(0);
            a_d[1].memset(0);
        }
    };
    
    
    template <typename T>
    class Engine_Threaded: public Engine<T> {
    public:
        int n_threads;
        Vec<std::shared_ptr<Engine<T>>> workers;
        
        void run_workers(std::function<void(int)> fn) {
            Vec<std::thread> threads;
            for (int t = 0; t < n_threads; t++) {
                threads.push_back(std::thread(fn, t));
            }
            for (int t = 0; t < n_threads; t++) {
                threads[t].join();
            }
        }
        
        Engine_Threaded(Vec<std::shared_ptr<Engine<T>>> workers):
            n_threads(workers.size()),
            workers(workers) {}
        
        EnergyScale energy_scale(SpMatBsr<T> const& H, double extend, int iters) {
            return workers[0]->energy_scale(H, extend, iters);
        }
        
        void set_R_identity(int n, int j_start, int j_end) {
            run_workers([&](int t) {
                int sz = j_end - j_start;
                int j1 = j_start + t * sz / n_threads;
                int j2 = j_start + (t + 1) * sz / n_threads;
                workers[t]->set_R_identity(n, j1, j2);
            });
        }
        
        void set_R_uncorrelated(int n, int s, RNG& rng, int j_start, int j_end) {
            run_workers([&](int t) {
                RNG rng0 = rng;
                int sz = j_end - j_start;
                int j1 = j_start + t * sz / n_threads;
                int j2 = j_start + (t + 1) * sz / n_threads;
                rng0.discard(j1 - j_start);
                workers[t]->set_R_uncorrelated(n, s, rng0, j1, j2);
            });
            rng.discard(j_end - j_start);
        }
        
        void set_R_correlated(Vec<int> const& groups, RNG& rng, int j_start, int j_end) {
            run_workers([&](int t) {
                RNG rng0 = rng;
                int sz = j_end - j_start;
                int j1 = j_start + t * sz / n_threads;
                int j2 = j_start + (t + 1) * sz / n_threads;
                rng0.discard(j1 - j_start);
                workers[t]->set_R_correlated(groups, rng0, j1, j2);
            });
            rng.discard(j_end - j_start);
        }
        
        void set_H(SpMatBsr<T> const& H, EnergyScale const& es) {
            run_workers([&](int t) {
                workers[t]->set_H(H, es);
            });
        }
        
        Vec<double> moments(int M) {
            Vec<Vec<double>> mus(n_threads);
            run_workers([&](int t) {
                mus[t] = workers[t]->moments(M);
            });
            for (int t = 1; t < n_threads; t++) {
                for (int m = 0; m < M; m++) {
                    mus[0][m] += mus[t][m];
                }
            }
            return mus[0];
        }
        
        Vec<Vec<cx_double>> moments2_v1(int M, SpMatBsr<T> const& j1op, SpMatBsr<T> const& j2op,
                                        int a_chunk_ncols=-1, int R_chunk_ncols=-1) {
            Vec<Vec<Vec<cx_double>>> mus(n_threads);
            run_workers([&](int t) {
                mus[t] = workers[t]->moments2_v1(M, j1op, j2op, a_chunk_ncols, R_chunk_ncols);
            });
            for (int t = 1; t < n_threads; t++) {
                for (int m1 = 0; m1 < M; m1++) {
                    for (int m2 = 0; m2 < M; m2++) {
                        mus[0][m1][m2] += mus[t][m1][m2];
                    }
                }
            }
            return mus[0];
        }
        
        Vec<Vec<cx_double>> moments2_v2(int M, SpMatBsr<T> const& j1op, SpMatBsr<T> const& j2op, int a_chunk_ncols=-1) {
            if (n_threads > 1) {
                std::cerr << "Threaded moments2_v2 not yet implemented!\n";
            }
            return workers[0]->moments2_v2(M, j1op, j2op, a_chunk_ncols);
        }
        
        void stoch_matrix(Vec<double> const& c, SpMatBsr<T>& D) {
            Vec<SpMatBsr<T>> Ds(n_threads, D);
            run_workers([&](int t) {
                workers[t]->stoch_matrix(c, Ds[t]);
            });
            D.zeros();
            for (auto &Dt : Ds) {
                for (int i = 0; i < D.val.size(); i++) {
                    D.val[i] += Dt.val[i];
                }
            }
        }
        
        void autodiff_matrix(Vec<double> const& c, SpMatBsr<T>& D) {
            Vec<SpMatBsr<T>> Ds(n_threads, D);
            run_workers([&](int t) {
                workers[t]->autodiff_matrix(c, Ds[t]);
            });
            D.zeros();
            for (auto &Dt : Ds) {
                for (int i = 0; i < D.val.size(); i++) {
                    D.val[i] += Dt.val[i];
                }
            }
        }
    };
    
    static bool printed_devices = false;
    template <typename T>
    std::shared_ptr<Engine<T>> mk_engine_cuSPARSE(Vec<int> devices) {
        std::stringstream msg;
        Vec<std::shared_ptr<Engine<T>>> workers;
        std::shared_ptr<Engine<T>> ret = nullptr;
        int count;
        int err = cudaGetDeviceCount(&count);
        switch (err) {
            case cudaSuccess:
                if (devices.size() == 0) {
                    for (int i = 0; i < count; i++) devices.push_back(i);
                }
                for (int d : devices) {
                    if (d < 0 || count < d) {
                        std::cerr << "Invalid device #" << d << " (" << count << " available)!\n";
                        std::exit(EXIT_FAILURE);
                    }
                    if (!printed_devices) {
                        cudaDeviceProp prop;
                        cudaGetDeviceProperties(&prop, d);
                        msg << "Device #" << d << " (of " << count << ");  "
                            << prop.name << ";  "
                            << prop.totalGlobalMem/(1024.*1024.*1024.) << " GB;  "
                            << 2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6 << " GB/s\n";
                    }
                    workers.push_back(std::make_shared<Engine_cuSPARSE<T>>(d));
                }
                ret = std::make_shared<Engine_Threaded<T>>(workers);
                break;
            case cudaErrorNoDevice:
                msg << "No CUDA devices are available!\n";
                break;
            case cudaErrorInsufficientDriver:
                msg << "Insufficient CUDA driver!\n";
                break;
            default:
                msg << "Unknown CUDA error " << err << "!\n";
                break;
        }
        if (!printed_devices) {
            std::cout << msg.str();
            printed_devices = true;
        }
        return ret;
    }
    
    template std::shared_ptr<Engine<float>> mk_engine_cuSPARSE(Vec<int> devices);
    template std::shared_ptr<Engine<double>> mk_engine_cuSPARSE(Vec<int> devices);
    template std::shared_ptr<Engine<cx_float>> mk_engine_cuSPARSE(Vec<int> devices);
    template std::shared_ptr<Engine<cx_double>> mk_engine_cuSPARSE(Vec<int> devices);
}

#endif // WITH_CUDA
