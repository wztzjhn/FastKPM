//
//  engine_cusparse.cpp
//  tibidy
//
//  Created by Kipton Barros on 7/25/14.
//
//


#include "fastkpm.h"

#ifndef WITH_CUDA

namespace fkpm {
    template <typename T>
    std::shared_ptr<Engine<T>> mk_engine_cuSPARSE(int device) {
        return nullptr;
    }
    template std::shared_ptr<Engine<double>> mk_engine_cuSPARSE(int device);
    template std::shared_ptr<Engine<cx_double>> mk_engine_cuSPARSE(int device);
}

#else

#include <cstdlib>
#include <cassert>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <cublas.h>


#define TRY(x) testCudaError((x), __FILE__, __LINE__, #x)

namespace fkpm {
    template <typename T>
    const char *genericCudaErrorString(T stat);
    template <>
    const char *genericCudaErrorString(cudaError_t stat) { return cudaGetErrorString(stat); }
    template <>
    const char *genericCudaErrorString(cublasStatus_t stat) {
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
    template <>
    const char *genericCudaErrorString(cusparseStatus_t stat) {
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
    void testCudaError(T stat, char const* file, int line, char const* code) {
        if (stat) {
            std::cerr << file << ":" << line <<  ", " << code << ", Error: " << genericCudaErrorString(stat) << std::endl;
            std::abort();
        }
    }
    
    inline float           cuda_type(float  x)    { abort(); };
    inline double          cuda_type(double x)    { abort(); };
    inline cuFloatComplex  cuda_type(cx_float x)  { abort(); };
    inline cuDoubleComplex cuda_type(cx_double x) { abort(); };
    
    template <typename T>
    inline decltype(cuda_type(T(0))) cast_cuda_val(T);
    template <> float           cast_cuda_val(float x)     { return x; }
    template <> double          cast_cuda_val(double x)    { return x; }
    template <> cuFloatComplex  cast_cuda_val(cx_float x)  { return make_cuFloatComplex(x.real(), x.imag()); }
    template <> cuDoubleComplex cast_cuda_val(cx_double x) { return make_cuDoubleComplex(x.real(), x.imag()); };
    
    template <typename T>
    cusparseStatus_t gen_csrmm(cusparseHandle_t handle, cusparseOperation_t transA, int m, int n, int k, int nnz, const T *alpha,
                               const cusparseMatDescr_t descrA, const T  *csrValA, const int *csrRowPtrA, const int *csrColIndA,
                               const T *B, int ldb, const T *beta, T *C, int ldc);
    template <> // float
    cusparseStatus_t gen_csrmm(cusparseHandle_t handle, cusparseOperation_t transA, int m, int n, int k, int nnz, const float *alpha,
                               const cusparseMatDescr_t descrA, const float  *csrValA, const int *csrRowPtrA, const int *csrColIndA,
                               const float *B, int ldb, const float *beta, float *C, int ldc) {
        return cusparseScsrmm(handle, transA, m, n, k, nnz, alpha, descrA, csrValA, csrRowPtrA, csrColIndA, B, ldb, beta, C, ldc);
    }
    template <> // double
    cusparseStatus_t gen_csrmm(cusparseHandle_t handle, cusparseOperation_t transA, int m, int n, int k, int nnz, const double *alpha,
                               const cusparseMatDescr_t descrA, const double  *csrValA, const int *csrRowPtrA, const int *csrColIndA,
                               const double *B, int ldb, const double *beta, double *C, int ldc) {
        return cusparseDcsrmm(handle, transA, m, n, k, nnz, alpha, descrA, csrValA, csrRowPtrA, csrColIndA, B, ldb, beta, C, ldc);
    }
    template <> // cx_float
    cusparseStatus_t gen_csrmm(cusparseHandle_t handle, cusparseOperation_t transA, int m, int n, int k, int nnz, const cuFloatComplex *alpha,
                               const cusparseMatDescr_t descrA, const cuFloatComplex  *csrValA, const int *csrRowPtrA, const int *csrColIndA,
                               const cuFloatComplex *B, int ldb, const cuFloatComplex *beta, cuFloatComplex *C, int ldc) {
        return cusparseCcsrmm(handle, transA, m, n, k, nnz, alpha, descrA, csrValA, csrRowPtrA, csrColIndA, B, ldb, beta, C, ldc);
    }
    template <> // cx_double
    cusparseStatus_t gen_csrmm(cusparseHandle_t handle, cusparseOperation_t transA, int m, int n, int k, int nnz, const cuDoubleComplex *alpha,
                               const cusparseMatDescr_t descrA, const cuDoubleComplex  *csrValA, const int *csrRowPtrA, const int *csrColIndA,
                               const cuDoubleComplex *B, int ldb, const cuDoubleComplex *beta, cuDoubleComplex *C, int ldc) {
        return cusparseZcsrmm(handle, transA, m, n, k, nnz, alpha, descrA, csrValA, csrRowPtrA, csrColIndA, B, ldb, beta, C, ldc);
    }
    
    template <typename T, typename T_re>
    void outer_product(int n_rows, int n_cols, T_re alpha, T *A, T *B,
                       int D_nnz, int *D_row_idx, int *D_col_idx, T *D_val);
    
    // Caution: cudaSetDevice() must be called before any operations.
    template <typename T>
    class CuVec {
    public:
        int size = 0;
        int capacity = 0;
        T *ptr = 0; // TODO: T* and remove casts
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
        typedef decltype(cuda_type(T(0))) T_cu;
        int device = 0;
        EnergyScale es;
        int Hs_n_rows = 0;
        int Hs_n_nonzero = 0;
        T_re Hs_trace = 0;
        Vec<T> Hs_val;
        
        cusparseHandle_t cs_handle;
        cusparseMatDescr_t cs_mat_descr;
        
        CuVec<T> R_d, xi_d;
        CuVec<T> a_d[3];
        CuVec<T> b_d[3];
        
        CuVec<int> HColIndex_d, HRowPtr_d;
        CuVec<T> HVal_d;
        
        CuVec<int> DRowIndex_d, DColIndex_d;
        CuVec<T> DVal_d;
        
        Engine_cuSPARSE(int device) {
            this->device = device;
            TRY(cudaSetDevice(device));
            
            TRY(cusparseCreate(&cs_handle));
            TRY(cusparseCreateMatDescr(&cs_mat_descr));
            // TODO: CUSPARSE_MATRIX_TYPE_HERMITIAN
            TRY(cusparseSetMatType(cs_mat_descr, CUSPARSE_MATRIX_TYPE_GENERAL));
            TRY(cusparseSetMatIndexBase(cs_mat_descr, CUSPARSE_INDEX_BASE_ZERO));
        }
        
        ~Engine_cuSPARSE() {
            TRY(cudaSetDevice(device));

            R_d.deallocate();
            xi_d.deallocate();
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
        }
        
        void transfer_R() {
            TRY(cudaSetDevice(device));
            
            int sz = this->R.size();
            this->R_d.from_host(sz, this->R.memptr());
            xi_d.resize(sz);
            for (int i = 0; i < 3; i++) {
                a_d[i].resize(sz);
                b_d[i].resize(sz);
            }
        }
        
        void set_H(SpMatCsr<T> const& H, EnergyScale const& es) {
            TRY(cudaSetDevice(device));
            assert(H.n_rows == H.n_cols);
            
            this->es = es;
            Hs_n_rows = H.n_rows;
            Hs_n_nonzero = H.size();
            Hs_val.resize(Hs_n_nonzero);
            Hs_trace = 0;
            int diag_cnt = 0;
            T_re es_mag_inv = 1.0 / es.mag();
            T_re es_shift = es.avg() / es.mag();
            for (int k = 0; k < H.size(); k++) {
                Hs_val[k] = H.val[k] * es_mag_inv;
                if (H.row_idx[k] == H.col_idx[k]) {
                    Hs_val[k] -= es_shift;
                    Hs_trace += std::real(Hs_val[k]);
                    diag_cnt++;
                }
            }
            assert(diag_cnt == H.n_rows);
            HRowPtr_d.from_host(H.row_ptr.size(), H.row_ptr.data());
            HColIndex_d.from_host(H.col_idx.size(), H.col_idx.data());
            HVal_d.from_host(Hs_val.size(), Hs_val.data());
        }
        
        // C = alpha H B + beta C
        void cgemm_H(T alpha, CuVec<T> B_d, T beta, CuVec<T> C_d) {
            int n = this->R.n_rows;
            int s = this->R.n_cols;
            auto alpha_f = cast_cuda_val(alpha);
            auto beta_f  = cast_cuda_val(beta);
            TRY(gen_csrmm(cs_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                          n, s, n, Hs_n_nonzero, // (H rows, B cols, H cols, H nnz)
                          &alpha_f,
                          cs_mat_descr, (T_cu *)HVal_d.ptr, HRowPtr_d.ptr, HColIndex_d.ptr, // H matrix
                          (T_cu *)B_d.ptr, n, // (B, B rows)
                          &beta_f,
                          (T_cu *)C_d.ptr, n)); // (C, C rows)
        }
        
        Vec<double> moments(int M) {
            TRY(cudaSetDevice(device));
            assert(Hs_n_rows == this->R.n_rows);
            assert(M % 2 == 0);
            
            Vec<double> mu(M);
            mu[0] = Hs_n_rows;
            mu[1] = Hs_trace;
            
            a_d[0].from_device(R_d);            // a0 = \alpha_0 = R
            cgemm_H(1, R_d, 0, a_d[1]);         // a1 = \alpha_1 = H R
            
            for (int m = 1; m < M/2; m++) {
                a_d[2].from_device(a_d[0]);
                cgemm_H(2, a_d[1], -1, a_d[2]); // a2 = 2 H a1 - a0
                
                auto temp = a_d[0];
                a_d[0] = a_d[1];
                a_d[1] = a_d[2];
                a_d[2] = temp;
                
                // 2 \alpha_m^\dagger \alpha_m - mu0
                mu[2*m]   = 2 * cublasCdotc(this->R.size(), (cuComplex *)a_d[0].ptr, 1, (cuComplex *)a_d[0].ptr, 1).x - mu[0];
                // 2 \alpha_{m+1}^\dagger \alpha_m - mu1
                mu[2*m+1] = 2 * cublasCdotc(this->R.size(), (cuComplex *)a_d[1].ptr, 1, (cuComplex *)a_d[0].ptr, 1).x - mu[1];
            }
            
            return mu;
        }
        
        void stoch_matrix(Vec<double> const& c, SpMatCsr<T>& D) {
            TRY(cudaSetDevice(device));
            assert(Hs_n_rows == this->R.n_rows);
            
            a_d[0].from_device(R_d);            // a0 = T_0[H] R = R
            cgemm_H(1, R_d, 0, a_d[1]);         // a1 = T_1[H] R = H R
            
            // xi = c0 a0 + c1 a1
            xi_d.memset(0);
            cublasCaxpy(this->R.size(), make_cuComplex(c[0], 0), (cuComplex *)a_d[0].ptr, 1, (cuComplex *)xi_d.ptr, 1);
            cublasCaxpy(this->R.size(), make_cuComplex(c[1], 0), (cuComplex *)a_d[1].ptr, 1, (cuComplex *)xi_d.ptr, 1);
            
            int M = c.size();
            for (int m = 2; m < M; m++) {
                a_d[2].from_device(a_d[0]);
                cgemm_H(2, a_d[1], -1, a_d[2]); // a2 = T_m[H] R = 2 H a1 - a0
                
                // xi += cm a2
                cublasCaxpy(this->R.size(), make_cuComplex(c[m], 0), (cuComplex *)a_d[2].ptr, 1, (cuComplex *)xi_d.ptr, 1);
                
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
            outer_product(this->R.n_rows, this->R.n_cols, T_re(0.5), (T_cu *)this->R_d.ptr, (T_cu *)this->xi_d.ptr,
                          D.size(), DRowIndex_d.ptr, DColIndex_d.ptr, (T_cu *)DVal_d.ptr);
            outer_product(this->R.n_rows, this->R.n_cols, T_re(0.5), (T_cu *)this->xi_d.ptr, (T_cu *)this->R_d.ptr,
                          D.size(), DRowIndex_d.ptr, DColIndex_d.ptr, (T_cu *)DVal_d.ptr);
            DVal_d.to_host(D.val.data());
            
            a_d[0].memset(0);
            a_d[1].memset(0);
        }
        
        void autodiff_matrix(Vec<double> const& c, SpMatCsr<T>& D) {
            int M = c.size();
            int n = this->R.n_rows;
            int s = this->R.n_cols;
            
            double diag = c[1];
            for (int m = 1; m < M/2; m++) {
                diag -= c[2*m+1];
            }
            for (int k = 0; k < D.size(); k++) {
                D.val[k] = (D.row_idx[k] == D.col_idx[k]) ? diag : 0;
            }
            
            DRowIndex_d.from_host(D.row_idx.size(), D.row_idx.data());
            DColIndex_d.from_host(D.col_idx.size(), D.col_idx.data());
            DVal_d.from_host(D.val.size(), D.val.data());
            
            // b1 = \beta_{M/2+1}, b0 = \beta_{M/2}
            b_d[1].memset(0);
            if (M <= 2) {
                b_d[0].memset(0);
            }
            else {
                // b0 = 2 c[M-1] a1
                b_d[0].from_device(a_d[1]);
                cublasCscal(b_d[0].size, make_cuComplex(2*c[M-1], 0), (cuComplex *)b_d[0].ptr, 1);
            }
            
            for (int m = M/2-1; m >= 1; m--) {
                // a0 = \alpha_m, b0 = \beta_{m+1}
                
                // D += 2 \alpha_m \beta_{m+1}^\dagger
                outer_product(n, s, T_re(2), (T_cu *)a_d[0].ptr, (T_cu *)b_d[0].ptr,
                              D.size(), DRowIndex_d.ptr, DColIndex_d.ptr, (T_cu *)DVal_d.ptr);
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
                cublasCaxpy(b_d[0].size,
                            make_cuComplex(4*c[2*m], 0),
                            (cuComplex *)a_d[1].ptr, 1,
                            (cuComplex *)b_d[0].ptr, 1);
                // b0 += 2*c[2*m+1]*a2
                cublasCaxpy(b_d[0].size,
                            make_cuComplex(2*c[2*m+1], 0),
                            (cuComplex *)a_d[2].ptr, 1,
                            (cuComplex *)b_d[0].ptr, 1);
                // b0 += 2*c[2*m-1]*a0
                if (m > 1) {
                    cublasCaxpy(b_d[0].size,
                                make_cuComplex(2*c[2*m-1], 0),
                                (cuComplex *)a_d[0].ptr, 1,
                                (cuComplex *)b_d[0].ptr, 1);
                }
            }
            
            // D += \alpha_0 \beta_1^\dagger
            outer_product(n, s, T_re(1), (T_cu *)a_d[0].ptr, (T_cu *)b_d[0].ptr,
                          D.size(), DRowIndex_d.ptr, DColIndex_d.ptr, (T_cu *)DVal_d.ptr);
            TRY(cudaPeekAtLastError());
            TRY(cudaDeviceSynchronize());
            
            DVal_d.to_host(D.val.data());
            D.symmetrize();
            for (auto& v: D.val) {
                v /= this->es.mag();
            }
            
            a_d[0].memset(0);
            a_d[1].memset(0);
        }
    };
    
    template <typename T>
    std::shared_ptr<Engine<T>> mk_engine_cuSPARSE(int device) {
        std::stringstream msg;
        static Vec<bool> printed_msg(16, false);
        assert(device >= 0 && device < printed_msg.size());
        std::shared_ptr<Engine<T>> ret = nullptr;
        int count;
        int err = cudaGetDeviceCount(&count);
        switch (err) {
            case cudaSuccess:
                if (device < count) {
                    cudaDeviceProp prop;
                    cudaGetDeviceProperties(&prop, device);
                    msg << "Using device " << device << "\n";
                    msg << "  Device name:           " << prop.name << "\n";
                    msg << "  Total global memory:   " << prop.totalGlobalMem/(1024.*1024.*1024.) << " (GB)\n";
                    msg << "  Peak memory bandwidth: " << 2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6 << " (GB/s)\n\n";
                    ret = std::make_shared<Engine_cuSPARSE<T>>(device);
                }
                else {
                    msg << "Device #" << device << " exceeds availability (" << count << " devices)!\n";
                }
                break;
            case cudaErrorNoDevice:
                msg << "No CUDA device is available!\n";
                break;
            case cudaErrorInsufficientDriver:
                msg << "Insufficient CUDA driver!\n";
                break;
            default:
                msg << "Unknown CUDA error " << err << "!\n";
                break;
        }
        if (!printed_msg[device]) {
            std::cout << msg.str();
            printed_msg[device] = true;
        }
        return ret;
    }
    
    template <>
    std::shared_ptr<Engine<double>> mk_engine_cuSPARSE(int device) {
        std::cerr << "cuSPARSE engine not yet implemented for type `double`!\n";
        return nullptr; // not yet implemented
    }
    template <>
    std::shared_ptr<Engine<float>> mk_engine_cuSPARSE(int device) {
        std::cerr << "cuSPARSE engine not yet implemented for type `double`!\n";
        return nullptr; // not yet implemented
    }

// #define YES
#ifdef WITH_DOUBLE
    template std::shared_ptr<Engine<cx_double>> mk_engine_cuSPARSE(int device);
#else
    template <>
    std::shared_ptr<Engine<cx_double>> mk_engine_cuSPARSE(int device) {
        std::cerr << "cuSPARSE engine not yet implemented for type `double`!\n";
        return nullptr; // not yet implemented
    }
#endif
    
    template std::shared_ptr<Engine<cx_float>> mk_engine_cuSPARSE(int device);
}

#endif // WITH_CUDA
