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
    
    void outer_product(int n_rows, int n_cols, float alpha, cuFloatComplex *A, cuFloatComplex *B,
                       int D_nnz, int *D_row_idx, int *D_col_idx, cuFloatComplex *D_val);
    
    template <typename S, typename T>
    void convert_array(S const* src, int n, T* dst) {
        for (int i = 0; i < n; i++)
            dst[i] = (T)src[i];
    }
    
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
    class Engine_cuSPARSE;
    
    template <>
    class Engine_cuSPARSE<cx_double>: public Engine<cx_double> {
    public:
        int device = 0;
        int n_nonzero = 0;
        double Hs_trace = 0;
        
        cusparseHandle_t cs_handle;
        cusparseMatDescr_t cs_mat_descr;
        
        CuVec<cx_float> R_d, xi_d;
        CuVec<cx_float> a_d[3];
        CuVec<cx_float> b_d[3];
        
        CuVec<int> HColIndex_d, HRowPtr_d;
        CuVec<cx_float> HVal_d;
        
        CuVec<int> DRowIndex_d, DColIndex_d;
        CuVec<cx_float> DVal_d;

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
        
        Vec<cx_float> cx_float_store;
        void device_to_host_cx(CuVec<cx_float> const& src, cx_double *dst) {
            int size = src.size;
            cx_float_store.resize(size);
            src.to_host(cx_float_store.data());
            convert_array(cx_float_store.data(), size, dst);
        }
        void host_to_device_cx(cx_double const* src, int size, CuVec<cx_float>& dst) {
            cx_float_store.resize(size);
            convert_array(src, size, cx_float_store.data());
            dst.from_host(size, cx_float_store.data());
        }
        
        void transfer_R() {
            TRY(cudaSetDevice(device));
            
            int sz = R.size();
            host_to_device_cx(R.memptr(), sz, R_d);
            xi_d.resize(sz);
            for (int i = 0; i < 3; i++) {
                a_d[i].resize(sz);
                b_d[i].resize(sz);
            }
        }
        
        void transfer_H() {
            TRY(cudaSetDevice(device));
            
            n_nonzero = Hs.size();
            Hs_trace = 0;
            for (int i = 0; i < Hs.n_rows; i++) {
                Hs_trace += std::real(Hs(i, i));
            }
            
            HRowPtr_d.from_host(Hs.row_ptr.size(), Hs.row_ptr.data());
            HColIndex_d.from_host(Hs.col_idx.size(), Hs.col_idx.data());
            host_to_device_cx(Hs.val.data(), n_nonzero, HVal_d);
        }
        
        // C = alpha H B + beta C
        void cgemm_H(cx_double alpha, CuVec<cx_float> B_d, cx_double beta, CuVec<cx_float> C_d) {
            int n = R.n_rows;
            int s = R.n_cols;
            auto alpha_f = make_cuComplex(alpha.real(), alpha.imag());
            auto beta_f  = make_cuComplex(beta.real(),  beta.imag());
            TRY(cusparseCcsrmm(cs_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                               n, s, n, n_nonzero, // (H rows, B cols, H cols, H nnz)
                               &alpha_f,
                               cs_mat_descr, (cuComplex *)HVal_d.ptr, HRowPtr_d.ptr, HColIndex_d.ptr, // H matrix
                               (cuComplex *)B_d.ptr, n, // (B, B rows)
                               &beta_f,
                               (cuComplex *)C_d.ptr, n)); // (C, C rows)
        }
        
        
        Vec<double> moments(int M) {
            TRY(cudaSetDevice(device));
            
            assert(Hs.n_rows == R.n_rows && Hs.n_cols == R.n_rows);
            
            Vec<double> mu(M);
            mu[0] = Hs.n_rows;  // Tr[T_0[H]] = Tr[1]
            mu[1] = Hs_trace;   // Tr[T_1[H]] = Tr[H]
            
            a_d[0].from_device(R_d);            // a0 = T_0[H] R = R
            cgemm_H(1, R_d, 0, a_d[1]);         // a1 = T_1[H] R = H R
            
            for (int m = 2; m < M; m++) {
                a_d[2].from_device(a_d[0]);
                cgemm_H(2, a_d[1], -1, a_d[2]); // a2 = T_m[H] R = 2 H a1 - a0
                
                mu[m] = cublasCdotc(R.size(), (cuComplex *)R_d.ptr, 1, (cuComplex *)a_d[2].ptr, 1).x; // R^\dag \dot alpha_2
                
                auto temp = a_d[0];
                a_d[0] = a_d[1];
                a_d[1] = a_d[2];
                a_d[2] = temp;
            }
            return mu;
        }
        
        void stoch_matrix(Vec<double> const& c, SpMatCsr<cx_double>& D) {
            TRY(cudaSetDevice(device));
            
            assert(Hs.n_rows == R.n_rows && Hs.n_cols == R.n_rows);
            
            a_d[0].from_device(R_d);            // a0 = T_0[H] R = R
            cgemm_H(1, R_d, 0, a_d[1]);         // a1 = T_1[H] R = H R
            
            // xi = c0 a0 + c1 a1
            xi_d.memset(0);
            cublasCaxpy(R.size(), make_cuComplex(c[0], 0), (cuComplex *)a_d[0].ptr, 1, (cuComplex *)xi_d.ptr, 1);
            cublasCaxpy(R.size(), make_cuComplex(c[1], 0), (cuComplex *)a_d[1].ptr, 1, (cuComplex *)xi_d.ptr, 1);
            
            int M = c.size();
            for (int m = 2; m < M; m++) {
                a_d[2].from_device(a_d[0]);
                cgemm_H(2, a_d[1], -1, a_d[2]); // a2 = T_m[H] R = 2 H a1 - a0
                
                // xi += cm a2
                cublasCaxpy(R.size(), make_cuComplex(c[m], 0), (cuComplex *)a_d[2].ptr, 1, (cuComplex *)xi_d.ptr, 1);
                
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
            outer_product(R.n_rows, R.n_cols, 0.5, (cuFloatComplex *)R_d.ptr, (cuFloatComplex *)xi_d.ptr,
                          D.size(), DRowIndex_d.ptr, DColIndex_d.ptr, (cuFloatComplex *)DVal_d.ptr);
            outer_product(R.n_rows, R.n_cols, 0.5, (cuFloatComplex *)xi_d.ptr, (cuFloatComplex *)R_d.ptr,
                          D.size(), DRowIndex_d.ptr, DColIndex_d.ptr, (cuFloatComplex *)DVal_d.ptr);
            device_to_host_cx(DVal_d, D.val.data());
        }
        
        void autodiff_matrix(Vec<double> const& c, SpMatCsr<cx_double>& D) {
            // need special logic since mu[1] was calculated exactly
            for (int k = 0; k < D.size(); k++) {
                D.val[k] = (D.row_idx[k] == D.col_idx[k]) ? c[1] : 0;
            }
            auto cp = [&](int m) { return (m == 1) ? 0 : c[m]; };
            
            DRowIndex_d.from_host(D.row_idx.size(), D.row_idx.data());
            DColIndex_d.from_host(D.col_idx.size(), D.col_idx.data());
            host_to_device_cx(D.val.data(), D.val.size(), DVal_d);
            
            int M = c.size();
            b_d[1].memset(0);                             // b1 = 0
            b_d[0].from_device(R_d);                      // b0 = c(M-1) R
            cublasCscal(b_d[0].size, make_cuComplex(cp(M-1), 0), (cuComplex *)b_d[0].ptr, 1);
            
            for (int m = M-2; m >= 0; m--) {
                // a0 = alpha_{m}
                // b0 = beta_{m}
                double scale = (m == 0) ? 1 : 2;
                // D_ij += scale a0_ik conj(\b0_jk)
                outer_product(R.n_rows, R.n_cols, scale, (cuFloatComplex *)a_d[0].ptr, (cuFloatComplex *)b_d[0].ptr,
                              D.size(), DRowIndex_d.ptr, DColIndex_d.ptr, (cuFloatComplex *)DVal_d.ptr);
                TRY(cudaPeekAtLastError());
                TRY(cudaDeviceSynchronize());
                
                // (a0, a1, a2) <= (2 H a1 - a2, a0, a1)
                auto temp = a_d[2];
                a_d[2] = a_d[1];
                a_d[1] = a_d[0];
                a_d[0] = temp;
                a_d[0].from_device(a_d[2]);
                cgemm_H(2, a_d[1], -1, a_d[0]); // a0 = 2 H a1 - a2
                
                // (b0, b1, b2) <= (2 H b1 - b2 + c(m) r, a0, a1)
                temp = b_d[2];
                b_d[2] = b_d[1];
                b_d[1] = b_d[0];
                b_d[0] = temp;
                b_d[0].from_device(b_d[2]);
                cgemm_H(2, b_d[1], -1, b_d[0]); // b0 = 2 H b1 - b2
                cublasCaxpy(R.size(),           // b0 += c(m) r
                            make_cuComplex(cp(m), 0),
                            (cuComplex *)R_d.ptr, 1,
                            (cuComplex *)b_d[0].ptr, 1);
            }
            
            device_to_host_cx(DVal_d, D.val.data());
            for (cx_double& v: D.val) {
                v /= this->es.mag();
            }
        }
    };
    
    
    template <typename T>
    std::shared_ptr<Engine<T>> mk_engine_cuSPARSE(int device) {
        int count;
        int err = cudaGetDeviceCount(&count);
        switch (err) {
            case cudaSuccess:
                if (device < count) {
                    return std::make_shared<Engine_cuSPARSE<T>>(device);
                }
                else {
                    std::cerr << "Requested device " << device << " but only " << count << " are available!\n";
                    return nullptr;
                }
            case cudaErrorNoDevice:
                std::cerr << "No CUDA device is available!\n";
                return nullptr;
            case cudaErrorInsufficientDriver:
                std::cerr << "Insufficient CUDA driver!\n";
                return nullptr;
            default:
                std::cerr << "Unknown CUDA error " << err << "!\n";
                return nullptr;
        }
    }
    template <>
    std::shared_ptr<Engine<double>> mk_engine_cuSPARSE(int device) {
        std::cerr << "cuSPARSE engine not yet implemented for type `double`!\n";
        return nullptr; // not yet implemented
    }
    template std::shared_ptr<Engine<cx_double>> mk_engine_cuSPARSE(int device);
}

#endif // WITH_CUDA
