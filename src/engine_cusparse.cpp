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
    std::shared_ptr<Engine<T>> mk_engine_cuSPARSE() {
        return nullptr;
    }
    template std::shared_ptr<Engine<double>> mk_engine_cuSPARSE();
    template std::shared_ptr<Engine<cx_double>> mk_engine_cuSPARSE();
}

#else

#include <cstdlib>
#include <cassert>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <cublas.h>

#define TRY(x) \
    { int stat = (x); \
      if (stat != cudaSuccess) { \
        std::cerr << __FILE__ << ":" << __LINE__ <<  ", " << #x << ", error " << stat << std::endl; \
        std::exit(EXIT_FAILURE); \
      } \
    };

namespace fkpm {
    
    void double_to_float(double *src, int n, float *dst) {
        for (int i = 0; i < n; i++)
            dst[i] = (float)src[i];
    }
    
    void float_to_double(float *src, int n, double *dst) {
        for (int i = 0; i < n; i++)
            dst[i] = (double)src[i];
    }
    
    template <typename T>
    class Engine_cuSPARSE;
    
    template <>
    class Engine_cuSPARSE<cx_double>: public Engine<cx_double> {
    public:
        int device = 0;
        int n_nonzero = 0;
        double Hs_trace = 0;
        Vec<float> flt_store;
        
        cusparseHandle_t cs_handle;
        cusparseMatDescr_t cs_mat_descr;
        
        int R_cp=0, HRowPtr_cp=0, HColIndex_cp=0, HVal_cp=0; // allocated capacities
        int R_sz=0, HRowPtr_sz=0, HColIndex_sz=0, HVal_sz=0; // utilized size
        void *a0_d=0, *a1_d=0, *a2_d=0, *R_d=0, *xi_d=0;
        void *HColIndex_d=0, *HRowPtr_d=0, *HVal_d=0;
        
        
        Engine_cuSPARSE(int device): Engine<cx_double>() {
            this->device = device;
            TRY(cudaSetDevice(device));
            
            TRY(cusparseCreate(&cs_handle));
            TRY(cusparseCreateMatDescr(&cs_mat_descr));
            // TODO: CUSPARSE_MATRIX_TYPE_HERMITIAN
            cusparseSetMatType(cs_mat_descr, CUSPARSE_MATRIX_TYPE_GENERAL);
            cusparseSetMatIndexBase(cs_mat_descr, CUSPARSE_INDEX_BASE_ZERO);
        }
        
        ~Engine_cuSPARSE() {
            TRY(cudaSetDevice(device));
            
            TRY(cudaFree(a0_d));
            TRY(cudaFree(a1_d));
            TRY(cudaFree(a2_d));
            TRY(cudaFree(R_d));
            TRY(cudaFree(xi_d));
            
            TRY(cudaFree(HRowPtr_d));
            TRY(cudaFree(HColIndex_d));
            TRY(cudaFree(HVal_d));
        }
        
        void transfer_R() {
            TRY(cudaSetDevice(device));

            R_sz = R.size()*sizeof(cx_float);
            
            if (R_sz > R_cp) {
                R_cp = R_sz;
                
                TRY(cudaFree(&a0_d));
                TRY(cudaFree(&a1_d));
                TRY(cudaFree(&a2_d));
                TRY(cudaFree(&R_d));
                TRY(cudaFree(&xi_d));
                
                TRY(cudaMalloc(&a0_d, R_cp));
                TRY(cudaMalloc(&a1_d, R_cp));
                TRY(cudaMalloc(&a2_d, R_cp));
                TRY(cudaMalloc(&R_d, R_cp));
                TRY(cudaMalloc(&xi_d, R_cp));
            }
            
            flt_store.resize(2*R.size());
            double_to_float((double *)R.memptr(), flt_store.size(), flt_store.data());
            TRY(cudaMemcpy(R_d, flt_store.data(), R_sz, cudaMemcpyHostToDevice));
        }
        
        void transfer_H() {
            TRY(cudaSetDevice(device));

            // TODO : perform COO to CSR conversion on device!
            int n = Hs.n_rows;
            auto Hs_a = Hs.to_arma();
            n_nonzero = Hs_a.n_nonzero;
            Hs_trace = std::real(arma::trace(Hs_a));
            
            HRowPtr_sz   = (n+1)*sizeof(uword);
            HColIndex_sz = n_nonzero*sizeof(uword);
            HVal_sz      = n_nonzero*sizeof(cx_float);
            
            if (HRowPtr_sz > HRowPtr_cp || HColIndex_sz > HColIndex_cp || HVal_sz > HVal_cp) {
                HRowPtr_cp   = HRowPtr_sz;
                HColIndex_cp = HColIndex_sz;
                HVal_cp      = HVal_sz;
                
                TRY(cudaFree(HRowPtr_d));
                TRY(cudaFree(HColIndex_d));
                TRY(cudaFree(HVal_d));
                
                TRY(cudaMalloc(&HRowPtr_d,   HRowPtr_cp));
                TRY(cudaMalloc(&HColIndex_d, HColIndex_cp));
                TRY(cudaMalloc(&HVal_d,      HVal_cp));
            }
            
            flt_store.resize(2*n_nonzero);
            double_to_float((double *)Hs_a.values, flt_store.size(), flt_store.data());
            TRY(cudaMemcpy(HRowPtr_d,   Hs_a.col_ptrs,    HRowPtr_sz,   cudaMemcpyHostToDevice));
            TRY(cudaMemcpy(HColIndex_d, Hs_a.row_indices, HColIndex_sz, cudaMemcpyHostToDevice));
            TRY(cudaMemcpy(HVal_d,      flt_store.data(), HVal_sz,      cudaMemcpyHostToDevice));
        }
        
        // C = alpha H B + beta C
        void cgemm_H(cx_double alpha, void *B_d, cx_double beta, void *C_d) {
            int n = R.n_rows;
            int s = R.n_cols;
            auto alpha_f = make_cuComplex(alpha.real(), alpha.imag());
            auto beta_f  = make_cuComplex(beta.real(),  beta.imag());
            TRY(cusparseCcsrmm(cs_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                           n, s, n, n_nonzero, // (H rows, B cols, H cols, H nnz)
                           &alpha_f,
                           cs_mat_descr, (cuComplex *)HVal_d, (int *)HRowPtr_d, (int *)HColIndex_d, // H matrix
                           (cuComplex *)B_d, n, // (B, B rows)
                           &beta_f,
                           (cuComplex *)C_d, n)); // (C, C rows)
        }
        
        
        Vec<double> moments(int M) {
            TRY(cudaSetDevice(device));
            
            assert(Hs.n_rows == R.n_rows && Hs.n_cols == R.n_rows);
            
            Vec<double> mu(M);
            mu[0] = Hs.n_rows;  // Tr[T_0[H]] = Tr[1]
            mu[1] = Hs_trace;   // Tr[T_1[H]] = Tr[H]
            
            Vec<void *> a_d { a0_d, a1_d, a2_d };
            TRY(cudaMemcpy(a_d[0], R_d, R_sz, cudaMemcpyDeviceToDevice));    // a0 = T_0[H] R = R
            cgemm_H(1, R_d, 0, a_d[1]);                                      // a1 = T_1[H] R = H R
            
            for (int m = 2; m < M; m++) {
                TRY(cudaMemcpy(a_d[2], a_d[0], R_sz, cudaMemcpyDeviceToDevice));
                cgemm_H(2, a_d[1], -1, a_d[2]);                              // a2 = T_m[H] R = 2 H a1 - a0
                
                mu[m] = cublasCdotc(R.size(), (cuComplex *)R_d, 1, (cuComplex *)a_d[2], 1).x; // R^\dag \dot alpha_2
                
                auto temp = a_d[0];
                a_d[0] = a_d[1];
                a_d[1] = a_d[2];
                a_d[2] = temp;
            }
            return mu;
        }
        
        void stoch_orbital(Vec<double> const& c) {
            TRY(cudaSetDevice(device));
            
            assert(Hs.n_rows == R.n_rows && Hs.n_cols == R.n_rows);
            
            Vec<void *> a_d { a0_d, a1_d, a2_d };
            TRY(cudaMemcpy(a_d[0], R_d, R_sz, cudaMemcpyDeviceToDevice));  // a0 = T_0[H] R = R
            cgemm_H(1, R_d, 0, a_d[1]);                                    // a1 = T_1[H] R = H R
            
            // xi = c0 a0 + c1 a1
            cudaMemset(xi_d, 0, R_sz);
            cublasCaxpy(R.size(), make_cuComplex(c[0], 0), (cuComplex *)a_d[0], 1, (cuComplex *)xi_d, 1);
            cublasCaxpy(R.size(), make_cuComplex(c[1], 0), (cuComplex *)a_d[1], 1, (cuComplex *)xi_d, 1);
            
            int M = c.size();
            for (int m = 2; m < M; m++) {
                TRY(cudaMemcpy(a_d[2], a_d[0], R_sz, cudaMemcpyDeviceToDevice));
                cgemm_H(2, a_d[1], -1, a_d[2]);                            // a2 = T_m[H] R = 2 H a1 - a0
                
                // xi += cm a2
                cublasCaxpy(R.size(), make_cuComplex(c[m], 0), (cuComplex *)a_d[2], 1, (cuComplex *)xi_d, 1);
                
                auto temp = a_d[0];
                a_d[0] = a_d[1];
                a_d[1] = a_d[2];
                a_d[2] = temp;
            }
            
            Vec<float> temp(2*R.size());
            cudaMemcpy(temp.data(), xi_d, R_sz, cudaMemcpyDeviceToHost);
            float_to_double(temp.data(), temp.size(), (double *)xi.memptr());
        }
    };
    
    
    template <typename T>
    std::shared_ptr<Engine<T>> mk_engine_cuSPARSE() {
        int count;
        int err = cudaGetDeviceCount(&count);
        switch (err) {
            case cudaSuccess:
                return std::make_shared<Engine_cuSPARSE<T>>(0);
            case cudaErrorNoDevice:
                std::cerr << "No CUDA device available!\n";
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
    std::shared_ptr<Engine<double>> mk_engine_cuSPARSE() {
        std::cerr << "Real KPM not yet implemented for real values!\n";
        return nullptr; // not yet implemented
    }
    template std::shared_ptr<Engine<cx_double>> mk_engine_cuSPARSE();
}

#endif // WITH_CUDA
