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
    std::shared_ptr<EngineCx> mk_engine_cx_cuSPARSE(int n, int s) {
        return nullptr;
    }
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
    
    class EngineCx_cuSPARSE: public EngineCx {
    public:
        arma::sp_cx_mat Hs;    // Scaled Hamiltonian
        arma::sp_cx_mat dE_dH; // Grand free energy matrix derivative
        
        int device;
        cusparseHandle_t cs_handle;
        cusparseMatDescr_t cs_mat_descr;
        
        int R_sz, HRowPtr_sz, HColIndex_sz, HVal_sz;
        void *a0_d, *a1_d, *a2_d, *R_d, *xi_d;
        void *HColIndex_d=0, *HRowPtr_d=0, *HVal_d=0;
        
        EngineCx_cuSPARSE(int n, int s, int device): EngineCx(n, s), device(device) {
            R_sz       = n*s*sizeof(arma::cx_float);
            HRowPtr_sz = (n+1)*sizeof(arma::uword);
            
            TRY(cudaSetDevice(device));
            TRY(cusparseCreate(&cs_handle));
            
            TRY(cusparseCreateMatDescr(&cs_mat_descr));
            // TODO: CUSPARSE_MATRIX_TYPE_HERMITIAN
            cusparseSetMatType(cs_mat_descr, CUSPARSE_MATRIX_TYPE_GENERAL);
            cusparseSetMatIndexBase(cs_mat_descr, CUSPARSE_INDEX_BASE_ZERO);
            
            TRY(cudaMalloc(&a0_d, R_sz));
            TRY(cudaMalloc(&a1_d, R_sz));
            TRY(cudaMalloc(&a2_d, R_sz));
            TRY(cudaMalloc(&R_d, R_sz));
            TRY(cudaMalloc(&xi_d, R_sz));
        }
        
        ~EngineCx_cuSPARSE() {
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
        
        void set_H(arma::sp_cx_mat const& H, EnergyScale const& es) {
            this-> es = es;
            Hs = es.scale(H);
            dE_dH = Hs;
            
            HColIndex_sz = Hs.n_nonzero*sizeof(arma::uword);
            HVal_sz      = Hs.n_nonzero*sizeof(arma::cx_float);
            
            TRY(cudaSetDevice(device));
            
            // TODO: avoid free and malloc if size is unchanged
            TRY(cudaFree(HRowPtr_d));
            TRY(cudaFree(HColIndex_d));
            TRY(cudaFree(HVal_d));
            
            TRY(cudaMalloc(&HRowPtr_d, HRowPtr_sz));
            TRY(cudaMalloc(&HColIndex_d, HColIndex_sz));
            TRY(cudaMalloc(&HVal_d, HVal_sz));
            
            arma::sp_cx_mat HsT = Hs.st(); // csc to csr format
            Vec<float> Hf(2*HsT.n_nonzero);
            double_to_float((double *)HsT.values, Hf.size(), Hf.data());
            
            TRY(cudaMemcpy(HRowPtr_d, HsT.col_ptrs, HRowPtr_sz, cudaMemcpyHostToDevice));
            TRY(cudaMemcpy(HColIndex_d, HsT.row_indices, HColIndex_sz, cudaMemcpyHostToDevice));
            TRY(cudaMemcpy(HVal_d, Hf.data(), HVal_sz, cudaMemcpyHostToDevice));
        }
        
        // C = alpha H B + beta C
        void cgemm_H(arma::cx_double alpha, void *B_d, arma::cx_double beta, void *C_d) {
            auto alpha_f = make_cuComplex(alpha.real(), alpha.imag());
            auto beta_f  = make_cuComplex(beta.real(),  beta.imag());
            TRY(cusparseCcsrmm(cs_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                           n, s, n, Hs.n_nonzero, // (H rows, B cols, H cols, H nnz)
                           &alpha_f,
                           cs_mat_descr, (cuComplex *)HVal_d, (int *)HRowPtr_d, (int *)HColIndex_d, // H matrix
                           (cuComplex *)B_d, n, // (B, B rows)
                           &beta_f,
                           (cuComplex *)C_d, n)); // (C, C rows)
        }
        
        
        Vec<double> moments(int M) {
            TRY(cudaSetDevice(device));
            Vec<float> Rf(2*R.size());
            double_to_float((double *)R.memptr(), Rf.size(), Rf.data());
            cudaMemcpy(R_d, Rf.data(), R_sz, cudaMemcpyHostToDevice);

            Vec<double> mu(M);
            mu[0] = n;                      // Tr[T_0[H]] = Tr[1]
            mu[1] = arma::trace(Hs).real(); // Tr[T_1[H]] = Tr[H]
            
            Vec<void *> a_d { a0_d, a1_d, a2_d };
            TRY(cudaMemcpy(a_d[0], R_d, R_sz, cudaMemcpyDeviceToDevice)); // a0 = T_0[H] R = R
            cgemm_H(1, R_d, 0, a_d[1]);                                      // a1 = T_1[H] R = H R
            
            for (int m = 2; m < M; m++) {
                TRY(cudaMemcpy(a_d[2], a_d[0], R_sz, cudaMemcpyDeviceToDevice));
                cgemm_H(2, a_d[1], -1, a_d[2]);                              // a2 = T_m[H] R = 2 H a1 - a0
                
                mu[m] = cublasCdotc(n*s, (cuComplex *)R_d, 1, (cuComplex *)a_d[2], 1).x; // R^\dag \dot alpha_2
                
                auto temp = a_d[0];
                a_d[0] = a_d[1];
                a_d[1] = a_d[2];
                a_d[2] = temp;
            }
            return mu;
        }
        
        void stoch_orbital(Vec<double> const& c) {
            TRY(cudaSetDevice(device));
            Vec<float> Rf(2*R.size());
            double_to_float((double *)R.memptr(), Rf.size(), Rf.data());
            cudaMemcpy(R_d, Rf.data(), R_sz, cudaMemcpyHostToDevice);

            int M = c.size();
            
            Vec<void *> a_d { a0_d, a1_d, a2_d };
            TRY(cudaMemcpy(a_d[0], R_d, R_sz, cudaMemcpyDeviceToDevice)); // a0 = T_0[H] R = R
            cgemm_H(1, R_d, 0, a_d[1]);                                      // a1 = T_1[H] R = H R

            // xi = c0 a0 + c1 a1
            cudaMemset(xi_d, 0, R_sz);
            cublasCaxpy(n*s, make_cuComplex(c[0], 0), (cuComplex *)a_d[0], 1, (cuComplex *)xi_d, 1);
            cublasCaxpy(n*s, make_cuComplex(c[1], 0), (cuComplex *)a_d[1], 1, (cuComplex *)xi_d, 1);
            
            for (int m = 2; m < M; m++) {
                TRY(cudaMemcpy(a_d[2], a_d[0], R_sz, cudaMemcpyDeviceToDevice));
                cgemm_H(2, a_d[1], -1, a_d[2]);                              // a2 = T_m[H] R = 2 H a1 - a0

                // xi += cm a2
                cublasCaxpy(n*s, make_cuComplex(c[m], 0), (cuComplex *)a_d[2], 1, (cuComplex *)xi_d, 1);
                
                auto temp = a_d[0];
                a_d[0] = a_d[1];
                a_d[1] = a_d[2];
                a_d[2] = temp;
            }
            
            Vec<float> temp(2*n*s);
            cudaMemcpy(temp.data(), xi_d, R_sz, cudaMemcpyDeviceToHost);
            float_to_double(temp.data(), temp.size(), (double *)xi.memptr());
        }
    };
    
    std::shared_ptr<EngineCx> mk_engine_cx_cuSPARSE(int n, int s) {
        int count;
        int err = cudaGetDeviceCount(&count);
        switch (err) {
            case cudaSuccess:
                return std::make_shared<EngineCx_cuSPARSE>(n, s, 0);
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
}

#endif // WITH_CUDA
