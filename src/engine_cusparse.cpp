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
    std::shared_ptr<EngineCx> mkEngineCx_cuSPARSE(int n, int s) {
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
    
    void doubleToFloat(double *src, int n, float *dst) {
        for (int i = 0; i < n; i++)
            dst[i] = (float)src[i];
    }
    
    void floatToDouble(float *src, int n, double *dst) {
        for (int i = 0; i < n; i++)
            dst[i] = (double)src[i];
    }
    
    class EngineCx_cuSPARSE: public EngineCx {
    public:
        int device;
        cusparseHandle_t csHandle;
        cusparseMatDescr_t csMatDescr;
        
        int size_R, size_HRowPtr, size_HColIndex, size_HVal;
        void *a0_d, *a1_d, *a2_d, *R_d, *xi_d;
        void *HColIndex_d=0, *HRowPtr_d=0, *HVal_d=0;
        
        EngineCx_cuSPARSE(int n, int s, int device): EngineCx(n, s), device(device) {
            size_R       = n*s*sizeof(arma::cx_float);
            size_HRowPtr = (n+1)*sizeof(arma::uword);
            
            TRY(cudaSetDevice(device));
            TRY(cusparseCreate(&csHandle));
            
            TRY(cusparseCreateMatDescr(&csMatDescr));
            // TODO: CUSPARSE_MATRIX_TYPE_HERMITIAN
            cusparseSetMatType(csMatDescr, CUSPARSE_MATRIX_TYPE_GENERAL);
            cusparseSetMatIndexBase(csMatDescr, CUSPARSE_INDEX_BASE_ZERO);
            
            TRY(cudaMalloc(&a0_d, size_R));
            TRY(cudaMalloc(&a1_d, size_R));
            TRY(cudaMalloc(&a2_d, size_R));
            TRY(cudaMalloc(&R_d, size_R));
            TRY(cudaMalloc(&xi_d, size_R));
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
        
        void setHamiltonian(arma::sp_cx_mat const& H, EnergyScale const& es) {
            EngineCx::setHamiltonian(H, es);
            size_HColIndex = Hs.n_nonzero*sizeof(arma::uword);
            size_HVal      = Hs.n_nonzero*sizeof(arma::cx_float);
            
            TRY(cudaSetDevice(device));
            
            TRY(cudaFree(HRowPtr_d));
            TRY(cudaFree(HColIndex_d));
            TRY(cudaFree(HVal_d));
            
            TRY(cudaMalloc(&HRowPtr_d, size_HRowPtr));
            TRY(cudaMalloc(&HColIndex_d, size_HColIndex));
            TRY(cudaMalloc(&HVal_d, size_HVal));
            
            arma::sp_cx_mat HsT = Hs.st(); // csc to csr format
            Vec<float> Hf(2*HsT.n_nonzero);
            doubleToFloat((double *)HsT.values, Hf.size(), Hf.data());
            
            TRY(cudaMemcpy(HRowPtr_d, HsT.col_ptrs, size_HRowPtr, cudaMemcpyHostToDevice));
            TRY(cudaMemcpy(HColIndex_d, HsT.row_indices, size_HColIndex, cudaMemcpyHostToDevice));
            TRY(cudaMemcpy(HVal_d, Hf.data(), size_HVal, cudaMemcpyHostToDevice));
        }
        
        // C = alpha H B + beta C
        void cgemmH(arma::cx_double alpha, void *B_d, arma::cx_double beta, void *C_d) {
            auto alpha_f = make_cuComplex(alpha.real(), alpha.imag());
            auto beta_f  = make_cuComplex(beta.real(),  beta.imag());
            TRY(cusparseCcsrmm(csHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                           n, s, n, Hs.n_nonzero, // (H rows, B cols, H cols, H nnz)
                           &alpha_f,
                           csMatDescr, (cuComplex *)HVal_d, (int *)HRowPtr_d, (int *)HColIndex_d, // H matrix
                           (cuComplex *)B_d, n, // (B, B rows)
                           &beta_f,
                           (cuComplex *)C_d, n)); // (C, C rows)
        }
        
        
        Vec<double> moments(int M) {
            TRY(cudaSetDevice(device));
            Vec<float> Rf(2*R.size());
            doubleToFloat((double *)R.memptr(), Rf.size(), Rf.data());
            cudaMemcpy(R_d, Rf.data(), size_R, cudaMemcpyHostToDevice);

            Vec<double> mu(M);
            mu[0] = n;                      // Tr[T_0[H]] = Tr[1]
            mu[1] = arma::trace(Hs).real(); // Tr[T_1[H]] = Tr[H]
            
            Vec<void *> a_d { a0_d, a1_d, a2_d };
            TRY(cudaMemcpy(a_d[0], R_d, size_R, cudaMemcpyDeviceToDevice)); // a0 = T_0[H] R = R
            cgemmH(1, R_d, 0, a_d[1]);                                      // a1 = T_1[H] R = H R
            
            for (int m = 2; m < M; m++) {
                TRY(cudaMemcpy(a_d[2], a_d[0], size_R, cudaMemcpyDeviceToDevice));
                cgemmH(2, a_d[1], -1, a_d[2]);                              // a2 = T_m[H] R = 2 H a1 - a0
                
                mu[m] = cublasCdotc(n*s, (cuComplex *)R_d, 1, (cuComplex *)a_d[2], 1).x; // R^\dag \dot alpha_2
                
                auto temp = a_d[0];
                a_d[0] = a_d[1];
                a_d[1] = a_d[2];
                a_d[2] = temp;
            }
            return mu;
        }
        
        arma::cx_mat& occupiedOrbital(Vec<double> const& c) {
            TRY(cudaSetDevice(device));
            Vec<float> Rf(2*R.size());
            doubleToFloat((double *)R.memptr(), Rf.size(), Rf.data());
            cudaMemcpy(R_d, Rf.data(), size_R, cudaMemcpyHostToDevice);

            int M = c.size();
            
            Vec<void *> a_d { a0_d, a1_d, a2_d };
            TRY(cudaMemcpy(a_d[0], R_d, size_R, cudaMemcpyDeviceToDevice)); // a0 = T_0[H] R = R
            cgemmH(1, R_d, 0, a_d[1]);                                      // a1 = T_1[H] R = H R

            // xi = c0 a0 + c1 a1
            cudaMemset(xi_d, 0, size_R);
            cublasCaxpy(n*s, make_cuComplex(c[0], 0), (cuComplex *)a_d[0], 1, (cuComplex *)xi_d, 1);
            cublasCaxpy(n*s, make_cuComplex(c[1], 0), (cuComplex *)a_d[1], 1, (cuComplex *)xi_d, 1);
            
            for (int m = 2; m < M; m++) {
                TRY(cudaMemcpy(a_d[2], a_d[0], size_R, cudaMemcpyDeviceToDevice));
                cgemmH(2, a_d[1], -1, a_d[2]);                              // a2 = T_m[H] R = 2 H a1 - a0

                // xi += cm a2
                cublasCaxpy(n*s, make_cuComplex(c[m], 0), (cuComplex *)a_d[2], 1, (cuComplex *)xi_d, 1);
                
                auto temp = a_d[0];
                a_d[0] = a_d[1];
                a_d[1] = a_d[2];
                a_d[2] = temp;
            }
            
            Vec<float> temp(2*n*s);
            cudaMemcpy(temp.data(), xi_d, size_R, cudaMemcpyDeviceToHost);
            floatToDouble(temp.data(), temp.size(), (double *)xi.memptr());
            return xi;
        }
    };
    
    std::shared_ptr<EngineCx> mkEngineCx_cuSPARSE(int n, int s) {
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
