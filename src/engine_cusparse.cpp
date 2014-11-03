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
        std::abort(); \
      } \
    };


namespace fkpm {
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
        Vec<cx_float> cx_float_store;
        
        cusparseHandle_t cs_handle;
        cusparseMatDescr_t cs_mat_descr;
        
        CuVec<cx_float> a0_d, a1_d, a2_d, R_d, xi_d, HVal_d;
        CuVec<int> HColIndex_d, HRowPtr_d;
        
        
        Engine_cuSPARSE(int device) {
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
            a0_d.deallocate();
            a1_d.deallocate();
            a2_d.deallocate();
            R_d.deallocate();
            xi_d.deallocate();
            HVal_d.deallocate();
            HRowPtr_d.deallocate();
            HColIndex_d.deallocate();
        }
        
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
            a0_d.resize(sz);
            a1_d.resize(sz);
            a2_d.resize(sz);
            xi_d.resize(sz);
            
            cx_float_store.resize(sz);
            convert_array(R.memptr(), sz, cx_float_store.data());
            R_d.from_host(sz, cx_float_store.data());
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
            
            cx_float_store.resize(n_nonzero);
            convert_array(Hs.val.data(), n_nonzero, cx_float_store.data());
            HVal_d.from_host(n_nonzero, cx_float_store.data());
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
            
            Vec<CuVec<cx_float>> a_d { a0_d, a1_d, a2_d };
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
            
            Vec<CuVec<cx_float>> a_d { a0_d, a1_d, a2_d };
            a_d[0].from_device(R_d);            // a0 = T_0[H] R = R
            cgemm_H(1, R_d, 0, a_d[1]);         // a1 = T_1[H] R = H R
            
            // xi = c0 a0 + c1 a1
            cudaMemset(xi_d.ptr, 0, xi_d.size*sizeof(cx_float));
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
            
            cx_float_store.resize(xi_d.size);
            xi_d.to_host(cx_float_store.data());
            
            // TODO: replace with kernel call
            /*
            int D_nnz = 0;
            int *D_row_idx = nullptr;
            int *D_col_idx = nullptr;
            cuFloatComplex *D_val = nullptr;
            outer_product(R.n_rows, R.n_cols, (cuFloatComplex *)R_d, (cuFloatComplex *)xi_d, 0.5,
                          D_nnz, D_row_idx, D_col_idx, D_val);
            */
            
            int n = R.n_rows;
            int s = R.n_cols;
            arma::Mat<cx_double> xi(n, s);
            convert_array(cx_float_store.data(), cx_float_store.size(), xi.memptr());
            for (int k = 0; k < D.size(); k++) {
                int i = D.row_idx[k];
                int j = D.col_idx[k];
                cx_double x1 = arma::cdot(R.row(j), xi.row(i)); // xi R^dagger
                cx_double x2 = arma::cdot(xi.row(j), R.row(i)); // R xi^dagger
                D.val[k] = 0.5*(x1+x2);
            }
        }
        
        void autodiff_matrix(Vec<double> const& c, SpMatCsr<cx_double>& D) {
            int M = c.size();
            arma::SpMat<cx_double> Hs_a = this->Hs.to_arma();
            int n = this->R.n_rows;
            int s = this->R.n_cols;
            
            // forward calculation
            
            arma::Mat<cx_double> a0 = this->R;          // T_0[H] |r> = 1 |r>
            arma::Mat<cx_double> a1 = Hs_a * this->R;   // T_1[H] |r> = H |r>
            arma::Mat<cx_double> a2(n, s);
            for (int m = 2; m < M; m++) {
                a2 = 2*Hs_a*a1 - a0;
                a0 = a1;
                a1 = a2;
            }
            
            // reverse calculation
            
            arma::Mat<cx_double> b2(n, s);
            arma::Mat<cx_double> b1(n, s, arma::fill::zeros);
            arma::Mat<cx_double> b0 = this->R * c[M - 1];
            
            // need special logic since mu[1] was calculated exactly
            for (int k = 0; k < D.size(); k++) {
                D.val[k] = (D.row_idx[k] == D.col_idx[k]) ? c[1] : 0;
            }
            Vec<double> cp = c; cp[1] = 0;
            
            for (int m = M-2; m >= 0; m--) {
                // a0 = alpha_{m}
                // b0 = beta_{m}
                for (int k = 0; k < D.size(); k++) {
                    int i = D.row_idx[k];
                    int j = D.col_idx[k];
                    D.val[k] += (m == 0 ? 1.0 : 2.0) * arma::cdot(b0.row(j), a0.row(i));
                }
                a2 = a1;
                b2 = b1;
                a1 = a0;
                b1 = b0;
                a0 = 2*Hs_a*a1 - a2;;
                b0 = cp[m]*this->R + 2*Hs_a*b1 - b2;
            }
            
            for (cx_double& v: D.val) {
                v /= this->es.mag();
            }
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
        std::cerr << "cuSPARSE engine not yet implemented for type `double`!\n";
        return nullptr; // not yet implemented
    }
    template std::shared_ptr<Engine<cx_double>> mk_engine_cuSPARSE();
}

#endif // WITH_CUDA
