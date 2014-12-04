#include <cassert>

#include "fastkpm.h"


namespace fkpm {
    template <typename T>
    class Engine_CPU: public Engine<T> {
        EnergyScale es{0, 0};  // Scaling bounds
        arma::SpMat<T> Hs;     // Scaled Hamiltonian
        arma::Mat<T> a0;
        arma::Mat<T> a1;
        arma::Mat<T> a2;
        
        
        void set_H(SpMatBsr<T> const& H, EnergyScale const& es) {
            assert(H.n_rows == H.n_cols);
            this->es = es;
            Hs = H.to_arma();
            for (int i = 0; i < Hs.n_rows; i++) {
                Hs(i, i) -= es.avg();
            }
            Hs /= es.mag();
        }
        
        Vec<double> moments(int M) {
            int n = this->R.n_rows;
            assert(Hs.n_rows == n && Hs.n_cols == n);
            assert(M % 2 == 0);
            
            Vec<double> mu(M);
            mu[0] = n;
            mu[1] = std::real(arma::trace(Hs));
            
            a0 = this->R;        // \alpha_0
            a1 = Hs * this->R;   // \alpha_1
            
            for (int m = 1; m < M/2; m++) {
                a2 = 2*Hs*a1 - a0;
                a0 = a1;         // \alpha_m
                a1 = a2;         // \alpha_{m+1}
                mu[2*m]   = 2 * std::real(arma::cdot(a0, a0)) - mu[0];
                mu[2*m+1] = 2 * std::real(arma::cdot(a1, a0)) - mu[1];
            }
            
            return mu;
        }
        
        // D += alpha A B^\dagger
        void outer_product(T alpha, arma::Mat<T> const& A, arma::Mat<T> const& B, SpMatBsr<T>& D) {
            for (int k = 0; k < D.n_blocks(); k++) {
                int b_len = D.b_len;
                int i = D.row_idx[k];
                int j = D.col_idx[k];
                T* v = &D.val[b_len*b_len*k];
                
                for (int bj = 0; bj < b_len; bj++) {
                    for (int bi = 0; bi < b_len; bi++) {
                        v[b_len*bj + bi] += alpha * arma::cdot(B.row(b_len*j+bj), A.row(b_len*i+bi));
                    }
                }
            }
        }
        
        
        void stoch_matrix(Vec<double> const& c, SpMatBsr<T>& D) {
            int M = c.size();
            int n = this->R.n_rows;
            assert(D.b_len*D.n_rows == Hs.n_rows && D.b_len*D.n_cols == Hs.n_rows);
            assert(Hs.n_rows == n);
            
            a0 = this->R;
            a1 = Hs * this->R;
            
            arma::Mat<T> xi = c[0]*a0 + c[1]*a1;
            for (int m = 2; m < M; m++) {
                a2 = 2*Hs*a1 - a0;
                xi += c[m]*a2;
                a0 = a1;
                a1 = a2;
            }
            
            D.zeros();
            outer_product(0.5, this->R, xi, D);
            outer_product(0.5, xi, this->R, D);
            
            // a0 and a1 matrices are invalid (too large)
            a0.clear();
            a1.clear();
        }
        
        void autodiff_matrix(Vec<double> const& c, SpMatBsr<T>& D) {
            int M = c.size();
            int n = this->R.n_rows;
            int s = this->R.n_cols;
            assert(D.b_len*D.n_rows == Hs.n_rows && D.b_len*D.n_cols == Hs.n_rows);
            assert(Hs.n_rows == n);
            
            double diag = c[1];
            for (int m = 1; m < M/2; m++) {
                diag -= c[2*m+1];
            }
            D.zeros();
            for (int k = 0; k < D.n_blocks(); k++) {
                if (D.row_idx[k] == D.col_idx[k]) {
                    T* v = &D.val[D.b_len*D.b_len*k];
                    for (int bi = 0; bi < D.b_len; bi++) {
                        v[D.b_len*bi + bi] = diag;
                    }
                }
            }
            
            arma::Mat<T> b2(n, s);
            arma::Mat<T> b1(n, s, arma::fill::zeros);   // \beta_{M/2+1}
            arma::Mat<T> b0(n, s, arma::fill::zeros);   // \beta_{M/2}
            if (M > 2)
                b0 = 2 * c[M-1] * a1;
            
            for (int m = M/2-1; m >= 1; m--) {
                // a0 = \alpha_m, b0 = \beta_{m+1}
                
                // D += 2 \alpha_m \beta_{m+1}^\dagger
                outer_product(2, a0, b0, D);
                
                a2 = a1;
                a1 = a0;
                a0 = 2*Hs*a1 - a2;
                
                b2 = b1;
                b1 = b0;
                b0 = 4*c[2*m]*a1 + 2*c[2*m+1]*a2 + 2*Hs*b1 - b2;
                if (m > 1) {
                    b0 += 2*c[2*m-1]*a0;
                }
            }
            
            // D += \alpha_0 \beta_1^\dagger
            outer_product(1, a0, b0, D);
            D.symmetrize();
            D.scale(1.0/es.mag());
            
            // a0 and a1 matrices have been invalidated
            a0.clear();
            a1.clear();
        }
    };
    
    template <typename T>
    std::shared_ptr<Engine<T>> mk_engine_cpu() {
        return std::make_shared<Engine_CPU<T>>();
    }
    template std::shared_ptr<Engine<float>> mk_engine_cpu();
    template std::shared_ptr<Engine<double>> mk_engine_cpu();
    template std::shared_ptr<Engine<cx_float>> mk_engine_cpu();
    template std::shared_ptr<Engine<cx_double>> mk_engine_cpu();
}
