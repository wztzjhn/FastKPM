#include <cassert>

#include "fastkpm.h"


namespace fkpm {
    template <typename T>
    class Engine_CPU: public Engine<T> {
        arma::Mat<T> a0;
        arma::Mat<T> a1;
        arma::Mat<T> a2;
        
        Vec<double> moments(int M) {
            arma::SpMat<T> Hs_a = this->Hs.to_arma();
            int n = this->R.n_rows;
            assert(this->Hs.n_rows == n && this->Hs.n_cols == n);
            assert(M % 2 == 0);
            
            Vec<double> mu(M);
            mu[0] = n;
            mu[1] = std::real(arma::trace(Hs_a));
            
            a0 = this->R;          // \alpha_0
            a1 = Hs_a * this->R;   // \alpha_1
            
            for (int m = 1; m < M/2; m++) {
                a2 = 2*Hs_a*a1 - a0;
                a0 = a1;           // \alpha_m
                a1 = a2;           // \alpha_{m+1}
                mu[2*m]   = 2 * std::real(arma::cdot(a0, a0)) - mu[0];
                mu[2*m+1] = 2 * std::real(arma::cdot(a1, a0)) - mu[1];
            }
            
            return mu;
        }
        
        void stoch_matrix(Vec<double> const& c, SpMatCsr<T>& D) {
            int M = c.size();
            
            arma::SpMat<T> Hs_a = this->Hs.to_arma();
            int n = this->R.n_rows;
            assert(this->Hs.n_rows == n && this->Hs.n_cols == n);
            
            a0 = this->R;
            a1 = Hs_a * this->R;
            
            arma::Mat<T> xi = c[0]*a0 + c[1]*a1;
            for (int m = 2; m < M; m++) {
                a2 = 2*Hs_a*a1 - a0;
                xi += c[m]*a2;
                a0 = a1;
                a1 = a2;
            }
            
            for (int k = 0; k < D.size(); k++) {
                int i = D.row_idx[k];
                int j = D.col_idx[k];
                T x1 = arma::cdot(this->R.row(j), xi.row(i)); // xi R^dagger
                T x2 = arma::cdot(xi.row(j), this->R.row(i)); // R xi^dagger
                D.val[k] = 0.5*(x1+x2);
            }
            
            // a0 and a1 matrices invalid (too large)
            a0.clear();
            a1.clear();
        }
        
        void autodiff_matrix(Vec<double> const& c, SpMatCsr<T>& D) {
            int M = c.size();
            arma::SpMat<T> Hs_a = this->Hs.to_arma();
            int n = this->R.n_rows;
            int s = this->R.n_cols;
            
            double diag = c[1];
            for (int m = 1; m < M/2; m++) {
                diag -= c[2*m+1];
            }
            for (int k = 0; k < D.size(); k++) {
                D.val[k] = (D.row_idx[k] == D.col_idx[k]) ? diag : 0;
            }
            
            arma::Mat<T> b2(n, s);
            arma::Mat<T> b1(n, s, arma::fill::zeros);   // \beta_{M/2+1}
            arma::Mat<T> b0(n, s, arma::fill::zeros);   // \beta_{M/2}
            if (M > 2)
                b0 += 2 * c[M-1] * a1;
            
            for (int m = M/2-1; m >= 1; m--) {
                // a0 = \alpha_m, b0 = \beta_{m+1}
                
                // D += 2 \alpha_m \beta_{m+1}^\dagger
                for (int k = 0; k < D.size(); k++) {
                    D.val[k] += 2.0 * arma::cdot(b0.row(D.col_idx[k]), a0.row(D.row_idx[k]));
                }
                
                a2 = a1;
                a1 = a0;
                a0 = 2*Hs_a*a1 - a2;
                
                b2 = b1;
                b1 = b0;
                b0 = 4*c[2*m]*a1 + 2*c[2*m+1]*a2 + 2*Hs_a*b1 - b2;
                if (m > 1) {
                    b0 += 2*c[2*m-1]*a0;
                }
            }
            
            // D += \alpha_0 \beta_1^\dagger
            for (int k = 0; k < D.size(); k++) {
                D.val[k] += arma::cdot(b0.row(D.col_idx[k]), a0.row(D.row_idx[k]));
            }
            D.symmetrize();
            for (T& v: D.val) {
                v /= this->es.mag();
            }
            
            // a0 and a1 matrices have been invalidated
            a0.clear();
            a1.clear();
        }
    };
    
    template <typename T>
    std::shared_ptr<Engine<T>> mk_engine_cpu() {
        return std::make_shared<Engine_CPU<T>>();
    }
    template std::shared_ptr<Engine<double>> mk_engine_cpu();
    template std::shared_ptr<Engine<cx_double>> mk_engine_cpu();
}
