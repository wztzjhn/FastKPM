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
            
            Vec<double> mu(M);
            mu[0] = n;
            mu[1] = std::real(arma::trace(Hs_a));
            
            a0 = this->R;
            a1 = Hs_a * this->R;
            
            for (int m = 2; m < M; m++) {
                a2 = 2*Hs_a*a1 - a0;
                mu[m] = std::real(arma::cdot(this->R, a2));
                a0 = a1;
                a1 = a2;
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
        }
        
        void autodiff_matrix(Vec<double> const& c, SpMatCsr<T>& D) {
            int M = c.size();
            arma::SpMat<T> Hs_a = this->Hs.to_arma();
            int n = this->R.n_rows;
            int s = this->R.n_cols;
            
            arma::Mat<T> b2(n, s);
            arma::Mat<T> b1(n, s, arma::fill::zeros);
            arma::Mat<T> b0 = this->R * c[M - 1];
            
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
