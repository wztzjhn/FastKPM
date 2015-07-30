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
            
            a0 = this->R;        // \alpha_0
            a1 = Hs * this->R;   // \alpha_1
            mu[0] = std::real(arma::cdot(a0, a0));
            mu[1] = std::real(arma::cdot(a1, a0));
            
            for (int m = 1; m < M/2; m++) {
                a2 = 2*Hs*a1 - a0;
                a0 = a1;         // \alpha_m
                a1 = a2;         // \alpha_{m+1}
                mu[2*m]   = 2 * std::real(arma::cdot(a0, a0)) - mu[0];
                mu[2*m+1] = 2 * std::real(arma::cdot(a1, a0)) - mu[1];
            }
            
            return mu;
        }

        Vec<Vec<cx_double>> moments_tensor(int M, SpMatBsr<T> const& j1_BSR, SpMatBsr<T> const& j2_BSR, int ncols_keep) {
            std::cout << "Calculating Moments Tensor..." << std::endl;
            //     each row: represents the order of Chebyshev polynomial
            //     each col: a_k(0:N-1) same dimension as the hamiltonian
            arma::Mat<T> alpha;         // size depends on memory usage
            arma::Mat<T> atild;
            //const int max_elems = 1e7;  // 1e7 elements ~ 100M memory to store alpha
            
            auto j1 = j1_BSR.to_arma();
            auto j2 = j2_BSR.to_arma();
            int n = this->R.n_rows;
            int alpha_ncols;
            if (ncols_keep >= 3 && ncols_keep <= M) {
                alpha_ncols = ncols_keep;
            } else {
                alpha_ncols = 10;
                //alpha_ncols = std::min(max_elems/n, M);
            }
            int atild_ncols = alpha_ncols;
            assert(Hs.n_rows == n && Hs.n_cols == n);
            assert(j1.n_rows == n && j1.n_cols == n);
            assert(j2.n_rows == n && j2.n_cols == n);
            assert(M % 2 == 0);
            
            alpha.set_size(n,alpha_ncols);
            atild.set_size(n,atild_ncols);
            Vec<Vec<cx_double>> mu(M);
            for (int i = 0; i < M; i++) {                   // initialize mu to 0
                mu[i].resize(M, cx_double(0.0,0.0));
            }
            //mu[0][0] = arma::trace(j2 * j1);
            //mu[0][1] = arma::trace(j2 * Hs * j1);
            //mu[1][0] = arma::trace(Hs * j2 * j1);
            //mu[1][1] = arma::trace(Hs * j2 * Hs * j1);
            
            // # of sparse * vector operation: O(M^2 * s) if naive, or O(M * s) in the block way
            // # of vector * vector operation: O(M^2 * s) so dominant in time
            for (int k=0; k < this->R.n_cols; k++) {
                int alpha_begin = 0;                        // *** divide the calculation into blocks, each
                int alpha_end   = alpha_ncols - 1;          //     dimension with two pointers for boundary ***
                alpha.col(0)    = this->R.col(k);           // \alpha_0
                alpha.col(1)    = Hs * alpha.col(0);        // \alpha_1
                while (alpha_begin <= alpha_end) {
                    if (alpha_begin != 0) {                 // build alpha.col(begin:end)
                        alpha.col(0) = 2 * Hs * alpha.col(alpha_ncols-1) - alpha.col(alpha_ncols-2);
                        alpha.col(1) = 2 * Hs * alpha.col(0) - alpha.col(alpha_ncols-1);
                    }
                    for (int m1 = 2; m1 <= alpha_end - alpha_begin; m1++)
                        alpha.col(m1) = 2 * Hs * alpha.col(m1-1) - alpha.col(m1-2);
                    int atild_begin = 0;
                    int atild_end   = atild_ncols - 1;
                    atild.col(0)    = j1 * this->R.col(k);  // reset \tilde{alpha}
                    atild.col(1)    = Hs * atild.col(0);
                    while (atild_begin <= atild_end) {
                        if (atild_begin != 0) {             // build atild.col(begin:end)
                            atild.col(0) = 2 * Hs * atild.col(atild_ncols-1) - atild.col(atild_ncols-2);
                            atild.col(1) = 2 * Hs * atild.col(0) - atild.col(atild_ncols-1);
                        }
                        for (int m2 = 2; m2 <= atild_end - atild_begin; m2++)
                            atild.col(m2) = 2 * Hs * atild.col(m2-1) - atild.col(m2-2);
                        for (int m1 = alpha_begin; m1 <= alpha_end; m1++) {
                            for (int m2 = atild_begin; m2 <= atild_end; m2++) {
                                mu[m1][m2] += arma::cdot(alpha.col(m1-alpha_begin), j2 * atild.col(m2-atild_begin));
                            }
                        }
                        atild_begin = atild_end + 1;
                        atild_end   = std::min(M-1, atild_end + atild_ncols);
                    }
                    alpha_begin = alpha_end + 1;
                    alpha_end   = std::min(M-1, alpha_end + alpha_ncols);
                }
                std::cout << "*";
            }
            std::cout << " -> 100%." << std::endl;
            alpha.reset();
            atild.reset();
            
            
//                        std::cout << "-----This part only for Bench Marking-----" << std::endl;
//                        arma::Mat<cx_double> mu_test(M, M);
//                        mu_test.zeros();
//                        Vec<arma::Mat<T>> cheby(M);
//                        cheby[0].eye(n,n);
//                        cheby[1] = Hs;
//                        for (int m=2; m < M; m++) {
//                            cheby[m] = 2 * Hs * cheby[m-1] - cheby[m-2];
//                        }
//                        for (int m1=0; m1 < M; m1++) {
//                            for (int m2=0; m2 < M; m2++) {
//                                mu_test(m1,m2)=arma::trace(cheby[m1]*j2*cheby[m2]*j1);
//                                if (std::abs(mu[m1][m2]-mu_test(m1,m2))>1e-12) {
//                                    std::cout << "error(" << m1 << "," << m2 << ")=" << mu[m1][m2]-mu_test(m1,m2) << std::endl;
//                                }
//                            }
//                        }
            
            j1.reset();
            j2.reset();
            return mu;
        }

        Vec<Vec<cx_double>> moments_tensor_version0(int M, SpMatBsr<T> const& j1_BSR, SpMatBsr<T> const& j2_BSR, int ncols_keep) {
            std::cout << "Calculating Moments Tensor..." << std::endl;
            //     each row: represents the order of Chebyshev polynomial
            //     each col: a_k(0:N-1) same dimension as the hamiltonian
            arma::Mat<T> alpha;         // size depends on memory usage
            arma::Mat<T> atild;
            //const int max_elems = 1e7;  // 1e7 elements ~ 100M memory to store alpha
            
            auto j1 = j1_BSR.to_arma();
            auto j2 = j2_BSR.to_arma();
            int n = this->R.n_rows;
            int alpha_ncols;
            if (ncols_keep >= 3 && ncols_keep <= M) {
                alpha_ncols = ncols_keep;
            } else {
                alpha_ncols = 10;
                //alpha_ncols = std::min(max_elems/n, M);
            }
            int atild_ncols = alpha_ncols;
            assert(Hs.n_rows == n && Hs.n_cols == n);
            assert(j1.n_rows == n && j1.n_cols == n);
            assert(j2.n_rows == n && j2.n_cols == n);
            assert(M % 2 == 0);

            alpha.set_size(n,alpha_ncols);
            atild.set_size(n,atild_ncols);
            Vec<Vec<cx_double>> mu(M);
            for (int i = 0; i < M; i++) {                   // initialize mu to 0
                mu[i].resize(M, cx_double(0.0,0.0));
            }
            mu[0][0] = arma::trace(j2 * j1);
            mu[0][1] = arma::trace(j2 * Hs * j1);
            mu[1][0] = arma::trace(Hs * j2 * j1);
            mu[1][1] = arma::trace(Hs * j2 * Hs * j1);
            
            // random number averaging, this loop can be MPI parallized
            // # of sparse * vector operation: O(M^2 * s) if naive, or O(M * s) in the block way
            // # of vector * vector operation: O(M^2 * s) so dominant in time
            for (int k=0; k < this->R.n_cols; k++) {
                int alpha_begin = 0;                        // *** divide the calculation into blocks, each
                int alpha_end   = alpha_ncols - 1;          //     dimension with two pointers for boundary ***
                alpha.col(0)    = this->R.col(k);           // \alpha_0
                alpha.col(1)    = Hs * alpha.col(0);        // \alpha_1
                while (alpha_begin <= alpha_end) {
                    if (alpha_begin != 0) {                 // build alpha.col(begin:end)
                        alpha.col(0) = 2 * Hs * alpha.col(alpha_ncols-1) - alpha.col(alpha_ncols-2);
                        alpha.col(1) = 2 * Hs * alpha.col(0) - alpha.col(alpha_ncols-1);
                    }
                    for (int m1 = 2; m1 <= alpha_end - alpha_begin; m1++)
                        alpha.col(m1) = 2 * Hs * alpha.col(m1-1) - alpha.col(m1-2);
                    int atild_begin = 0;
                    int atild_end   = atild_ncols - 1;
                    atild.col(0)    = j1 * this->R.col(k);  // reset \tilde{alpha}
                    atild.col(1)    = Hs * atild.col(0);
                    while (atild_begin <= atild_end) {
                        if (atild_begin != 0) {             // build atild.col(begin:end)
                            atild.col(0) = 2 * Hs * atild.col(atild_ncols-1) - atild.col(atild_ncols-2);
                            atild.col(1) = 2 * Hs * atild.col(0) - atild.col(atild_ncols-1);
                        }
                        for (int m2 = 2; m2 <= atild_end - atild_begin; m2++)
                            atild.col(m2) = 2 * Hs * atild.col(m2-1) - atild.col(m2-2);
                        for (int m1 = alpha_begin; m1 <= alpha_end; m1++) {
                            for (int m2 = atild_begin; m2 <= atild_end; m2++) {
                                mu[m1][m2] += (m1>1 || m2>1)?arma::cdot(alpha.col(m1-alpha_begin), j2 * atild.col(m2-atild_begin)):0.0;
                            }
                        }
                        atild_begin = atild_end + 1;
                        atild_end   = std::min(M-1, atild_end + atild_ncols);
                    }
                    alpha_begin = alpha_end + 1;
                    alpha_end   = std::min(M-1, alpha_end + alpha_ncols);
                }
                std::cout << "*";
            }
            std::cout << " -> 100%." << std::endl;
            alpha.reset();
            atild.reset();
            j1.reset();
            j2.reset();
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
            // ? why not clean a2?
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
            outer_product(diag, this->R, this->R, D);
            
            // Note that the \beta_m matrices are the "adjoint nodes" to alpha_m. Specifically,
            // \beta^*_m = (d / d\alpha_m) tr g,
            // This is a total derivative, including all implicit dependencies in the call graph.
            arma::Mat<T> b2(n, s);
            arma::Mat<T> b1(n, s, arma::fill::zeros);   // \beta_{M/2+1}
            arma::Mat<T> b0(n, s, arma::fill::zeros);   // \beta_{M/2}
            if (M > 2)
                b0 = 2 * c[M-1] * a0;   // a0 = \alpha_{M/2-1} after moments() calculation
            
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
            // ? why not clean a2?
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
