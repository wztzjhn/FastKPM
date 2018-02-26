//
//  fastkpm.cpp
//  tibidy
//
//  Created by Kipton Barros on 5/22/14.
//
//

#include <cassert>
#include <algorithm>
#include <boost/math/tools/roots.hpp>

#ifdef WITH_FFTW
#include <fftw3.h>
#endif

#include "fastkpm.h"


namespace fkpm {
    
    Vec<double> jackson_kernel(int M) {
        auto ret = Vec<double>(M);
        double Mp = M+1.0;
        for (int m = 0; m < M; m++) {
            ret[m] = (1.0/Mp)*((Mp-m)*cos(Pi*m/Mp) + sin(Pi*m/Mp)/tan(Pi/Mp));
        }
        return ret;
    }
    
    Vec<double> lorentz_kernel(int M, double lambda) {
        auto ret = Vec<double>(M);
        for (int m = 0; m < M; m++) {
            ret[m] = sinh(lambda * (1.0 - ((double) m)/ M)) / sinh(lambda);
        }
        return ret;
    }
    
    void chebyshev_fill_array(double x, Vec<double>& ret, int kind) {
        assert(kind == 1 || kind == 2);
        if (ret.size() > 0)
            ret[0] = 1.0;
        if (ret.size() > 1)
            ret[1] = kind * x;
        for (int m = 2; m < ret.size(); m++) {
            ret[m] = 2*x*ret[m-1] - ret[m-2];
        }
    }
    
    Vec<double> expansion_coefficients(int M, int Mq, std::function<double(double)> f, EnergyScale es) {
        assert(Mq >= M);
        auto kernel = jackson_kernel(M);
        auto ret = Vec<double>(M, 0.0);
#ifdef WITH_FFTW
        double *xc, *yc;
        xc = (double*) fftw_malloc(sizeof(double) * Mq);
        yc = (double*) fftw_malloc(sizeof(double) * Mq);
        for (int i = 0; i < Mq; i++) {
            double x_i = cos(Pi * (i+0.5) / Mq);
            xc[i] = f(es.unscale(x_i));
        }
        fftw_plan p;
        p = fftw_plan_r2r_1d(Mq, xc, yc, FFTW_REDFT10, FFTW_ESTIMATE);  // DCT-II
        fftw_execute(p);
        fftw_destroy_plan(p);
        for (int m = 0; m < M; m++) {
            ret[m] = (m == 0 ? 0.5 : 1.0) * kernel[m] * yc[m] / Mq;
        }
        fftw_free(xc);
        fftw_free(yc);
#else
        std::cout << "Warning: Not using FFTW (expansion_coefficients)." << std::endl;
        auto fp = Vec<double>(M, 0.0);
        auto T = Vec<double>(M);
        for (int i = 0; i < Mq; i++) {
            double x_i = cos(Pi * (i+0.5) / Mq);
            double f_i = f(es.unscale(x_i));
            chebyshev_fill_array(x_i, T);
            for (int m = 0; m < M; m++) {
                fp[m] += f_i * T[m];
            }
        }
        for (int m = 0; m < M; m++) {
            ret[m] = (m == 0 ? 1.0 : 2.0) * kernel[m] * fp[m] / Mq;
        }
#endif
        return ret;
    }
    
    Vec<Vec<cx_double>> electrical_conductivity_coefficients(int M, int Mq, double kT, double mu,
                                                             double omega, EnergyScale es, Vec<double> const& kernel) {
        assert(kernel.size() == M);
        assert(Mq >= 2*M);                                                      // To simplify usage of Y_{m_1+m_2} (see notes on fft)
        assert(omega >= 0.0);
        double cutoff = 1e-4;                                                   // neglect points near the boundary
        double omega_scaled = omega / es.mag();                                 // rescale omega
        Vec<Vec<cx_double>> ret(M);
        for (int i = 0; i < M; i++) {                                           // initialize cmn to 0
            ret[i].resize(M, 0.0);
        }
        if (omega_scaled >= 2.0 ) return ret;
        
        if (omega_scaled < 1e-10) {                                             // static conductivity
#ifdef WITH_FFTW
            double *x1, *x2, *yc1, *yc2, *ys1, *ys2;
            fftw_plan p1, p2, p3, p4;
            x1  = (double*) fftw_malloc(sizeof(double) * Mq);
            x2  = (double*) fftw_malloc(sizeof(double) * Mq);
            yc1 = (double*) fftw_malloc(sizeof(double) * Mq);
            yc2 = (double*) fftw_malloc(sizeof(double) * Mq);
            ys1 = (double*) fftw_malloc(sizeof(double) * Mq);
            ys2 = (double*) fftw_malloc(sizeof(double) * Mq);
            for (int i = 0; i < Mq; i++) {
                double x_i = cos(Pi * (i+0.5) / Mq);
                if (1.0-x_i*x_i < cutoff) {                                     // neglect points near boundary
                    x1[i] = 0.0;
                    x2[i] = 0.0;
                } else {
                    double f_i = fermi_density(es.unscale(x_i), kT, mu);
                    x1[i] = f_i * x_i / std::pow(1.0-x_i*x_i, 1.5);
                    x2[i] = f_i / (1.0-x_i*x_i);
                }
            }
            p1  = fftw_plan_r2r_1d(Mq, x1, yc1, FFTW_REDFT10, FFTW_ESTIMATE);   // DCT-II
            fftw_execute(p1);
            p2  = fftw_plan_r2r_1d(Mq, x2, yc2, FFTW_REDFT10, FFTW_ESTIMATE);   // DCT-II
            fftw_execute(p2);
            p3  = fftw_plan_r2r_1d(Mq, x1, ys1, FFTW_RODFT10, FFTW_ESTIMATE);   // DST-II
            fftw_execute(p3);
            p4  = fftw_plan_r2r_1d(Mq, x2, ys2, FFTW_RODFT10, FFTW_ESTIMATE);   // DST-II
            fftw_execute(p4);
            fftw_destroy_plan(p1);
            fftw_destroy_plan(p2);
            fftw_destroy_plan(p3);
            fftw_destroy_plan(p4);
            fftw_free(x1);
            fftw_free(x2);
            for (int m1 = 0; m1 < M; m1++) {
                double temp_m1 = Pi * (m1 == 0 ? 1.0 : 2.0) * kernel[m1] / (2.0 * Mq * es.mag() * es.mag());
                for (int m2 = 0; m2 < M; m2++) {
                    double temp_m2 = temp_m1 * (m2 == 0 ? 1.0 : 2.0) * kernel[m2];
                    int m_sum   = m1 + m2;
                    int m_dif   = m1 - m2;
                    double y_re =  2.0 * yc1[m_sum]
                                 + 2.0 * (m_dif >= 0 ? yc1[m_dif] : yc1[-m_dif])
                                 + m_sum * (m_sum > 0 ? ys2[m_sum-1] : 0.0)
                                 + m_dif * (m_dif > 0 ? ys2[m_dif-1] : (m_dif < 0 ? -ys2[-m_dif-1] : 0.0));
                    double y_im =  2.0 * (m_dif > 0 ? ys1[m_dif-1] : (m_dif < 0 ? -ys1[-m_dif-1] : 0.0))
                                 - m_dif * (yc2[m_sum] + (m_dif >=0 ? yc2[m_dif] : yc2[-m_dif]));
                    ret[m1][m2] = cx_double(temp_m2 * y_re, temp_m2 * y_im);
                }
            }
            fftw_free(yc1);
            fftw_free(yc2);
            fftw_free(ys1);
            fftw_free(ys2);
#else
            std::cout << "Warning: Not using FFTW (electrical_conductivity_coefficients)." << std::endl;
            auto T_i = Vec<double>(M);
            auto T_j = Vec<double>(M);
            for (int i = 0; i < Mq; i++) {
                double x_i = cos(Pi * (i+0.5) / Mq);
                if (1.0-x_i*x_i < cutoff) continue;                             // neglect points near boundary
                double temp_squareroot1 = std::sqrt(1.0-x_i*x_i);
                double f_i;
                chebyshev_fill_array(x_i, T_i);
                chebyshev_fill_array(x_i, T_j, 2);                              // fill T_j with sin[m * arccos(x)]
                for (int m = M-1; m > 0; m--) {
                    T_j[m] = T_j[m-1] * temp_squareroot1;
                }
                T_j[0] = 0.0;
                f_i = fermi_density(es.unscale(x_i), kT, mu) / std::pow(1.0-x_i*x_i, 1.5);
                for (int m1 = 0; m1 < M; m1++) {
                    for (int m2 = 0; m2 < M; m2++) {
                        ret[m1][m2] += ( T_i[m1] * cx_double(T_i[m2],-T_j[m2])
                                        * cx_double(x_i, m2 * temp_squareroot1)
                                        +T_i[m2] * cx_double(T_i[m1], T_j[m1])
                                        * cx_double(x_i,-m1 * temp_squareroot1) ) * f_i;
                    }
                }
            }
            for (int m1 = 0; m1 < M; m1++) {
                double temp_m1 = 2.0 * Pi * (m1 == 0 ? 1.0 : 2.0) * kernel[m1] / (Mq * es.mag() * es.mag());
                for (int m2 = 0; m2 < M; m2++) {
                    double temp_m2 = temp_m1 * (m2 == 0 ? 1.0 : 2.0) * kernel[m2];
                    ret[m1][m2] *= cx_double(temp_m2, 0.0);
                }
            }
#endif
        } else {                                                                // optical conductivity
            std::cout << "Warning: FFTW not implemented yet (electrical_conductivity_coefficients)." << std::endl;
            int i_start = std::ceil(acos(1.0 - omega_scaled) / Pi * Mq - 0.5);
            assert(M - i_start >= 20);                                          // at least 20 points to do integration
            auto T_i = Vec<double>(M);
            auto T_j = Vec<double>(M);
            for (int i = i_start; i < Mq; i++) {
                double x_i = cos(Pi * (i+0.5) / Mq);
                if (1.0-x_i*x_i < cutoff) continue;                             // neglect points near boundary
                double temp_squareroot2 = std::sqrt(1.0 - (x_i + omega_scaled) * (x_i + omega_scaled));
                double f_i;
                chebyshev_fill_array(x_i + omega_scaled, T_i);
                chebyshev_fill_array(x_i, T_j);           // fill T_j with cos[m * arccos(x)]
                f_i = (fermi_density(es.unscale(x_i), kT, mu) - fermi_density(es.unscale(x_i) + omega, kT, mu))
                     / (omega * temp_squareroot2);
                for (int m1 = 0; m1 < M; m1++) {
                    for (int m2 = 0; m2 < M; m2++) {
                        ret[m1][m2] += T_i[m1] * T_j[m2] * f_i;
                    }
                }
            }
            for (int m1 = 0; m1 < M; m1++) {
                double temp_m1 = 2.0 * Pi * (m1 == 0 ? 1.0 : 2.0) * kernel[m1] / (Mq * es.mag() * es.mag());
                for (int m2 = 0; m2 < M; m2++) {
                    double temp_m2 = temp_m1 * (m2 == 0 ? 1.0 : 2.0) * kernel[m2];
                    ret[m1][m2] *= cx_double(temp_m2, 0.0);
                }
            }
            
        }
        return ret;
    }

    
    Vec<double> moment_transform(Vec<double> const& moments, int Mq) {
        int M = moments.size();
        auto T = Vec<double>(M);
        auto mup = Vec<double>(M);
        auto gamma = Vec<double>(Mq);
        
        auto kernel = jackson_kernel(M);
        for (int m = 0; m < M; m++)
            mup[m] = moments[m] * kernel[m];
        
        // TODO: replace with DCT-III, mup -> gamma (caution, double check FFTW docs)
        for (int i = 0; i < Mq; i++) {
            gamma[i] = 0.0;     // add this line in case some compilers initialize gamma with arbitrary number
            double x_i = cos(Pi * (i+0.5) / Mq);
            chebyshev_fill_array(x_i, T); // T_m(x_i) = cos(m pi (i+1/2) / Mq)
            for (int m = 0; m < M; m++) {
                gamma[i] += (m == 0 ? 1 : 2) * mup[m] * T[m];
            }
        }
        return gamma;
    }
    
    Vec<Vec<cx_double>> moment_transform(Vec<Vec<cx_double>> const& moments, int Mq, Vec<double> const& kernel) {
        int M = moments.size();
        auto T_i = Vec<double>(M);
        auto T_j = Vec<double>(M);
        Vec<Vec<cx_double>> mup(M);
        Vec<Vec<cx_double>> gamma(Mq);
        for (int i = 0; i < M; i++)  mup[i].resize(M, cx_double(0.0,0.0));
        for (int i = 0; i < Mq; i++) gamma[i].resize(Mq, cx_double(0.0,0.0));
        
        for (int m1 = 0; m1 < M; m1++) {
            for (int m2 = 0; m2 < M; m2++) {
                mup[m1][m2] = cx_double((m1 == 0 ? 1.0 : 2.0) * (m2 == 0 ? 1.0 : 2.0)
                                        * kernel[m1] * kernel[m2], 0.0) * moments[m1][m2];
            }
        }
        
        // TODO replace with fftw
        for (int i = 0; i < Mq; i++) {
            double x_i = cos(Pi * (i+0.5) / Mq);
            chebyshev_fill_array(x_i, T_i);
            for (int j = 0; j < Mq; j++) {
                double x_j = cos(Pi * (j+0.5) / Mq);
                chebyshev_fill_array(x_j, T_j);
                for (int m1 = 0; m1 < M; m1++) {
                    for (int m2 = 0; m2 < M; m2++) {
                        gamma[i][j] += (T_i[m1] * T_j[m2]) * mup[m1][m2];
                    }
                }
            }
        }
        return gamma;
    }
    
    double moment_product(Vec<double> const& c, Vec<double> const& mu) {
        int M = c.size();
        double ret = 0;
        for (int i = 0; i < M; i++) {
            ret += c[i]*mu[i];
        }
        return ret;
    }

    cx_double moment_product(Vec<Vec<cx_double>> const& c, Vec<Vec<cx_double>> const& mu) {
        int M1 = c.size();
        int M2 = c[0].size();
        assert(mu.size() == M1);
        assert(mu[0].size() == M2);
        cx_double ret(0.0, 0.0);
        for (int m1 = 0; m1 < M1; m1++) {
            for (int m2 = 0; m2 < M2; m2++) {
                ret += c[m1][m2] * mu[m1][m2];
            }
        }
        return ret;
    }
    
    double density_product(Vec<double> const& gamma, std::function<double(double)> f, EnergyScale es) {
        int Mq = gamma.size();
        double ret = 0.0;
        for (int i = 0; i < Mq; i++) {
            double x_i = cos(Pi * (i+0.5) / Mq);
            ret += gamma[i] * f(es.unscale(x_i));
        }
        return ret / Mq;
    }
    
    void density_function(Vec<double> const& gamma, EnergyScale es, Vec<double>& x, Vec<double>& rho) {
        int Mq = gamma.size();
        x.resize(Mq);
        rho.resize(Mq);
        for (int i = Mq-1; i >= 0; i--) {
            double x_i = cos(Pi * (i+0.5) / Mq);
            x[Mq-1-i] = es.unscale(x_i);
            rho[Mq-1-i] = gamma[i] / (Pi * sqrt(1-x_i*x_i) * es.mag());
        }
    }
    
    void density_function(Vec<Vec<cx_double>> const& gamma, EnergyScale es, Vec<double>& x, Vec<double>& y, Vec<Vec<cx_double>>& rho) {
        int Mq1 = gamma.size();
        int Mq2 = gamma[0].size();
        x.resize(Mq1);
        y.resize(Mq2);
        rho.resize(Mq1);
        for (int i = 0; i < Mq1; i++) {
            rho[i].resize(Mq2);
            for (int j = 0; j < Mq2; j++) {
                rho[i][j] = 0.0;
            }
        }
        for (int i2 = Mq2-1; i2 >= 0; i2--) {
            double x2   = cos(Pi * (i2+0.5) / Mq2);
            y[Mq2-1-i2] = es.unscale(x2);
        }
        for (int i1 = Mq1-1; i1 >= 0; i1--) {
            double x1   = cos(Pi * (i1+0.5) / Mq1);
            x[Mq1-1-i1] = es.unscale(x1);
            for (int i2 = Mq2-1; i2 >= 0; i2--) {
                double x2 = cos(Pi * (i2+0.5) / Mq2);
                rho[Mq1-1-i1][Mq2-1-i2] = gamma[i1][i2] / (Pi * sqrt(1.0 - x1*x1)
                                                          * Pi * sqrt(1.0 - x2*x2) * es.mag() * es.mag());
            }
        }
    }
    
    void integrated_density_function(Vec<double> const& gamma, EnergyScale es, Vec<double>& x, Vec<double>& irho) {
        int Mq = gamma.size();
        x.resize(Mq);
        irho.resize(Mq);
        double acc = 0.0;
        for (int i = Mq-1; i >= 0; i--) {
            double x_i = cos(Pi * (i+0.5) / Mq);
            x[Mq-1-i] = es.unscale(x_i);
            irho[Mq-1-i] = (acc+0.5*gamma[i]) / Mq;
            acc += gamma[i];
        }
    }
    
    double fermi_energy(double x, double kT, double mu) {
        double alpha = (x-mu)/std::abs(kT);
        if (kT < 1e-15 || std::abs(alpha) > 20) {
            return (x < mu) ? (x-mu) : 0.0;
        }
        else {
            return -kT*log(1 + exp(-alpha));
        }
    }
    
    double fermi_density(double x, double kT, double mu) {
        double alpha = (x-mu)/std::abs(kT);
        if (kT < 1e-15 || std::abs(alpha) > 20) {
            return (x < mu) ? 1.0 : 0.0;
        }
        else {
            return 1.0/(exp(alpha)+1.0);
        }
    }
    
    double mu_to_filling(Vec<double> const& gamma, EnergyScale const& es, double kT, double mu) {
        using std::placeholders::_1;
        double n_occ = density_product(gamma, std::bind(fermi_density, _1, kT, mu), es);
        double n_tot = density_product(gamma, [](double x){return 1;}, es);
        return n_occ/n_tot;
    }
    double mu_to_filling(arma::vec const& evals, double kT, double mu) {
        double n_occ = 0;
        double n_tot = evals.size();
        for (double const& x : evals) {
            n_occ += fermi_density(x, kT, mu);
        }
        return n_occ/n_tot;
    }
    
    static double root_solver(std::function<double(double)> f, double lo, double hi) {
        int precision_bits = 30;
        boost::math::tools::eps_tolerance<double> tol(precision_bits);
        boost::uintmax_t max_iter=50;
        auto bds = boost::math::tools::toms748_solve(f, lo, hi, tol, max_iter);
        return 0.5 * (bds.first + bds.second);
    }
    
    double filling_to_mu(Vec<double> const& gamma, EnergyScale const& es, double kT, double filling, double delta_filling) {
        // thermal smearing for faster convergence
        kT = std::max(kT, 0.1*es.mag()/gamma.size());
        double c = kT * std::log(1.0/filling - 1.0);

        auto f1 = [&](double x) { return mu_to_filling(gamma, es, kT, x) - (filling+delta_filling); };
        auto f2 = [&](double x) { return mu_to_filling(gamma, es, kT, x) - (filling-delta_filling); };
        if (delta_filling == 0) {
            return root_solver(f1, es.lo - std::max(c,0.0), es.hi + std::max(-c,0.0));
        }
        else {
            return 0.5 * (root_solver(f1, es.lo - std::max(c,0.0), es.hi + std::max(-c,0.0))
                        + root_solver(f2, es.lo - std::max(c,0.0), es.hi + std::max(-c,0.0)));
        }
    }
    double filling_to_mu(arma::vec const& evals, double kT, double filling) {
        assert(kT > 0 && "filling_to_mu() requires thermal smearing!");
        auto f = [&](double x) { return mu_to_filling(evals, kT, x) - filling; };
        auto minmax = std::minmax_element(evals.begin(), evals.end());
        return root_solver(f, *minmax.first, *minmax.second);
    }
    
    double electronic_grand_energy(Vec<double> const& gamma, EnergyScale const& es, double kT, double mu) {
        using std::placeholders::_1;
        return density_product(gamma, std::bind(fermi_energy, _1, kT, mu), es);
    }
    double electronic_grand_energy(arma::vec const& evals, double kT, double mu) {
        double acc = 0;
        for (double const& x : evals) {
            acc += fermi_energy(x, kT, mu);
        }
        return acc;
    }
    
    double electronic_energy(Vec<double> const& gamma, EnergyScale const& es, double kT, double filling, double mu) {
        double n_tot = density_product(gamma, [](double x){return 1;}, es);
        double n_occ = filling*n_tot;
        return electronic_grand_energy(gamma, es, kT, mu) + mu*n_occ;
    }
    double electronic_energy(arma::vec const& evals, double kT, double filling) {
        // at zero temperature, need special logic to correctly count degenerate eigenvalues
        if (kT == 0) {
            auto evals_sorted = evals;
            std::sort(evals_sorted.begin(), evals_sorted.end());
            int n_occ = int(filling*evals.size() + 0.5);
            assert(n_occ >= 0 && n_occ <= evals.size());
            double acc = 0;
            for (int i = 0; i < n_occ; i++) {
                acc += evals_sorted[i];
            }
            return acc;
        }
        else {
            double n_tot = evals.size();
            double n_occ = filling*n_tot;
            double mu = filling_to_mu(evals, kT, filling);
            return electronic_grand_energy(evals, kT, mu) + mu*n_occ;
        }
    }
    
}
