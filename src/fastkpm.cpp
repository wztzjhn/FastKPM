//
//  fastkpm.cpp
//  tibidy
//
//  Created by Kipton Barros on 5/22/14.
//
//

// #include <cstdlib>
#include <cassert>
#include <algorithm>
#ifdef WITH_TBB
#include <tbb/tbb.h>
#endif
#include "fastkpm.h"


namespace fkpm {
    static const double Pi = 3.141592653589793238463;
    
    std::ostream& operator<< (std::ostream& stream, EnergyScale const& es) {
        return stream << "< lo = " << es.lo << " hi = " << es.hi << " >\n";
    }
    
    // TODO: fix Armadillo's eigs_sym
    template <>
    EnergyScale energy_scale(SpMatCsr<double> const& H, double extra, double tolerance) {
        arma::sp_mat H_a_re = H.to_arma();
        arma::sp_mat H_a_im = H_a_re;
        H_a_im.zeros();
        arma::sp_cx_mat H_a(H_a_re, H_a_im);
        arma::cx_vec eigval;
        arma::eigs_gen(eigval, H_a, 1, "sr", tolerance);
        double eig_min = eigval(0).real();
        arma::eigs_gen(eigval, H_a, 1, "lr", tolerance);
        double eig_max = eigval(0).real();
        double slack = extra * (eig_max - eig_min);
        return {eig_min-slack, eig_max+slack};
    }
    
    template <>
    EnergyScale energy_scale(SpMatCsr<cx_double> const& H, double extra, double tolerance) {
        auto H_a = H.to_arma();
        arma::cx_vec eigval;
        arma::eigs_gen(eigval, H_a, 1, "sr", tolerance);
        double eig_min = eigval(0).real();
        arma::eigs_gen(eigval, H_a, 1, "lr", tolerance);
        double eig_max = eigval(0).real();
        double slack = extra * (eig_max - eig_min);
        return {eig_min-slack, eig_max+slack};
    }
    
    Vec<double> jackson_kernel(int M) {
        auto ret = Vec<double>(M);
        double Mp = M+1.0;
        for (int m = 0; m < M; m++) {
            ret[m] = (1.0/Mp)*((Mp-m)*cos(Pi*m/Mp) + sin(Pi*m/Mp)/tan(Pi/Mp));
        }
        return ret;
    }
    
    
    void chebyshev_fill_array(double x, Vec<double>& ret) {
        if (ret.size() > 0)
            ret[0] = 1.0;
        if (ret.size() > 1)
            ret[1] = x;
        for (int m = 2; m < ret.size(); m++) {
            ret[m] = 2*x*ret[m-1] - ret[m-2];
        }
    }
    
    Vec<double> expansion_coefficients(int M, int Mq, std::function<double(double)> f, EnergyScale es) {
        // TODO: replace with DCT-II, f -> fp
        auto fp = Vec<double>(Mq, 0.0);
        auto T = Vec<double>(M);
        for (int i = 0; i < Mq; i++) {
            double x_i = cos(Pi * (i+0.5) / Mq);
            chebyshev_fill_array(x_i, T);
            for (int m = 0; m < M; m++) {
                fp[m] += f(es.unscale(x_i)) * T[m];
            }
        }
        auto kernel = jackson_kernel(M);
        auto ret = Vec<double>(M);
        for (int m = 0; m < M; m++) {
            ret[m] = (m == 0 ? 1.0 : 2.0) * kernel[m] * fp[m] / Mq;
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
        
        // TODO: replace with DCT-III, mup -> gamma
        for (int i = 0; i < Mq; i++) {
            double x_i = cos(Pi * (i+0.5) / Mq);
            chebyshev_fill_array(x_i, T); // T_m(x_i) = cos(m pi (i+1/2) / Mq)
            for (int m = 0; m < M; m++) {
                gamma[i] += (m == 0 ? 1 : 2) * mup[m] * T[m];
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
    
    double fermi_energy(double x, double kB_T, double mu) {
        double alpha = (x-mu)/std::abs(kB_T);
        if (kB_T == 0.0 || std::abs(alpha) > 20) {
            return (x < mu) ? (x-mu) : 0.0;
        }
        else {
            return -kB_T*log(1 + exp(-alpha));
        }
    }
    
    double fermi_density(double x, double kB_T, double mu) {
        double alpha = (x-mu)/std::abs(kB_T);
        if (kB_T == 0.0 || std::abs(alpha) > 20) {
            return (x < mu) ? 1.0 : 0.0;
        }
        else {
            return 1.0/(exp(alpha)+1.0);
        }
    }

    double mu_to_filling(Vec<double> const& gamma, EnergyScale const& es, double kB_T, double mu) {
        using std::placeholders::_1;
        double n_occ = density_product(gamma, std::bind(fermi_density, _1, kB_T, mu), es);
        double n_tot = density_product(gamma, [](double x){return 1;}, es);
        return n_occ/n_tot;
    }
    double mu_to_filling(arma::vec const& evals, double kB_T, double mu) {
        double n_occ = 0;
        double n_tot = evals.size();
        for (double const& x : evals) {
            n_occ += fermi_density(x, kB_T, mu);
        }
        return n_occ/n_tot;
    }
    
    // TODO: use modified newton method at finite temperature.
    double filling_to_mu(Vec<double> const& gamma, EnergyScale const& es, double kB_T, double filling, double delta_filling) {
        assert(kB_T == 0);
        assert(gamma.size() >= 3);
        assert(0 <= filling && filling <= 1.0);
        assert(delta_filling >= 0);
        Vec<double> x, irho;
        integrated_density_function(gamma, es, x, irho);
        double n_tot = density_product(gamma, [](double x){return 1;}, es);
        double n1 = n_tot * std::max(filling - delta_filling, 0.0);
        double n2 = n_tot * std::min(filling + delta_filling, 1.0);
        int i1 = std::find_if(irho.begin(), irho.end(), [&](double x){return x > n1;}) - irho.begin();
        int i2 = std::find_if(irho.begin(), irho.end(), [&](double x){return x > n2;}) - irho.begin();
        i1 = std::min<int>(std::max(i1, 1), irho.size()-1);
        i2 = std::min<int>(std::max(i2, 1), irho.size()-1);
        return (x[i1] + x[i1-1] + x[i2] + x[i2-1]) / 4.0;
    }
    double filling_to_mu(arma::vec const& evals, double kB_T, double filling) {
        assert(kB_T > 0 && "filling_to_mu() requires thermal smearing");
        std::abort();
//        auto evals_sorted = evals;
//        std::sort(evals_sorted.begin(), evals_sorted.end());
//        int n_occ = int(filling*evals.size() + 0.5);
//        assert(n_occ >= 1 && n_occ < evals.size());
//        return 0.5 * (evals_sorted[n_occ-1] + evals_sorted[n_occ]);
    }
    
    double electronic_grand_energy(Vec<double> const& gamma, EnergyScale const& es, double kB_T, double mu) {
        using std::placeholders::_1;
        return density_product(gamma, std::bind(fermi_energy, _1, kB_T, mu), es);
    }
    double electronic_grand_energy(arma::vec const& evals, double kB_T, double mu) {
        double acc = 0;
        for (double const& x : evals) {
            acc += fermi_energy(x, kB_T, mu);
        }
        return acc;
    }
    
    double electronic_energy(Vec<double> const& gamma, EnergyScale const& es, double kB_T, double filling, double mu) {
        double n_tot = density_product(gamma, [](double x){return 1;}, es);
        double n_occ = filling*n_tot;
        return electronic_grand_energy(gamma, es, kB_T, mu) + mu*n_occ;
    }
    double electronic_energy(arma::vec const& evals, double kB_T, double filling) {
        // at zero temperature, need special logic to correctly count degenerate eigenvalues
        if (kB_T == 0) {
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
            double mu = filling_to_mu(evals, kB_T, filling);
            return electronic_grand_energy(evals, kB_T, mu) + mu*n_occ;
        }
    }
}
