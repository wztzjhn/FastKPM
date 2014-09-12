//
//  fastkpm.cpp
//  tibidy
//
//  Created by Kipton Barros on 5/22/14.
//
//

#include <cstdlib>
#include <cassert>
#include <algorithm>
#ifdef WITH_TBB
#include <tbb/tbb.h>
#endif
#include "fastkpm.h"

namespace fkpm {
    const double Pi = 3.141592653589793238463;
    
    double EnergyScale::avg() const { return (hi + lo) / 2.0; }
    double EnergyScale::mag() const { return (hi - lo) / 2.0; }
    double EnergyScale::scale(double x) const {
        return (x - avg()) / mag();
    }
    double EnergyScale::unscale(double x) const {
        return x * mag() + avg();
    }
    arma::sp_cx_mat EnergyScale::scale(arma::sp_cx_mat const& H) const {
        int n = H.n_rows;
        auto I = arma::speye<arma::sp_cx_mat>(n, n);
        return (H - I*avg()) / mag();
    }
    
    std::ostream& operator<< (std::ostream& stream, EnergyScale const& es) {
        return stream << "< lo = " << es.lo << " hi = " << es.hi << " >\n";
    }
    
    Vec<double> jacksonKernel(int M) {
        auto ret = Vec<double>(M);
        double Mp = M+1.0;
        for (int m = 0; m < M; m++) {
            ret[m] = (1.0/Mp)*((Mp-m)*cos(Pi*m/Mp) + sin(Pi*m/Mp)/tan(Pi/Mp));
        }
        return ret;
    }
    
    
    void chebyshevFillArray(double x, Vec<double>& ret) {
        if (ret.size() > 0)
            ret[0] = 1.0;
        if (ret.size() > 1)
            ret[1] = x;
        for (int m = 2; m < ret.size(); m++) {
            ret[m] = 2*x*ret[m-1] - ret[m-2];
        }
    }
    
    Vec<double> expansionCoefficients(int M, int Mq, std::function<double(double)> f, EnergyScale es) {
        // TODO: replace with DCT-II, f -> fp
        auto fp = Vec<double>(Mq, 0.0);
        auto T = Vec<double>(M);
        for (int i = 0; i < Mq; i++) {
            double x_i = cos(Pi * (i+0.5) / Mq);
            chebyshevFillArray(x_i, T);
            for (int m = 0; m < M; m++) {
                fp[m] += f(es.unscale(x_i)) * T[m];
            }
        }
        auto kernel = jacksonKernel(M);
        auto ret = Vec<double>(M);
        for (int m = 0; m < M; m++) {
            ret[m] = (m == 0 ? 1.0 : 2.0) * kernel[m] * fp[m] / Mq;
        }
        return ret;
    }
    
    Vec<double> momentTransform(Vec<double> const& moments, int Mq) {
        int M = moments.size();
        auto T = Vec<double>(M);
        auto mup = Vec<double>(M);
        auto gamma = Vec<double>(Mq);
        
        auto kernel = jacksonKernel(M);
        for (int m = 0; m < M; m++)
            mup[m] = moments[m] * kernel[m];
        
        // TODO: replace with DCT-III, mup -> gamma
        for (int i = 0; i < Mq; i++) {
            double x_i = cos(Pi * (i+0.5) / Mq);
            chebyshevFillArray(x_i, T); // T_m(x_i) = cos(m pi (i+1/2) / Mq)
            for (int m = 0; m < M; m++) {
                gamma[i] += (m == 0 ? 1 : 2) * mup[m] * T[m];
            }
        }
        return gamma;
    }
    
    double densityProduct(Vec<double> const& gamma, std::function<double(double)> f, EnergyScale es) {
        int Mq = gamma.size();
        double ret = 0.0;
        for (int i = 0; i < Mq; i++) {
            double x_i = cos(Pi * (i+0.5) / Mq);
            ret += gamma[i] * f(es.unscale(x_i));
        }
        return ret / Mq;
    }
    
    void densityFunction(Vec<double> const& gamma, EnergyScale es, Vec<double>& x, Vec<double>& rho) {
        int Mq = gamma.size();
        x.resize(Mq);
        rho.resize(Mq);
        for (int i = Mq-1; i >= 0; i--) {
            double x_i = cos(Pi * (i+0.5) / Mq);
            x[Mq-1-i] = es.unscale(x_i);
            rho[Mq-1-i] = gamma[i] / (Pi * sqrt(1-x_i*x_i) * es.mag());
        }
    }
    
    void integratedDensityFunction(Vec<double> const& gamma, EnergyScale es, Vec<double>& x, Vec<double>& irho) {
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
    
    EnergyScale energyScale(arma::sp_cx_mat const& H, double extra, double tolerance) {
        arma::cx_vec eigval;
        eigs_gen(eigval, H, 1, "sr", tolerance);
        double eig_min = eigval(0).real();
        eigs_gen(eigval, H, 1, "lr", tolerance);
        double eig_max = eigval(0).real();
        double slack = extra * (eig_max - eig_min);
        return {eig_min-slack, eig_max+slack};
    }
    
    double fermiEnergy(double x, double kB_T, double mu) {
        double alpha = (x-mu)/abs(kB_T);
        if (kB_T == 0.0 || abs(alpha) > 20) {
            return (x < mu) ? (x-mu) : 0.0;
        }
        else {
            return -kB_T*log(1 + exp(-alpha));
        }
    }
    
    double fermiDensity(double x, double kB_T, double mu) {
        double alpha = (x-mu)/abs(kB_T);
        if (kB_T == 0.0 || abs(alpha) > 20) {
            return (x < mu) ? 1.0 : 0.0;
        }
        else {
            return 1.0/(exp(alpha)+1.0);
        }
    }
    
    double electronicGrandEnergy(Vec<double> const& gamma, EnergyScale const& es, double kB_T, double mu) {
        using std::placeholders::_1;
        return densityProduct(gamma, std::bind(fermiEnergy, _1, kB_T, mu), es);
    }
    
    // TODO: use fermiEnergy() and be smarter about filling
    double electronicEnergyExact(arma::vec evals, double filling) {
        std::sort(evals.begin(), evals.end());
        int numFilled = (int)(filling*evals.size());
        double acc = 0;
        for (int i = 0; i < numFilled; i++) {
            acc += evals[i];
        }
        return acc;
    }
    
    double fillingToMu(Vec<double> const& gamma, EnergyScale const& es, double kB_T, double filling, double delta_filling) {
        Vec<double> x, irho;
        integratedDensityFunction(gamma, es, x, irho);
        double numTot = densityProduct(gamma, [](double x){return 1;}, es);
        assert(0 <= filling && filling <= 1.0);
        double n1 = numTot * std::max(filling - delta_filling, 0.0);
        double n2 = numTot * std::min(filling + delta_filling, numTot);
        // TODO: generalize to finite temperature
        //assert(kB_T == 0);
        int i1 = std::find_if(irho.begin(), irho.end(), [&](double x){return x > n1;}) - irho.begin();
        int i2 = std::find_if(irho.begin(), irho.end(), [&](double x){return x > n2;}) - irho.begin();
        assert(0 <= i1 && i1 <= irho.size()-1);
        assert(0 <= i2 && i2 <= irho.size()-1);
        return (x[i1] + x[i1-1] + x[i2] + x[i2-1]) / 4.0;
    }
    
    double muToFilling(Vec<double> const& gamma, EnergyScale const& es, double kB_T, double mu) {
        using std::placeholders::_1;
        double numOcc = densityProduct(gamma, std::bind(fermiDensity, _1, kB_T, mu), es);
        double numTot = densityProduct(gamma, [](double x){return 1;}, es);
        return numOcc/numTot;
    }

    arma::sp_cx_mat buildSparseCx(Vec<arma::uword>& idx, Vec<arma::cx_double>& val, int n_rows, int n_cols) {
#ifdef WITH_TBB
        struct MatrixElem {
            arma::uword i;
            arma::uword j;
            arma::cx_double val;
        };
        int n_elems = val.size();
        Vec<MatrixElem> elems(n_elems);
        for (int i = 0; i < n_elems; i++) {
            elems[i] = { idx[2*i+0], idx[2*i+1], val[i] };
        }
        auto cmp = [&](MatrixElem const& x, MatrixElem const& y) {
            return (x.j == y.j) ? (x.i < y.i) : (x.j < y.j);
        };
        tbb::parallel_sort(elems.begin(), elems.end(), cmp);
        
        for (int i = 0; i < n_elems; i++) {
            idx[2*i+0] = elems[i].i;
            idx[2*i+1] = elems[i].j;
            val[i] = elems[i].val;
        }
        return arma::sp_cx_mat(arma::umat(idx.data(), 2, idx.size()/2), arma::cx_vec(val), n_rows, n_cols, false);
#else
        return arma::sp_cx_mat(arma::umat(idx.data(), 2, idx.size()/2), arma::cx_vec(val), n_rows, n_cols);
#endif
    }
    
    arma::cx_double randomPhaseCx(RNG& rng) {
        std::uniform_int_distribution<uint32_t> dist4(0,3);
        switch (dist4(rng)) {
            case 0: return {+1,  0};
            case 1: return { 0, +1};
            case 2: return {-1,  0};
            case 3: return { 0, -1};
        }
        assert(false);
    }
    
    arma::cx_double randomNormalCx(RNG& rng) {
        std::normal_distribution<double> normalDist;
        return arma::cx_double(normalDist(rng), normalDist(rng));
    }
    
    void uncorrelatedVectorsCx(RNG& rng, arma::cx_mat& R) {
        double x = 1.0 / sqrt(R.n_cols);
        for (int i = 0; i < R.n_rows; i++) {
            for (int j = 0; j < R.n_cols; j++) {
                R(i, j) = randomPhaseCx(rng) * x;
            }
        }
    }
    
    void correlatedVectorsCx(Vec<int> const& grouping, RNG& rng, arma::cx_mat& R) {
        assert(R.n_rows == grouping.size());
        R.fill(0.0);
        for (int i = 0; i < R.n_rows; i++) {
            int g = grouping[i];
            assert(0 <= g && g < R.n_cols);
            R(i, g) = randomPhaseCx(rng);
        }
    }
    
    void allVectorsCx(arma::cx_mat& R) {
        assert(R.n_rows == R.n_cols);
        R.fill(0.0);
        for (int i = 0; i < R.n_rows; i++) {
            R(i, i) = 1.0;
        }
    }
    
    EngineCx::EngineCx(int n, int s): n(n), s(s) {
        R = arma::cx_mat(n, s);
        xi = arma::cx_mat(n, s);
    }
    
    void EngineCx::setHamiltonian(arma::sp_cx_mat const& H, EnergyScale const& es) {
        this-> es = es;
        Hs = es.scale(H);
        dE_dH = Hs;
    }
    
    double EngineCx::trace(Vec<double> const& c) {
        int M = c.size();
        auto mu = moments(M);
        double ret = 0;
        for (int i = 0; i < M; i++) {
            ret += c[i]*mu[i];
        }
        return ret;
    }
    
    double EngineCx::trace(Vec<double> const& c, arma::sp_cx_mat const& A) {
        return arma::cdot(R, A*occupiedOrbital(c)).real();
    }
    
    arma::sp_cx_mat& EngineCx::deriv(Vec<double> const& c) {
        auto xi = occupiedOrbital(c);
        dE_dH = Hs;
        for (int j = 0; j < n; j++) {
            for (int iter = Hs.col_ptrs[j]; iter < Hs.col_ptrs[j+1]; iter++) {
                int i = Hs.row_indices[iter];
                dE_dH(i, j) = arma::cdot(R.row(j), xi.row(i)).real();
            }
        }

        return dE_dH;
    }
    
    std::shared_ptr<EngineCx> mkEngineCx(int n, int s) {
        std::shared_ptr<EngineCx> ret;
        ret = mkEngineCx_cuSPARSE(n, s);
        if (ret == nullptr)
            ret = mkEngineCx_CPU(n, s);
        return ret;
    }
}
