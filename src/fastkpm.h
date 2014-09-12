//
//  fastkpm.h
//  tibidy
//
//  Created by Kipton Barros on 5/22/14.
//
//

#ifndef __tibidy__fastkpm__
#define __tibidy__fastkpm__

#include <random>
#include <memory>
#include <vector>
#include <armadillo>
#include "timer.h"


namespace fkpm {
    typedef std::mt19937 RNG;
    
    template <typename T>
    using Vec = std::vector<T>;
    
    template <typename T>
    arma::Mat<T> sparseToDense(arma::SpMat<T> that) {
        int m = that.n_rows;
        return arma::eye<arma::Mat<T>>(m, m) * that;
    }
    
    template <typename T>
    class MatrixBuilder {
    public:
        Vec<arma::uword> idx;
        Vec<arma::cx_double> val;
        void add(int i, int j, T v) {
            idx.push_back(i);
            idx.push_back(j);
            val.push_back(v);
        }
        arma::SpMat<T> build(int n_rows, int n_cols) {
            return arma::SpMat<T>(arma::umat(idx.data(), 2, idx.size()/2), arma::cx_vec(val), n_rows, n_cols);
        }
    };
    
    struct EnergyScale {
        double lo, hi;
        EnergyScale(double lo=0, double hi=0): lo(lo), hi(hi) {}
        double avg() const;
        double mag() const;
        double scale(double x) const;
        double unscale(double x) const;
        arma::sp_cx_mat scale(arma::sp_cx_mat const& H) const;
    };
    std::ostream& operator<< (std::ostream& stream, EnergyScale const& es);
    
    
    // Used to damp Gibbs oscillations in KPM estimates
    Vec<double> jacksonKernel(int M);
    
    // Chebyshev polynomials T_m(x) evaluated at x
    void chebyshevFillArray(double x, Vec<double>& ret);
    
    // Coefficients c_m that satisfy f(x) = \sum_m T_m(x) c_m
    Vec<double> expansionCoefficients(int M, int Mq, std::function<double(double)> f, EnergyScale es);
    
    // Transformation of moments from mu to gamma, which corresponds to the density of states
    Vec<double> momentTransform(Vec<double> const& moments, int Mq);
    
    // Calculate \int dx rho(x) f(x)
    double densityProduct(Vec<double> const& gamma, std::function<double(double)> f, EnergyScale es);
    
    // Density of states rho(x) at Chebyshev points x
    void densityFunction(Vec<double> const& gamma, EnergyScale es, Vec<double>& x, Vec<double>& rho);
    
    // Density of states \int theta(x-x') rho(x') dx' at Chebyshev points x
    void integratedDensityFunction(Vec<double> const& gamma, EnergyScale es, Vec<double>& x, Vec<double>& irho);
    
    // Use Lanczos to bound eigenvalues of H, and determine appropriate rescale scaling for KPM
    EnergyScale energyScale(arma::sp_cx_mat const& H, double extend, double tolerance);
    
    // Grand potential energy density of an electronic state at x
    double fermiEnergy(double x, double kB_T, double mu);
    
    // Fermi function at x
    double fermiDensity(double x, double kB_T, double mu);
    
    // "Grand" (fixed mu) free energy at fixed chemical potential mu
    // To get canonical (fixed filling) free energy, add term (mu N_electrons)
    double electronicGrandEnergy(Vec<double> const& gamma, EnergyScale const& es, double kB_T, double mu);
    
    // Zero temperature energy at at fixed filling fraction
    double electronicEnergyExact(arma::vec evals, double filling);
    
    // Find chemical potential mu corresponding to given filling fraction (+- delta_filling)
    double fillingToMu(Vec<double> const& gamma, EnergyScale const& es, double kB_T, double filling, double delta_filling);
    
    // Find filling fraction corresponding to chemical potential
    double muToFilling(Vec<double> const& gamma, EnergyScale const& es, double kB_T, double mu);
    
    // Builds sparse Armadillo matrix efficiently
    arma::sp_cx_mat buildSparseCx(Vec<arma::uword> & idx, Vec<arma::cx_double> & val, int n_rows, int n_cols);
    
    // Random number uniformly distributed in {1, i, -1, -i}
    arma::cx_double randomPhaseCx(RNG& rng);
    
    // Random real and imaginary components
    arma::cx_double randomNormalCx(RNG& rng);
    
    // Vectors with uncorrelated random elements
    void uncorrelatedVectorsCx(RNG& rng, arma::cx_mat& R);
    
    // Vectors with correlated random elements such that rows are mostly orthogonal
    void correlatedVectorsCx(Vec<int> const& grouping, RNG& rng, arma::cx_mat& R);
    
    // Identity matrix
    void allVectorsCx(arma::cx_mat& R);
    
    class EngineCx {
    public:
        int n;                 // Rows (columns) of Hamiltonian
        int s;                 // Columns of random matrix
        EnergyScale es;        // Scaling bounds
        arma::sp_cx_mat Hs;    // Scaled Hamiltonian
        arma::cx_mat R;        // Random vectors
        arma::cx_mat xi;       // Occupied orbitals
        arma::sp_cx_mat dE_dH; // Grand free energy matrix derivative
        
        EngineCx(int n, int s);
        
        // Set Hamiltonian and energy scale
        virtual void setHamiltonian(arma::sp_cx_mat const& H, EnergyScale const& es);
        
        // Chebyshev moments: mu_m = <R| T_m(Hs) |R>
        virtual Vec<double> moments(int M) = 0;
        
        // Stochastic orbital: \sum_m c_m T_m(Hs) |R>
        virtual arma::cx_mat& occupiedOrbital(Vec<double> const& c) = 0;
        
        // Approximate trace: <R| \sum c_m T_m(Hs) |R>
        virtual double trace(Vec<double> const& c);
        
        // Approximate trace: <R| A \sum c_m T_m(Hs) |R>
        virtual double trace(Vec<double> const& c, arma::sp_cx_mat const& A);
        
        // Re(\sum_m c_m T_m(Hs) |R><R|) at nonzero H_ij
        // When f(x) = \sum_m c_m T_m(x) is the Fermi function, the return value is the
        // free energy matrix derivative dE/dH.
        virtual arma::sp_cx_mat& deriv(Vec<double> const& c);
    };
    
    std::shared_ptr<EngineCx> mkEngineCx_CPU(int n, int s);
    std::shared_ptr<EngineCx> mkEngineCx_ViennaCL(int n, int s);
    std::shared_ptr<EngineCx> mkEngineCx_cuSPARSE(int n, int s);
    
    // Fastest EngineCx available
    std::shared_ptr<EngineCx> mkEngineCx(int n, int s);
}

#endif /* defined(__tibidy__fastkpm__) */
