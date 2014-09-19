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
#include <vector>
#include <armadillo>
#include <chrono>


namespace fkpm {
    typedef std::mt19937 RNG;
    
    template <typename T>
    using Vec = std::vector<T>;
    
    class Timer {
    public:
        std::chrono::time_point<std::chrono::system_clock> t0;
        Timer();
        void reset();
        double measure();
    };
    extern Timer timer[10];
    
    // Sparse matrix in coordinate list format
    template <typename T>
    class SpMatCoo {
    public:
        int n_rows, n_cols;
        Vec<arma::uword> idx;
        Vec<T> val;
        SpMatCoo(int n_rows, int n_cols): n_rows(n_rows), n_cols(n_cols) {}
        void clear() {
            idx.resize(0);
            val.resize(0);
        }
        void add(int i, int j, T v) {
            idx.push_back(i);
            idx.push_back(j);
            val.push_back(v);
        }
        arma::SpMat<T> to_arma() const {
            return arma::SpMat<T>(arma::umat(idx.data(), 2, idx.size()/2), arma::Col<T>(val), n_rows, n_cols);
        }
    };
    
    // Scale eigenvalues within range (-1, +1)
    struct EnergyScale {
        double lo, hi;
        EnergyScale(double lo=0, double hi=0): lo(lo), hi(hi) {}
        double avg() const;
        double mag() const;
        double scale(double x) const;
        double unscale(double x) const;
    };
    
    // Print EnergyScale
    std::ostream& operator<< (std::ostream& stream, EnergyScale const& es);
    
    // Use Lanczos to bound eigenvalues of H, and determine appropriate rescaling
    template<typename T>
    EnergyScale energy_scale(SpMatCoo<T> const& H, double extend, double tolerance);
    
    
    // Used to damp Gibbs oscillations in KPM estimates
    Vec<double> jackson_kernel(int M);
    
    // Chebyshev polynomials T_m(x) evaluated at x
    void chebyshev_fill_array(double x, Vec<double>& ret);
    
    // Coefficients c_m that satisfy f(x) = \sum_m T_m(x) c_m
    Vec<double> expansion_coefficients(int M, int Mq, std::function<double(double)> f, EnergyScale es);
    
    // Calculate \sum c_m mu_m
    double moment_product(Vec<double> const& c, Vec<double> const& mu);
    
    // Transformation of moments from mu to gamma, which corresponds to the density of states
    Vec<double> moment_transform(Vec<double> const& moments, int Mq);
    
    // Calculate \int dx rho(x) f(x)
    double density_product(Vec<double> const& gamma, std::function<double(double)> f, EnergyScale es);
    
    // Density of states rho(x) at Chebyshev points x
    void density_function(Vec<double> const& gamma, EnergyScale es, Vec<double>& x, Vec<double>& rho);
    
    // Density of states \int theta(x-x') rho(x') dx' at Chebyshev points x
    void integrated_density_function(Vec<double> const& gamma, EnergyScale es, Vec<double>& x, Vec<double>& irho);
    
    // Grand potential energy density of an electronic state at x
    double fermi_energy(double x, double kB_T, double mu);
    
    // Fermi function at x
    double fermi_density(double x, double kB_T, double mu);
    
    // "Grand" (fixed mu) free energy at fixed chemical potential mu
    // To get canonical (fixed filling) free energy, add term (mu N_electrons)
    double electronic_grand_energy(Vec<double> const& gamma, EnergyScale const& es, double kB_T, double mu);
    
    // Zero temperature energy at at fixed filling fraction
    double electronic_energy_exact(arma::vec evals, double filling);
    
    // Find chemical potential mu corresponding to given filling fraction (+- delta_filling)
    double filling_to_mu(Vec<double> const& gamma, EnergyScale const& es, double kB_T, double filling, double delta_filling);
    
    // Find filling fraction corresponding to chemical potential
    double mu_to_filling(Vec<double> const& gamma, EnergyScale const& es, double kB_T, double mu);
    
    
    class EngineCx {
    public:
        int n;                 // Rows (columns) of Hamiltonian
        int s;                 // Columns of random matrix
        // TODO: hide?
        EnergyScale es;        // Scaling bounds
        arma::cx_mat R;        // Random vectors
        arma::cx_mat xi;       // Occupied orbitals
        
        EngineCx(int n, int s);
        
        // R has uncorrelated random elements
        virtual void set_R_uncorrelated(RNG& rng);
        
        // R has correlated random elements with mostly orthogonal rows
        virtual void set_R_correlated(Vec<int> const& grouping, RNG& rng);
        
        // R is the identity matrix
        virtual void set_R_identity();
        
        // Set Hamiltonian and energy scale
        virtual void set_H(SpMatCoo<arma::cx_double> const& H, EnergyScale const& es) = 0;
        
        // Chebyshev moments: mu_m = tr T_m(Hs) ~ <R| T_m(Hs) |R>
        virtual Vec<double> moments(int M) = 0;
        
        // Stochastic orbital: xi = \sum_m c_m T_m(Hs) |R>
        virtual void stoch_orbital(Vec<double> const& c) = 0;

        // Approximates B_{ij} ~ Re(xi R^\dagger)_{ij}.
        // Assumes stoch_orbital() has already been called.
        arma::cx_double stoch_element(int i, int j);
    };
    
    std::shared_ptr<EngineCx> mk_engine_cx_CPU(int n, int s);
    std::shared_ptr<EngineCx> mk_engine_cx_cuSPARSE(int n, int s);
    // Fastest EngineCx available
    std::shared_ptr<EngineCx> mk_engine_cx(int n, int s);
}

#endif /* defined(__tibidy__fastkpm__) */
