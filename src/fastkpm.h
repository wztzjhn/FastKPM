#ifndef __fastkpm__
#define __fastkpm__

#include <random>
#include <vector>
#include <chrono>
#include <memory>
#include <armadillo>

#include "spmat.h"


namespace fkpm {
    typedef std::mt19937 RNG;
    
    template <typename T>
    using Vec = std::vector<T>;
    
    typedef std::complex<double> cx_double;
    typedef std::complex<float>  cx_float;
    
    class Timer {
    public:
        std::chrono::time_point<std::chrono::system_clock> t0;
        Timer();
        void reset();
        double measure();
    };
    extern Timer timer[10];
    
    // Scale eigenvalues within range (-1, +1)
    struct EnergyScale {
        double lo, hi;
        double avg() const { return (hi + lo) / 2.0; }
        double mag() const { return (hi - lo) / 2.0; }
        double scale(double x) const { return (x - avg()) / mag(); }
        double unscale(double x) const { return x * mag() + avg(); }
    };
    
    // Print EnergyScale
    std::ostream& operator<< (std::ostream& stream, EnergyScale const& es);
    
    // Use Lanczos to bound eigenvalues of H, and determine appropriate rescaling
    template <typename T>
    EnergyScale energy_scale(SpMatCsr<T> const& H, double extend, double tolerance);
    
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
    
    // Filling fraction corresponding to chemical potential
    double mu_to_filling(Vec<double> const& gamma, EnergyScale const& es, double kB_T, double mu);
    double mu_to_filling(arma::vec const& evals, double kB_T, double mu);
    
    // Chemical potential mu corresponding to given filling fraction (+- delta_filling)
    double filling_to_mu(Vec<double> const& gamma, EnergyScale const& es, double kB_T, double filling, double delta_filling);
    double filling_to_mu(arma::vec const& evals, double kB_T, double filling);
    
    // "Grand" free energy at fixed chemical potential mu
    double electronic_grand_energy(Vec<double> const& gamma, EnergyScale const& es, double kB_T, double mu);
    double electronic_grand_energy(arma::vec const& evals, double kB_T, double mu);
    
    // "Canonical" free energy at fixed filling fraction
    // mu should have been obtained from filling_to_mu() function
    double electronic_energy(Vec<double> const& gamma, EnergyScale const& es, double kB_T, double filling, double mu);
    double electronic_energy(arma::vec const& evals, double kB_T, double filling);
    
    
    template <typename T>
    class Engine {
    public:
        EnergyScale es;        // Scaling bounds
        SpMatCsr<T> Hs;        // Scaled Hamiltonian
        arma::Mat<T> R;        // Random vectors
        
        // Uncorrelated random elements
        void set_R_uncorrelated(int n, int s, RNG& rng);
        
        // Correlated random elements with mostly orthogonal rows
        void set_R_correlated(Vec<int> const& groups, RNG& rng);
        
        // Identity matrix
        void set_R_identity(int n);
        
        // Set Hamiltonian and energy scale
        void set_H(SpMatCsr<T> const& H, EnergyScale const& es);
        
        // Transfer R matrix to device
        virtual void transfer_R() {}
        
        // Transfer H matrix to device
        virtual void transfer_H() {}
        
        // Chebyshev moments: mu_m = tr T_m(Hs) ~ tr R^\dagger T_m(Hs) R
        virtual Vec<double> moments(int M) = 0;
        
        // Approximates D ~ (xi R^\dagger + R xi^\dagger)/2 where xi = D R
        // and D ~ (\sum_m c_m T_m(Hs))R
        virtual void stoch_matrix(Vec<double> const& c, SpMatCsr<T>& D) = 0;
        
        // Approximates D ~ (d/dH^T) tr g where tr g ~ tr R^\dagger g R,
        // g~\sum_m c_m T_m(Hs) and coefficients c_m chosen such that
        // dg(x)/dx = D(x).
        // REQUIREMENT: moments() must have been called previously.
        virtual void autodiff_matrix(Vec<double> const& c, SpMatCsr<T>& D) = 0;
    };
    
    // CPU engine
    template <typename T>
    std::shared_ptr<Engine<T>> mk_engine_cpu();

    // CuSPARSE engine
    template <typename T>
    std::shared_ptr<Engine<T>> mk_engine_cuSPARSE();
    
    // Fastest engine available
    template <typename T>
    std::shared_ptr<Engine<T>> mk_engine();
    std::shared_ptr<Engine<double>> mk_engine_re();
    std::shared_ptr<Engine<cx_double>> mk_engine_cx();
}

#endif /* defined(__fastkpm__) */
