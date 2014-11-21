#ifndef __fastkpm__
#define __fastkpm__

#include <random>
#include <vector>
#include <chrono>
#include <memory>
#include <armadillo>

#ifdef WITH_TBB
#include <tbb/tbb.h>
#endif


namespace fkpm {
    typedef std::mt19937 RNG;
    
    template <typename T>
    using Vec = std::vector<T>;
    
    template <typename S, typename T>
    void copy_vec(Vec<S> const& src_vec, Vec<T>& dst_vec) {
        dst_vec.resize(src_vec.size());
        std::copy(src_vec.begin(), src_vec.end(), dst_vec.begin());
    };

    typedef std::complex<float>  cx_float;
    typedef std::complex<double> cx_double;
    
    // complex conjugation that preserves real values
    template <typename T>   T conj(T x);
    template <>             inline float conj(float x) { return x; }
    template <>             inline double conj(double x) { return x; }
    template <>             inline cx_float conj(cx_float x) { return std::conj(x); }
    template <>             inline cx_double conj(cx_double x) { return std::conj(x); }
    
    constexpr double Pi = 3.141592653589793238463;
    constexpr cx_double I(0, 1);
    
#ifdef WITH_TBB
    inline void parallel_for(size_t start, size_t end, std::function<void(size_t)> fn) {
        tbb::parallel_for(size_t(start),size_t(end), fn);
    }
#else
    inline void parallel_for(size_t start, size_t end, std::function<void(size_t)> fn) {
        for (int i = start; i < end; i++) { fn(i); }
    }
#endif
    
    
    // -- spmat.cpp ------------------------------------------------------------------------
    
    // Sparse matrix in Coordinate list format
    template <typename T>
    class SpMatElems {
    public:
        Vec<int> row_idx, col_idx;
        Vec<T> val;
        int size() const;
        void clear();
        void add(int i, int j, T v);
    };
    // Sparse matrix in Compressed Sparse Row format
    template <typename T>
    class SpMatCsr {
    private:
        Vec<Vec<int>> sorted_ptr;
    public:
        int n_rows = 0, n_cols = 0;
        Vec<int> row_idx, col_idx, row_ptr;
        Vec<T> val;
        SpMatCsr();
        SpMatCsr(int n_rows, int n_cols, SpMatElems<T> const& that);
        int size() const;
        void clear();
        int find_index(int i, int j) const;
        T& operator()(int i, int j);
        T const& operator()(int i, int j) const;
        void build(int n_rows, int n_cols, SpMatElems<T> const& elems);
        void zeros();
        void symmetrize();
        arma::SpMat<T> to_arma() const;
        arma::Mat<T> to_arma_dense() const;
        
        template<typename S>
        SpMatCsr<T>& operator=(SpMatCsr<S> const& that) {
            n_rows = that.n_rows;
            n_cols = that.n_cols;
            copy_vec(that.row_idx, row_idx);
            copy_vec(that.col_idx, col_idx);
            copy_vec(that.row_ptr, row_ptr);
            copy_vec(that.val, val);
            return *this;
        }
    };
    
    
    // -- fastkpm.cpp ------------------------------------------------------------------------
    
    // Scale eigenvalues within range (-1, +1)
    struct EnergyScale {
        double lo, hi;
        double avg() const { return (hi + lo) / 2.0; }
        double mag() const { return (hi - lo) / 2.0; }
        double scale(double x) const { return (x - avg()) / mag(); }
        double unscale(double x) const { return x * mag() + avg(); }
        friend std::ostream& operator<< (std::ostream& stream, EnergyScale const& es);
    };
    
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
    double fermi_energy(double x, double kT, double mu);
    
    // Fermi function at x
    double fermi_density(double x, double kT, double mu);
    
    // Filling fraction corresponding to chemical potential
    double mu_to_filling(Vec<double> const& gamma, EnergyScale const& es, double kT, double mu);
    double mu_to_filling(arma::vec const& evals, double kT, double mu);
    
    // Chemical potential mu corresponding to given filling fraction (+- delta_filling)
    double filling_to_mu(Vec<double> const& gamma, EnergyScale const& es, double kT, double filling, double delta_filling);
    double filling_to_mu(arma::vec const& evals, double kT, double filling);
    
    // "Grand" free energy at fixed chemical potential mu
    double electronic_grand_energy(Vec<double> const& gamma, EnergyScale const& es, double kT, double mu);
    double electronic_grand_energy(arma::vec const& evals, double kT, double mu);
    
    // "Canonical" free energy at fixed filling fraction
    // mu should have been obtained from filling_to_mu() function
    double electronic_energy(Vec<double> const& gamma, EnergyScale const& es, double kT, double filling, double mu);
    double electronic_energy(arma::vec const& evals, double kT, double filling);
    
    
    // -- engine*.cpp ------------------------------------------------------------------------
    
    template <typename T>
    class Engine {
    public:
        arma::Mat<T> R;        // Random vectors
        
        // Uncorrelated random elements
        void set_R_uncorrelated(int n, int s, RNG& rng);
        
        // Correlated random elements with mostly orthogonal rows
        void set_R_correlated(Vec<int> const& groups, RNG& rng);
        
        // Identity matrix
        void set_R_identity(int n);
        
        // Transfer R matrix to device
        virtual void transfer_R() {}
        
        // Set Hamiltonian and energy scale
        virtual void set_H(SpMatCsr<T> const& H, EnergyScale const& es) = 0;
        
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
    std::shared_ptr<Engine<T>> mk_engine_cuSPARSE(int device);
    
    // Fastest engine available
    template <typename T>
    std::shared_ptr<Engine<T>> mk_engine();
    
    
    // -- timer.cpp ------------------------------------------------------------------------
    
    class Timer {
    public:
        std::chrono::time_point<std::chrono::system_clock> t0;
        Timer();
        void reset();
        double measure(); // elapsed time in seconds
    };
    extern Timer timer[10];
}

#endif /* defined(__fastkpm__) */
