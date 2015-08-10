#ifndef __fastkpm__
#define __fastkpm__

#include <random>
#include <vector>
#include <chrono>
#include <memory>
#include <string>
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
    
    constexpr double Pi = 3.141592653589793238463;
    
    typedef std::complex<float>  cx_float;
    typedef std::complex<double> cx_double;
    
    // complex conjugation that preserves real values
    inline float     conj(float x)     { return x; }
    inline double    conj(double x)    { return x; }
    inline cx_float  conj(cx_float x)  { return std::conj(x); }
    inline cx_double conj(cx_double x) { return std::conj(x); }
    
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
    
    // Blocks of matrix elements in unsorted order
    template <typename T>
    class SpMatElems {
    public:
        int n_rows, n_cols, b_len;
        Vec<int> row_idx, col_idx;
        Vec<T> val;
        SpMatElems(int n_rows, int n_cols, int b_len);
        int n_blocks() const;
        void clear();
        void add(int i, int j, T const* v);
    };
    // Sparse matrix in block compressed sparse row (BSR) format. Each dense block is stored
    // in column-major order (the BLAS and LAPACK standard).
    template <typename T>
    class SpMatBsr {
    private:
        Vec<Vec<int>> sorted_ptr_bin;
        Vec<int> sorted_ptr;
    public:
        int n_rows = 0, n_cols = 0, b_len = 0;
        Vec<int> row_idx, col_idx, row_ptr;
        Vec<T> val;
        SpMatBsr();
        SpMatBsr(SpMatElems<T> const& elems);
        void build(SpMatElems<T> const& elems);
        int n_blocks() const;
        void clear();
        int find_index(int i, int j) const;
        T* operator()(int i, int j);
        T const* operator()(int i, int j) const;
        void zeros();
        void symmetrize();
        void scale(T alpha);
        arma::SpMat<T> to_arma() const;
        arma::Mat<T> to_arma_dense() const;
        
        template<typename S>
        SpMatBsr(SpMatBsr<S> const& that): SpMatBsr() {
            *this = that;
        }
        template<typename S>
        SpMatBsr<T>& operator=(SpMatBsr<S> const& that) {
            if ((void *) this != (void *) &that) {
                n_rows = that.n_rows;
                n_cols = that.n_cols;
                b_len = that.b_len;
                copy_vec(that.row_idx, row_idx);
                copy_vec(that.col_idx, col_idx);
                copy_vec(that.row_ptr, row_ptr);
                copy_vec(that.val, val);
            }
            return *this;
        }
    };
    
    
    // -- fastkpm.cpp ------------------------------------------------------------------------
    
    // Scale eigenvalues within range (-1, +1)
    class EnergyScale {
    public:
        double lo, hi;
        double avg() const { return (hi + lo) / 2.0; }
        double mag() const { return (hi - lo) / 2.0; }
        double scale(double x) const { return (x - avg()) / mag(); }
        double unscale(double x) const { return x * mag() + avg(); }
        friend std::ostream& operator<< (std::ostream& stream, EnergyScale const& es) {
            return stream << "< lo = " << es.lo << " hi = " << es.hi << " >\n";
        }
    };
    
    // Use Lanczos to bound eigenvalues of H, and determine appropriate rescaling
    template <typename T>
    EnergyScale energy_scale(SpMatBsr<T> const& H, double extend, double tolerance);
    
    // Optimally damp Gibbs oscillations in KPM estimates
    Vec<double> jackson_kernel(int M);
    
    // Damp Gibbs oscillations while preserving complex analytic behavior of Green's function
    Vec<double> lorentz_kernel(int M, double lambda);
    
    // Chebyshev polynomials evaluated at x, default to be the first kind T_m(x)
    // if kind == 2, fill the second kind U_m(x)
    void chebyshev_fill_array(double x, Vec<double>& ret, int kind = 1);
    
    // Coefficients c_m that satisfy f(x) = \sum_m T_m(x) c_m
    Vec<double> expansion_coefficients(int M, int Mq, std::function<double(double)> f, EnergyScale es);
    
    // C_{mn} for optical conductivity (finite omega) or static conductivity (zero omega)
    // unit: e^2/h
    Vec<Vec<cx_double>> electrical_conductivity_coefficients(int M, int Mq, double kT, double mu, double omega, EnergyScale es, Vec<double> const& kernel);
    
    // Calculate \sum c_m mu_m
    double moment_product(Vec<double> const& c, Vec<double> const& mu);
    cx_double moment_product(Vec<Vec<cx_double>> const& c, Vec<Vec<cx_double>> const& mu);
    
    // Transformation of moments from mu to gamma, which corresponds to the density of states
    Vec<double> moment_transform(Vec<double> const& moments, int Mq);
    Vec<Vec<cx_double>> moment_transform(Vec<Vec<cx_double>> const& moments, int Mq, Vec<double> const& kernel);
    
    // Calculate \int dx rho(x) f(x)
    double density_product(Vec<double> const& gamma, std::function<double(double)> f, EnergyScale es);
    
    // Density of states rho(x) at Chebyshev points x
    void density_function(Vec<double> const& gamma, EnergyScale es, Vec<double>& x, Vec<double>& rho);
    // Generalized density map in 2d, involving "sandwiched" current operator
    void density_function(Vec<Vec<cx_double>> const& gamma, EnergyScale es, Vec<double>& x, Vec<double>& y, Vec<Vec<cx_double>>& rho);
    
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
        arma::Mat<T> R2;       // Another Random vector for 2nd version of moments calculation
        
        // Identity matrix
        virtual void set_R_identity(int n, int j_start, int j_end);
        virtual void set_R_identity(int n);
        virtual void set_R2_identity(int n, int j_start, int j_end);
        virtual void set_R2_identity(int n);
        
        // Uncorrelated random elements
        virtual void set_R_uncorrelated(int n, int s, RNG& rng, int j_start, int j_end);
        virtual void set_R_uncorrelated(int n, int s, RNG& rng);
        virtual void set_R2_uncorrelated(int n, int s, RNG& rng, int j_start, int j_end);
        virtual void set_R2_uncorrelated(int n, int s, RNG& rng);
        
        // Correlated random elements with mostly orthogonal rows
        virtual void set_R_correlated(Vec<int> const& groups, RNG& rng, int j_start, int j_end);
        virtual void set_R_correlated(Vec<int> const& groups, RNG& rng);
        
        // Transfer R matrix to device
        virtual void transfer_R() {}
        virtual void transfer_R2() {}
        
        // Set Hamiltonian and energy scale
        virtual void set_H(SpMatBsr<T> const& H, EnergyScale const& es) = 0;
        
        // Chebyshev moments: mu_m = tr T_m(Hs) ~ tr R^\dagger T_m(Hs) R
        virtual Vec<double> moments(int M) = 0;
        
        // Chebyshev moments: mu_{m1,m2} = tr( j1 T_{m1}(Hs) j2 T_{m2}(Hs) )
        virtual Vec<Vec<cx_double>> moments2_v1(int M, SpMatBsr<T> const& j1op,
                                                   SpMatBsr<T> const& j2op, int a_chunk_ncols=-1) = 0;
        virtual Vec<Vec<cx_double>> moments2_v2(int M, SpMatBsr<T> const& j1op,
                                                   SpMatBsr<T> const& j2op, int a_chunk_ncols=-1) = 0;
        
        // Approximates D ~ (xi R^\dagger + R xi^\dagger)/2 where xi = D R
        // and D ~ (\sum_m c_m T_m(Hs))R
        virtual void stoch_matrix(Vec<double> const& c, SpMatBsr<T>& D) = 0;
        
        // Approximates D ~ (d/dH^T) tr g where tr g ~ tr R^\dagger g R,
        // g ~ \sum_m c_m T_m(Hs) and coefficients c_m chosen such that
        // dg(x)/dx = D(x).
        // Typically D is dense, but only a small fraction of its elements are
        // desired. The sparsity structure of D is specified as an input.
        // REQUIREMENT: moments() must have been called previously.
        virtual void autodiff_matrix(Vec<double> const& c, SpMatBsr<T>& D) = 0;
    };
    
    // CPU engine
    template <typename T>
    std::shared_ptr<Engine<T>> mk_engine_cpu();
    
    // CuSPARSE engine
    template <typename T>
    std::shared_ptr<Engine<T>> mk_engine_cuSPARSE(int device);
    
    // MPI engine
    template <typename T>
    std::shared_ptr<Engine<T>> mk_engine_mpi();
    
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
