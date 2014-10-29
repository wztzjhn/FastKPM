#include <cassert>
#include "fastkpm.h"


namespace fkpm {
    
    template <typename T>
    T random_phase(RNG& rng);
    
    // Random number from {+1, -1}
    template <>
    double random_phase(RNG& rng) {
        std::uniform_int_distribution<uint32_t> dist2(0,1);
        return double(dist2(rng));
    }
    
    // Random number from {+1, i, -1, -i}
    template <>
    cx_double random_phase(RNG& rng) {
        std::uniform_int_distribution<uint32_t> dist4(0,3);
        switch (dist4(rng)) {
            case 0: return {+1,  0};
            case 1: return { 0, +1};
            case 2: return {-1,  0};
            case 3: return { 0, -1};
        }
        assert(false);
    }

    
    template <typename T>
    void Engine<T>::set_R_uncorrelated(int n, int s, RNG& rng) {
        xi.set_size(n, s);
        R.set_size(n, s);
        double x = 1.0 / sqrt(s);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < s; j++) {
                R(i, j) = random_phase<T>(rng) * x;
            }
        }
        transfer_R();
    }
    
    template <typename T>
    void Engine<T>::set_R_correlated(Vec<int> const& groups, RNG& rng) {
        auto minmax = std::minmax_element(groups.begin(), groups.end());
        assert(*minmax.first == 0);
        int s = *minmax.second + 1;
        int n = groups.size();
        xi.set_size(n, s);
        R.set_size(n, s);
        R.fill(0.0);
        for (int i = 0; i < n; i++) {
            int g = groups[i];
            R(i, g) = random_phase<T>(rng);
        }
        transfer_R();
    }
    
    template <typename T>
    void Engine<T>::set_R_identity(int n) {
        xi.set_size(n, n);
        R.set_size(n, n);
        R.fill(0.0);
        for (int i = 0; i < n; i++) {
            R(i, i) = 1.0;
        }
        transfer_R();
    }
    
    template <typename T>
    void Engine<T>::set_H(SpMatCoo<T> const& H, EnergyScale const& es) {
        assert(H.n_rows == H.n_cols);
        this->es = es;
        Hs = H;
        for (T& x : Hs.val)
            x /= es.mag();
        for (int i = 0; i < Hs.n_rows; i++)
            Hs.add(i, i, -es.avg()/es.mag());
        transfer_H();
    }
    
    template <typename T>
    T Engine<T>::stoch_element(int i, int j) {
        T x1 = arma::cdot(R.row(j), xi.row(i)); // xi R^dagger
        T x2 = arma::cdot(xi.row(j), R.row(i)); // R xi^dagger
        return 0.5*(x1+x2);
    }
    
    template <typename T>
    void Engine<T>::transfer_R() {};
    
    template <typename T>
    void Engine<T>::transfer_H() {};
    
    template <typename T>
    Vec<double> Engine<T>::moments(int M) {
        arma::SpMat<T> Hs_a = Hs.to_arma();
        int n = R.n_rows;
        assert(Hs.n_rows == n && Hs.n_cols == n);
        
        Vec<double> mu(M);
        mu[0] = n;
        mu[1] = std::real(arma::trace(Hs_a));
        
        arma::Mat<T> a0 = R;
        arma::Mat<T> a1 = Hs_a * R;
        
        for (int m = 2; m < M; m++) {
            arma::Mat<T> a2 = 2*Hs_a*a1 - a0;
            mu[m] = std::real(arma::cdot(R, a2));
            a0 = a1;
            a1 = a2;
        }
        
        return mu;
    }
    
    template <typename T>
    void Engine<T>::stoch_orbital(Vec<double> const& c) {
        int M = c.size();
        
        arma::SpMat<T> Hs_a = Hs.to_arma();
        int n = R.n_rows;
        assert(Hs.n_rows == n && Hs.n_cols == n);
        
        arma::Mat<T> a0 = R;
        arma::Mat<T> a1 = Hs_a * R;
        
        xi = c[0]*a0 + c[1]*a1;
        for (int m = 2; m < M; m++) {
            arma::Mat<T> a2 = 2*Hs_a*a1 - a0;
            xi += c[m]*a2;
            a0 = a1;
            a1 = a2;
        }
    }
    
//    template <typename T>
//    arma::sp_cx_mat Engine<T>::autodiff(Vec<double> const& c) { abort(); }
    
    template <typename T>
    arma::SpMat<T> Engine<T>::autodiff(Vec<double> const& c) {
        int M = c.size();
        arma::SpMat<T> Hs_a = Hs.to_arma();
        int n = R.n_rows;
        int s = R.n_cols;

        arma::SpMat<T> grad = Hs_a;
        grad.zeros();
        
        // forward calculation
        
        Vec<double> mu(M);
        mu[0] = n;                            // Tr[T_0[H]] = Tr[1]
        mu[1] = std::real(arma::trace(Hs_a)); // Tr[T_1[H]] = Tr[H]
        
        arma::Mat<T> a0 = R;          // T_0[H] |r> = 1 |r>
        arma::Mat<T> a1 = Hs_a * R;   // T_1[H] |r> = H |r>
        arma::Mat<T> a2(n, s);
        
        for (int m = 2; m < M; m++) {
            a2 = 2*Hs_a*a1 - a0;
            mu[m] = std::real(arma::cdot(R, a2));
            a0 = a1;
            a1 = a2;
        }
        
        // reverse calculation
        
        arma::Mat<T> b2(n, s);
        arma::Mat<T> b1(n, s, arma::fill::zeros);
        arma::Mat<T> b0 = R * c[M - 1];
        
        // need special logic since mu[1] was calculated exactly
        for (int i = 0; i < n; i++) { grad(i, i) += c[1]; }
        Vec<double> cp = c; cp[1] = 0;
        
        for (int m = M-2; m >= 0; m--) {
            // a0 = alpha_{m}
            // b0 = beta_{m}
            for (int j = 0; j < n; j++) {
                for (int ptr = Hs_a.col_ptrs[j]; ptr < Hs_a.col_ptrs[j+1]; ptr++) {
                    int i = Hs_a.row_indices[ptr];
                    grad(i, j) += (m == 0 ? 1.0 : 2.0) * arma::cdot(b0.row(j), a0.row(i));
                }
            }
            a2 = a1;
            b2 = b1;
            a1 = a0;
            b1 = b0;
            a0 = 2*Hs_a*a1 - a2;;
            b0 = cp[m]*R + 2*Hs_a*b1 - b2;
        }
        return grad / es.mag();
        // (c, mu).zipped.map(_*_).sum
    }
    
    
    template class Engine<double>;
    template class Engine<cx_double>;
    
    template <typename T>
    std::shared_ptr<Engine<T>> mk_engine() {
        std::shared_ptr<Engine<T>> ret;
        ret = mk_engine_cuSPARSE<T>();
        if (ret == nullptr)
            ret = std::make_shared<Engine<T>>();
        return ret;
    }
    template std::shared_ptr<Engine<double>> mk_engine();
    template std::shared_ptr<Engine<cx_double>> mk_engine();
    
    std::shared_ptr<Engine<double>> mk_engine_re() {
        return mk_engine<double>();
    }
    
    std::shared_ptr<Engine<cx_double>> mk_engine_cx() {
        return mk_engine<cx_double>();
    }
}
