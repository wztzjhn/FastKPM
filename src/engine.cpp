#include <cassert>
#include "fastkpm.h"


namespace fkpm {
    
    template <typename T>
    T random_phase(RNG& rng);
    
    // Random number from {+1, -1}
    template <>
    double random_phase(RNG& rng) {
        std::uniform_int_distribution<int> dist2(0,1);
        return 2*dist2(rng)-1;
    }
    template <>
    float random_phase(RNG& rng) { return float(random_phase<double>(rng)); }
    
    // Random number from {+1, +i, -1, -i}
    template <>
    cx_double random_phase(RNG& rng) {
        std::uniform_int_distribution<int> dist4(0,3);
        switch (dist4(rng)) {
            case 0: return {+1,  0};
            case 1: return { 0, +1};
            case 2: return {-1,  0};
            case 3: return { 0, -1};
        }
        assert(false);
    }
    template <>
    cx_float random_phase(RNG& rng) { return cx_float(random_phase<cx_double>(rng)); }
    
    template <typename T>
    void Engine<T>::set_R_uncorrelated(int n, int s, RNG& rng, int j_start, int j_end) {
        if (j_start < 0) j_start = 0;
        if (j_end < 0) j_end = s;
        assert (0 <= j_start && j_start < j_end && j_end <= s);
        
        R.set_size(n, j_end-j_start);
        T x = 1.0 / sqrt(s);
        for (int j = j_start; j < j_end; j++) {
            RNG rng_j(rng()); // new RNG sequence for each column j
            for (int i = 0; i < n; i++) {
                R(i, j-j_start) = random_phase<T>(rng_j) * x;
            }
        }
        transfer_R();
    }
    
    template <typename T>
    void Engine<T>::set_R_correlated(Vec<int> const& groups, RNG& rng, int j_start, int j_end) {
        auto minmax = std::minmax_element(groups.begin(), groups.end());
        assert(*minmax.first == 0);
        assert(*minmax.second >= 0);
        int s = *minmax.second + 1; // number of columns in full R matrix
        
        if (j_start < 0) j_start = 0;
        if (j_end < 0) j_end = s;
        assert (0 <= j_start && j_start < j_end && j_end <= s);
        
        int n = groups.size();
        if (j_end > n) {
            std::cerr << "Inefficiency warning: Using more random vectors than matrix rows!\n";
        }
        
        R.set_size(n, j_end-j_start);
        R.fill(0.0);
        for (int j = j_start; j < j_end; j++) {
            RNG rng_j(rng()); // new RNG sequence for each column j
            for (int i = 0; i < n; i++) {
                if (groups[i] == j) {
                    R(i, j-j_start) = random_phase<T>(rng_j);
                }
            }
        }
        transfer_R();
    }
    
    template <typename T>
    void Engine<T>::set_R_identity(int n, int j_start, int j_end) {
        if (j_start < 0) j_start = 0;
        if (j_end < 0) j_end = n;
        assert (0 <= j_start && j_start < j_end && j_end <= n);
        
        R.set_size(n, j_end-j_start);
        R.fill(0.0);
        for (int j = j_start; j < j_end; j++) {
            R(j, j-j_start) = 1.0;
        }
        transfer_R();
    }
    
    
    template class Engine<float>;
    template class Engine<double>;
    template class Engine<cx_float>;
    template class Engine<cx_double>;
    
    
    template <typename T>
    std::shared_ptr<Engine<T>> mk_engine() {
        std::shared_ptr<Engine<T>> ret;
        ret = mk_engine_cuSPARSE<T>(0);
        if (ret == nullptr)
            ret = mk_engine_cpu<T>();
        return ret;
//        return mk_engine_cpu<T>();       // for testing cpu code only
    }
    template std::shared_ptr<Engine<float>> mk_engine();
    template std::shared_ptr<Engine<double>> mk_engine();
    template std::shared_ptr<Engine<cx_float>> mk_engine();
    template std::shared_ptr<Engine<cx_double>> mk_engine();
}
