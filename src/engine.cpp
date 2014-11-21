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
    template <>
    float random_phase(RNG& rng) { return float(random_phase<double>(rng)); }
    
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
    template <>
    cx_float random_phase(RNG& rng) { return cx_float(random_phase<double>(rng)); }
    
    template <typename T>
    void Engine<T>::set_R_uncorrelated(int n, int s, RNG& rng) {
        R.set_size(n, s);
        T x = 1.0 / sqrt(s);
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
        R.set_size(n, n);
        R.fill(0.0);
        for (int i = 0; i < n; i++) {
            R(i, i) = 1.0;
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
    }
    template std::shared_ptr<Engine<double>> mk_engine();
    template std::shared_ptr<Engine<cx_double>> mk_engine();
}
