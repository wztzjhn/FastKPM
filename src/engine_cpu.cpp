#include "fastkpm.h"

namespace fkpm {
    
    class EngineCx_CPU: public EngineCx {
    public:
        arma::sp_cx_mat Hs;    // Scaled Hamiltonian
        arma::sp_cx_mat dE_dH; // Grand free energy matrix derivative
        
        EngineCx_CPU(int n, int s): EngineCx(n, s) {}
        
        void set_H(arma::sp_cx_mat const& H, EnergyScale const& es) {
            this-> es = es;
            Hs = es.scale(H);
            dE_dH = Hs;
        }
        
        Vec<double> moments(int M) {
            Vec<double> mu(M);
            mu[0] = n;
            mu[1] = arma::trace(Hs).real();
            
            arma::cx_mat a0(n, s), a1(n, s), a2(n, s);
            
            a0 = R;
            a1 = Hs * R;
            
            for (int m = 2; m < M; m++) {
                a2 = 2*Hs*a1 - a0;
                mu[m] = arma::cdot(R, a2).real();
                a0 = a1;
                a1 = a2;
            }
            
            return mu;
        }
        
        void stoch_orbital(Vec<double> const& c) {
            int M = c.size();
            arma::cx_mat a0(n, s), a1(n, s), a2(n, s);
            a0 = R;
            a1 = Hs * R;
            xi = c[0]*a0 + c[1]*a1;
            for (int m = 2; m < M; m++) {
                a2 = 2*Hs*a1 - a0;
                xi += c[m]*a2;
                a0 = a1;
                a1 = a2;
            }
        }
    };
    
    std::shared_ptr<EngineCx> mk_engine_cx_CPU(int n, int s) {
        return std::make_shared<EngineCx_CPU>(n, s);
    }
}
