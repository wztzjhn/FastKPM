//
//  timer.cpp
//  tibidy
//
//  Created by Kipton Barros on 8/1/14.
//
//

#include "fastkpm.h"

namespace fkpm {
    Timer timer[10];
    
    Timer::Timer() {
        reset();
    }
    
    void Timer::reset() {
        t0 = std::chrono::system_clock::now();
    }
    
    double Timer::measure() {
        auto t1 = std::chrono::system_clock::now();
        std::chrono::duration<double> dt = t1 - t0;
        reset();
        return dt.count();
    }
}
