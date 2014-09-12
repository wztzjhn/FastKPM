//
//  timer.cpp
//  tibidy
//
//  Created by Kipton Barros on 8/1/14.
//
//

#include <iostream>
#include "timer.h"

namespace fkpm {
    Timer timer[10];
    
    Timer::Timer() {
        reset();
    }
    
    void Timer::enable() {
        mute = false;
        reset();
    }
    
    void Timer::reset() {
        t = std::chrono::system_clock::now();
    }
    
    void Timer::print(std::string msg) {
        if (!mute) {
            std::chrono::duration<double> dt = std::chrono::system_clock::now() - t;
            std::cout << msg << " : " << dt.count() << "s\n";
        }
        reset();
    }
}
