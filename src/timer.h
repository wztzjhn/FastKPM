//
//  timer.h
//  tibidy
//
//  Created by Kipton Barros on 8/1/14.
//
//


#ifndef __tibidy__timer___
#define __tibidy__timer___

#include <string>
#include <chrono>


namespace fkpm {
    class Timer {
    public:
        std::chrono::time_point<std::chrono::system_clock> t;
        bool mute = true;
        
        Timer();
        void enable();
        void reset();
        void print(std::string msg);
    };

    extern Timer timer[10];
}

#endif /* defined(__tibidy__timer__) */
