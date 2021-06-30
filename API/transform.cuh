#pragma once

#include <string>
#include <thread>

#include "SafeQ.cuh"

namespace praj {

    class transform {
        static std::string directory;
        static void worker(praj::fileQueue&);
    public:
        void batch();
    };

}
