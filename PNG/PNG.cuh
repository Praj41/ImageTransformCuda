
#ifndef PNG_CUH
#define PNG_CUH

#include <vector>
#include <iostream>

#include <cuda.h>


#include "Pixel.cuh"

namespace praj {

    class PNGpu {
    public:
        PNGpu();
        PNGpu(unsigned int width, unsigned int height);
        bool readFile(const std::string &fileName);
        bool writeFile(const std::string &fileName);
        void toRGB();
        void toHSL();
        void greyscale();
    private:
        unsigned int width_{};
        unsigned int height_{};
        std::vector<rgbaColor> rgbaImage_;
        std::vector<hslaColor> hslaImage_;

    };
}

#endif //PNG_CUH
