
#ifndef PNG_CUH
#define PNG_CUH

#include <vector>
#include <iostream>

#include "Pixel.h"

namespace praj {

    class PNGpu {
    public:
        PNGpu();
        PNGpu(unsigned int width, unsigned int height);
        bool readFile(const std::string &fileName);
        bool writeFile(const std::string &fileName);
    private:
        unsigned int width_{};
        unsigned int height_{};
        std::vector<rgbaColor> image_;
    };
}

#endif //PNG_CUH
