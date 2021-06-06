
#include "PNG.cuh"

#include "lodepng.h"

namespace praj {
    PNGpu::PNGpu()
            : width_(0),
              height_(0) {}

    PNGpu::PNGpu(unsigned int width, unsigned int height) : width_(width), height_(height) {}

    bool PNGpu::readFile(const std::string &fileName) {
        std::vector<unsigned char> data;
        unsigned int error = lodepng::decode(data, width_, height_, fileName);
        image_.resize(width_ * height_);
        memcpy((void*)image_.data(), (void*)data.data(), width_*height_*4);
        return error;
    }

    bool PNGpu::writeFile(const std::string &fileName) {
        unsigned int error = lodepng::encode(fileName, (unsigned char*)image_.data(), width_, height_);
        return error;
    }


}
