
#include <cuda_runtime.h>
#include <driver_types.h>

#include "PNG.cuh"

#include "lodepng.cuh"

namespace praj {
    PNGpu::PNGpu()
            : width_(0),
              height_(0) {}

    PNGpu::PNGpu(unsigned int width, unsigned int height)
            : width_(width),
              height_(height) {}

    bool PNGpu::readFile(const std::string &fileName) {
        std::vector<unsigned char> data;
        unsigned int error = lodepng::decode(data, width_, height_, fileName);
        rgbaImage_.resize(width_ * height_);
        memcpy((void *) rgbaImage_.data(), (void *) data.data(), width_ * height_ * 4);
        return error;
    }

    bool PNGpu::writeFile(const std::string &fileName) {
        unsigned int error = lodepng::encode(fileName, (unsigned char *) rgbaImage_.data(), width_, height_);
        return error;
    }

    __global__ void rgb2hsl(rgbaColor *in, hslaColor *out) {

        unsigned int blockNum = blockIdx.y * gridDim.x + blockIdx.x;
        unsigned int globalThreadId = blockNum * blockDim.x + threadIdx.y * blockDim.x + threadIdx.x;

        const auto &rgb = in[globalThreadId];
        auto &hsl = out[globalThreadId];

        double r, g, b, min, max, chroma;

        // Change rgb into [0, 1]
        r = rgb.r / 255.0;
        g = rgb.g / 255.0;
        b = rgb.b / 255.0;

        // HSV Calculations -- formulas sourced from https://en.wikipedia.org/wiki/HSL_and_HSV
        // Compute constants
        min = (r < g) ? r : g;
        min = (min < b) ? min : b;

        max = (r > g) ? r : g;
        max = (max > b) ? max : b;

        chroma = max - min;

        // Compute A
        hsl.a = rgb.a / 255.0;

        // Compute L
        hsl.l = 0.5 * (max + min);

        // Check for black, white, and shades of gray, where H is undefined,
        // S is always 0, and L controls the shade of gray.  Mathematically, this
        // is true when chroma == 0, but we'll use a near-zero value to account
        // for floating point errors.
        //
        // This check is required here, or division by zero will occur when
        // calculating S and H below.
        if (chroma < 0.0001 || max < 0.0001) {
            hsl.h = hsl.s = 0;
            return;
        }

        // Compute S
        hsl.s = chroma / (1 - fabs((2 * hsl.l) - 1));

        // Compute H
        if (max == r) { hsl.h = fmod((g - b) / chroma, (double) 6); }
        else if (max == g) { hsl.h = ((b - r) / chroma) + 2; }
        else { hsl.h = ((r - g) / chroma) + 4; }

        hsl.h *= 60;
        if (hsl.h < 0) { hsl.h += 360; }

    }

    __global__ void hsl2rgb(hslaColor *in, rgbaColor *out) {


        unsigned int blockNum = blockIdx.y * gridDim.x + blockIdx.x;
        unsigned int globalThreadId = blockNum * blockDim.x + threadIdx.y * blockDim.x + threadIdx.x;

        const auto &hsl = in[globalThreadId];
        auto &rgb = out[globalThreadId];

        if (hsl.s <= 0.001) {
            rgb.r = rgb.g = rgb.b = round(hsl.l * 255);
        } else {
            double c = (1 - fabs((2 * hsl.l) - 1)) * hsl.s;
            double hh = hsl.h / 60;
            double x = c * (1 - fabs(fmod(hh, (double) 2) - 1));
            double r, g, b;

            if (hh <= 1) {
                r = c;
                g = x;
                b = 0;
            } else if (hh <= 2) {
                r = x;
                g = c;
                b = 0;
            } else if (hh <= 3) {
                r = 0;
                g = c;
                b = x;
            } else if (hh <= 4) {
                r = 0;
                g = x;
                b = c;
            } else if (hh <= 5) {
                r = x;
                g = 0;
                b = c;
            } else {
                r = c;
                g = 0;
                b = x;
            }

            double m = hsl.l - (0.5 * c);
            rgb.r = round((r + m) * 255);
            rgb.g = round((g + m) * 255);
            rgb.b = round((b + m) * 255);
        }

        rgb.a = round(hsl.a * 255);
    }

    void PNGpu::toRGB() {

        hslaColor *in;
        cudaMalloc((void **) &in, height_ * width_ * 32);
        cudaMemcpy(in, hslaImage_.data(), height_ * width_ * 32, cudaMemcpyHostToDevice);
        rgbaColor *out;
        cudaMalloc((void **) &out, height_ * width_ * 4);
        praj::hsl2rgb<<<height_, width_>>>(in, out);
        rgbaImage_.resize(height_ * width_);
        cudaMemcpy(rgbaImage_.data(), out, height_ * width_ * 32, cudaMemcpyDeviceToHost);
        std::cout << "done" << std::endl;
    }

    void PNGpu::toHSL() {
        rgbaColor *in;
        cudaMalloc((void **) &in, height_ * width_ * 4);
        cudaMemcpy(in, rgbaImage_.data(), height_ * width_ * 4, cudaMemcpyHostToDevice);
        hslaColor *out;
        cudaMalloc((void **) &out, height_ * width_ * 32);
        //dim3 block(512, 1, 1);
        //dim3 grid(height_, width_);
        praj::rgb2hsl<<<height_, width_>>>(in, out);
        hslaImage_.resize(height_ * width_);
        cudaMemcpy(hslaImage_.data(), out, height_ * width_ * 32, cudaMemcpyDeviceToHost);
        std::cout << "done" << std::endl;
    }

}
