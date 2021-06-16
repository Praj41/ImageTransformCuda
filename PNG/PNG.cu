
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

    __global__ void rgb2hsl(rgbaColor *in, hslaColor *out, unsigned int h, unsigned int w) {

        //unsigned int blockNum = blockIdx.y * gridDim.x + blockIdx.x;
        //unsigned int globalThreadId = blockNum * blockDim.x + threadIdx.y * blockDim.x + threadIdx.x;

        unsigned int x = blockDim.x * blockIdx.y + threadIdx.x;
        unsigned int y = blockIdx.x;

        if (y <= h && x <= w) {

            const auto &rgb = in[y*w + x];
            auto &hsl = out[y*w + x];

            float r, g, b, minimum, maximum, chroma;

            // Change rgb into [0, 1]
            r = (float) rgb.r / 255.0f;
            g = (float) rgb.g / 255.0f;
            b = (float) rgb.b / 255.0f;

            // HSV Calculations -- formulas sourced from https://en.wikipedia.org/wiki/HSL_and_HSV
            // Compute constants
            minimum = min(r, min(g, b));
            maximum = max(r, max(g, b));

            chroma = maximum - minimum;

            // Compute A
            hsl.a = (float) rgb.a / 255.0f;

            // Compute L
            hsl.l = 0.5f * (maximum + minimum);

            // Check for black, white, and shades of gray, where H is undefined,
            // S is always 0, and L controls the shade of gray.  Mathematically, this
            // is true when chroma == 0, but we'll use a near-zero value to account
            // for floating point errors.
            //
            // This check is required here, or division by zero will occur when
            // calculating S and H below.
            if (chroma < 0.0001f || maximum < 0.0001f) {
                hsl.h = hsl.s = 0;
                return;
            }

            // Compute S
            hsl.s = chroma / (1 - fabs((2 * hsl.l) - 1));

            // Compute H
            if (maximum == r) { hsl.h = fmod((g - b) / chroma, (float) 6); }
            else if (maximum == g) { hsl.h = ((b - r) / chroma) + 2; }
            else { hsl.h = ((r - g) / chroma) + 4; }

            hsl.h *= 60.0f;
            if (hsl.h < 0) { hsl.h += 360; }
        }
    }

    __global__ void hsl2rgb(hslaColor *in, rgbaColor *out, unsigned int h, unsigned int w) {


        //unsigned int blockNum = blockIdx.y * gridDim.x + blockIdx.x;
        //unsigned int globalThreadId = blockNum * blockDim.x + threadIdx.y * blockDim.x + threadIdx.x;

        unsigned int x = blockDim.x * blockIdx.y + threadIdx.x;
        unsigned int y = blockIdx.x;

        if (y <= h && x <= w) {

            const auto &hsl = in[y*w + x];
            auto &rgb = out[y*w + x];

            if (hsl.s <= 0.001) {
                rgb.r = rgb.g = rgb.b = round((double) hsl.l * 255);
            } else {
                float c = (1 - fabs((2 * hsl.l) - 1)) * hsl.s;
                float hh = hsl.h / 60;
                float x = c * (1 - fabs(fmod(hh, (float) 2) - 1));
                float r, g, b;

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

                float m = hsl.l - (0.5f * c);
                rgb.r = round((double) (r + m) * 255);
                rgb.g = round((double) (g + m) * 255);
                rgb.b = round((double) (b + m) * 255);
            }

            rgb.a = round((double) hsl.a * 255);
        }
    }

    __global__ void toGrey(hslaColor *out, unsigned int h, unsigned int w) {
        unsigned int x = blockDim.x * blockIdx.y + threadIdx.x;
        unsigned int y = blockIdx.x;

        if (y <= h && x <= w) {
            out[y * w + x].s = 0;
        }
    }

    __global__ void edge(unsigned char *in, unsigned char *out, unsigned int h, unsigned int w) {
        unsigned int x = blockDim.x * blockIdx.y + threadIdx.x;
        unsigned int y = blockIdx.x;

        if (y < h && x < w && x > 0 && y > 0) {
            int val = - in[(y - 1) * w + x - 1] - in[(y - 1) * w + x] - in[(y - 1) * w + x + 1]
            - in[(y) * w + x - 1] + 8 * in[(y) * w + x] - in[(y) * w + x + 1]
            - in[(y + 1) * w + x - 1] - in[(y + 1) * w + x] - in[(y + 1) * w + x + 1];

            val = (val < 0 ? 0 : val);
            val = (val > 255 ? 255 : val);

            out[y * w + x] = val;
        }
    }

    void PNGpu::toRGB() {

        hslaColor *in;
        cudaMalloc((void **) &in, height_ * width_ * 16);
        cudaMemcpy(in, hslaImage_.data(), height_ * width_ * 16, cudaMemcpyHostToDevice);
        rgbaColor *out;
        cudaMalloc((void **) &out, height_ * width_ * 4);
        dim3 block(512, 1, 1);
        dim3 grid(height_, (width_/512)+1);
        praj::hsl2rgb<<<grid, block>>>(in, out, height_, width_);
        rgbaImage_.resize(height_ * width_);
        cudaMemcpy(rgbaImage_.data(), out, height_ * width_ * 4, cudaMemcpyDeviceToHost);
        cudaFree(in);
        cudaFree(out);
        std::cout << "To RGB done" << std::endl;
    }

    void PNGpu::toHSL() {

        rgbaColor *in;
        cudaMalloc((void **) &in, height_ * width_ * 4);
        cudaMemcpy(in, rgbaImage_.data(), height_ * width_ * 4, cudaMemcpyHostToDevice);
        hslaColor *out;
        cudaMalloc((void **) &out, height_ * width_ * 16);
        dim3 block(512, 1, 1);
        dim3 grid(height_, (width_/512)+1);
        praj::rgb2hsl<<<grid, block>>>(in, out, height_, width_);
        hslaImage_.resize(height_ * width_);
        cudaMemcpy(hslaImage_.data(), out, height_ * width_ * 16, cudaMemcpyDeviceToHost);
        cudaFree(in);
        cudaFree(out);
        std::cout << "To HSL done" << std::endl;
    }

    void PNGpu::greyscale() {
        hslaColor *out;
        cudaMalloc((void **) &out, height_ * width_ * 16);
        cudaMemcpy(out, hslaImage_.data(), height_ * width_ * 16, cudaMemcpyHostToDevice);
        dim3 block(512, 1, 1);
        dim3 grid(height_, (width_/512)+1);
        praj::toGrey<<<grid, block>>>(out, height_, width_);
        cudaMemcpy(hslaImage_.data(), out, height_ * width_ * 16, cudaMemcpyDeviceToHost);
        cudaFree(out);
        std::cout << "To Greyscale done" << std::endl;
    }

    void PNGpu::edge() {
        std::vector<unsigned char> grey(height_ * width_);

        for (int i = 0; i < height_*width_; ++i) {
            grey[i] = rgbaImage_[i].r;
        }

        unsigned char *in, *out;
        cudaMalloc((void**) &in, height_*width_);
        cudaMalloc((void**) &out, height_*width_);

        cudaMemcpy(in, grey.data(), height_*width_, cudaMemcpyHostToDevice);
        dim3 block(512, 1, 1);
        dim3 grid(height_, (width_/512)+1);

        praj::edge<<<grid, block>>>(in, out, height_, width_);

        cudaMemcpy(grey.data(), out, height_*width_, cudaMemcpyDeviceToHost);

        for (int i = 0; i < height_*width_; ++i) {
            rgbaImage_[i].r = rgbaImage_[i].g = rgbaImage_[i].b = grey[i];
        }
        cudaFree(out);
        cudaFree(in);
    }

}
