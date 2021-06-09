#include <iostream>
#include <vector>

#include "lodepng.cuh"
#include "Pixel.cuh"
#include "PNG.cuh"

int main() {
    std::cout << "Hello, World!" << std::endl;
/*
    int x(256) , y(256);

    std::vector<uint32_t> arr(x * y);

    for (int i = 0; i < x; ++i) {
        for (int j = 0; j < y; ++j) {
            rgbaColor pixel;

            pixel.r = i;
            pixel.g = 0;
            pixel.b = j;
            pixel.a = 0xFF;

            arr[i * y + j] = pixel.raw;
        }
    }

    lodepng::encode("../pattern.png", (unsigned char*)arr.data(), x, y);
*/

    praj::PNGpu png;

    png.readFile("../paattern2.png");

    png.toHSL();
    png.greyscale();
    png.toRGB();

    png.writeFile("../pattern1.png");

    return 0;
}
