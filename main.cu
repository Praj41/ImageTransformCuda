#include <iostream>

#include "transform.cuh"

int main() {
    std::cout << "Hello, World!" << std::endl;

    praj::transform processor;

    processor.batch();

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


    praj::PNGpu png;
    std::string filename("../Paattern2.png");
    png.readFile(filename);

    png.toHSL();
    png.greyscale();
    png.toRGB();
    png.writeFile(filename.substr(0, filename.size()-4) + "_greyscale.png");
    png.edge();

    png.writeFile(filename.substr(0, filename.size()-4) + "_edge.png");
*/

    return 0;
}
