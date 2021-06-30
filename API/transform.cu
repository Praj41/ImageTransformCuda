#include <windows.h>
#include <iostream>


#include "PNG.cuh"
#include "transform.cuh"

namespace praj {

    std::string transform::directory;

    void transform::batch() {
        directory = "C:/Users/ADMIN/Pictures/Screenshots/";
        praj::fileQueue queue[8];
        unsigned int i = 0;

        WIN32_FIND_DATA data;
        HANDLE hFind = FindFirstFile("C:/Users/ADMIN/Pictures/Screenshots/Screenshot*", &data);      // DIRECTORY


        if (hFind != INVALID_HANDLE_VALUE) {
            do {

                queue[i].push(data.cFileName);

                /*
                png.readFile(filename);

                filename[34] = 'N';
                png.toHSL();
                png.greyscale();
                png.toRGB();
                png.writeFile(filename.substr(0, filename.size()-4) + "_greyscale.png");
                png.edge();

                png.writeFile(filename.substr(0, filename.size()-4) + "_edge.png");
    */
            i = ++i % 8;
            } while (FindNextFile(hFind, &data));
            FindClose(hFind);
        }

        std::vector<std::thread> workers;


        for (i = 0; i < 8; i++) {
            workers.emplace_back(worker, queue[i]);
        }

        for (auto &item : workers)
            item.join();

    }

    void transform::worker(fileQueue &queue) {
        praj::PNGpu png;

        while (!queue.empty()) {

            std::string filename(directory + queue.pop());

            std::cout << filename << std::endl;

            png.readFile(filename);

            filename[34] = 'N';
            png.toHSL();
            png.greyscale();
            png.toRGB();
            png.writeFile(filename.substr(0, filename.size()-4) + "_greyscale.png");
            png.edge();

            png.writeFile(filename.substr(0, filename.size()-4) + "_edge.png");
        }
    }
}
