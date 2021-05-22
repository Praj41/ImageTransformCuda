cmake_minimum_required(VERSION 3.19)
project(ImageTransformGPU CUDA)

set(CMAKE_CUDA_STANDARD 17)

include_directories(lodepng PNG)

add_executable(ImageTransformGPU main.cu lodepng/lodepng.cu PNG/PNG.cu PNG/PNG.cuh PNG/Pixel.h)

set_target_properties(
        ImageTransformGPU
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)