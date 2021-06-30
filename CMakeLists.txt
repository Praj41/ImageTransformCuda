cmake_minimum_required(VERSION 3.17)
project(ImageTransformGPU CUDA)

set(CMAKE_CUDA_STANDARD 17)

include_directories(lodepng PNG API)

add_executable(ImageTransformGPU main.cu lodepng/lodepng.cu PNG/PNG.cu PNG/PNG.cuh PNG/Pixel.cuh API/transform.cu API/transform.cuh API/SafeQ.cuh API/SafeQ.cu)

set_target_properties(
        ImageTransformGPU
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)