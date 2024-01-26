
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "pfc/bitmap.h"
#include "pfc/complex.h"
#include "pfc/config.h"
#include "gpu_kernel.cuh"

#include <stdio.h>

__global__ void mandelbrot_kernel(pfc::bmp::pixel_t* output, size_t width, size_t height, complex_t ll, complex_t ur) {
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    real_t real = ll.real + (ur.real - ll.real) * x / width;
    real_t imag = ll.imag + (ur.imag - ll.imag) * y / height;
    complex_t c = { real, imag };
    complex_t z = { 0, 0 };
    size_t iteration = 0;

    while (iteration++ < g_colors && z.norm() < g_infinity) {
        z = pfc::square(z) + c;
    }

    pfc::bmp::details::BGR_4_t a;
    a.green = static_cast <pfc::byte_t> (1.0 * iteration / g_colors * 255);
    output[y * width + x] = a;
}


cudaError_t call_mandelbrot_kernel(dim3 gridSize, dim3 blockSize, pfc::bmp::pixel_t* output, size_t width, size_t height, complex_t ll, complex_t ur) {
    mandelbrot_kernel<<<gridSize, blockSize>>> (output, width, height, ll, ur);
    return cudaPeekAtLastError();
}

