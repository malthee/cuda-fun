
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "pfc/bitmap.h"
#include "pfc/complex.h"
#include "pfc/shared.h"
#include "gpu_kernel.cuh"

#include <stdio.h>

__global__ void mandelbrot_kernel(uint8_t* output, size_t height, complex_t ll, complex_t ur) {
	size_t x = blockIdx.x * blockDim.x + threadIdx.x;
	size_t y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= g_width || y >= height) return; // Handle out of bounds threads, is difficult to avoid with non-power-of-two sizes

	real_t real = ll.real + (ur.real - ll.real) * x / g_width;
	real_t imag = ll.imag + (ur.imag - ll.imag) * y / height;
	complex_t c = { real, imag };
	complex_t z = { 0, 0 };
	uint8_t iteration{};

	while (iteration++ < g_colors && z.norm() < g_infinity) {
		z = pfc::square(z) + c;
	}

	output[y * g_width + x] = iteration;
}


cudaError_t call_mandelbrot_kernel(dim3 gridSize, dim3 blockSize, uint8_t* output, size_t height, complex_t ll, complex_t ur) {
	mandelbrot_kernel << <gridSize, blockSize >> > (output, height, ll, ur);
	return cudaPeekAtLastError();
}

