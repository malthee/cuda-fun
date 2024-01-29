
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "pfc/bitmap.h"
#include "pfc/complex.h"
#include "pfc/shared.h"
#include "gpu_kernel.cuh"

#include <stdio.h>

__global__ void mandelbrot_kernel(uint8_t* output, uint16_t height, complex_t ll, complex_t ur) {
	uint32_t x{ blockIdx.x * blockDim.x + threadIdx.x };
	uint32_t y{ blockIdx.y * blockDim.y + threadIdx.y };
	if (x >= g_width || y >= height) return; // Handle out of bounds threads, is difficult to avoid with non-power-of-two sizes

	float real = fmaf((ur.real - ll.real), x / static_cast<float>(g_width), ll.real);
	float imag = fmaf((ur.imag - ll.imag), static_cast<float>(y) / height, ll.imag);
	complex_t z(real, imag);
	uint8_t iteration{ 1 }; // "Unroll" first iteration by initializing z to c

	#pragma unroll
	for (; iteration < g_colors; ++iteration) {
		if (z.norm() >= g_infinity) break;
		z = pfc::square(z);
		z.imag += imag;
		z.real += real;
	}

	output[y * g_width + x] = ++iteration;
}


cudaError_t call_mandelbrot_kernel(dim3 gridSize, dim3 blockSize, cudaStream_t stream, uint8_t* output, uint16_t height, complex_t ll, complex_t ur) {
	mandelbrot_kernel << <gridSize, blockSize, 0, stream>> > (output, height, ll, ur);
	return cudaPeekAtLastError();
}

