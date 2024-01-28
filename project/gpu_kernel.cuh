#pragma once

#if __has_include (<pfc/compiler.h>)
	#include <pfc/compiler.h>
#endif

#include "pfc/complex.h"
#include "pfc/shared.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstddef>

using namespace shared;

cudaError_t call_mandelbrot_kernel(dim3 gridSize, dim3 blockSize, uint8_t* output, uint16_t height, complex_t ll, complex_t ur);