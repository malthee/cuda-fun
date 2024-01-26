#pragma once

#if __has_include (<pfc/compiler.h>)
	#include <pfc/compiler.h>
#endif

#include "pfc/complex.h"
#include "pfc/config.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstddef>

using namespace config;

cudaError_t call_mandelbrot_kernel(dim3 gridSize, dim3 blockSize, pfc::bmp::pixel_t* output, size_t width, size_t height, complex_t ll, complex_t ur);