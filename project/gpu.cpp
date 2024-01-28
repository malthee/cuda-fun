#include "pfc/bitmap.h"
#include "pfc/jobs.h"
#include "pfc/chrono.h"
#include "pfc/cuda_helper.h"
#include "pfc/shared.h"
#include "pfc/colors.h"
#include "gpu_kernel.cuh"

#include <cstdio>
#include <filesystem>
#include <format>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>
#include <cmath>
#include <omp.h>

using namespace shared;

// -------------------------------------------------------------------------------------------------

// These values are tied to a specific job
constexpr size_t g_job_to_test{ 200 };
constexpr auto g_job_image_size{ debug_release(pfc::mebibyte{1.68}, pfc::mebibyte{144}) };
auto const g_job_total_size{ g_job_image_size * g_job_to_test };
constexpr double g_best_cpu_mibs{ 123.661 }; // 123.66 MiB/s, best parallelized CPU implementation for comparison
constexpr double g_best_gpu_mibs{ 7214.62 };

std::ostream nout{ nullptr };
auto& dout{ debug_release(std::cout, nout) };
auto const g_bmp_path{ "./bitmaps/"s };
auto const g_jbs_path{ "./jobs/"s };
constexpr bool g_save_images{ false };
constexpr size_t g_block_size{ 16 };

// -------------------------------------------------------------------------------------------------

std::string make_filename_bmp(std::size_t const t, std::size_t const n) {
	return std::format("{}fractal-test_{}-{:0{}}.bmp", g_bmp_path, t, n, 3);
}

pfc::bitmap make_bitmap(std::size_t const width, double const aspect_ratio) {
	return pfc::bitmap{ width, static_cast <std::size_t> (width / aspect_ratio) };
}

void throw_on_not_existent(std::string const& entity) {
	if (!std::filesystem::exists(entity))
		throw std::runtime_error{ "Filesystem entity '" + entity + "' does not exist." };
}

// -------------------------------------------------------------------------------------------------

// TODOS & info
// blocksize gridsize optimieren? -< occupancy in nsight
// loop unrolling
// streams (async copy and calculate same time)
// compute über der line compute bound /memory bound
// pipe util in compute, occupance für kernel config
// file - source, branching enable compile with source
// compute -> occupacy, punkt on top soll oben sein memory/compute bound, memory transfers etc.
// system -> zeigt memory, compute time an
// bissl screenshots, vergleiche auch wenn schlechter
void process_jobs_with_cuda(const pfc::jobs<real_t>& jobs, pfc::bitmap& output) {
	auto const image_size_bytes = output.size_bytes();
	auto const image_size = static_cast<int64_t>(output.size());
	uint16_t const height = static_cast<uint16_t>(output.height());

	pixel_t* h_pixels = output.data();
	// Use uint8_t for iterations to save memory and bandwidth
	uint8_t* h_iterations{ nullptr }; cuda::check(cudaMallocHost(&h_iterations, image_size));
	uint8_t* dp_iterations{ nullptr }; cuda::check(cudaMalloc(&dp_iterations, image_size));

	// Set up kernel parameters depending on image size
	dim3 const blockSize(g_block_size, g_block_size);
	dim3 const gridSize((g_width + blockSize.x - 1) / blockSize.x,
		(height + blockSize.y - 1) / blockSize.y);

	for (std::size_t i{}; auto const& [ll, ur, cp, wh] : jobs) {
		cuda::check(call_mandelbrot_kernel(gridSize, blockSize, raw_pointer(dp_iterations), height, 
			ll, 
			ur)
		);

		// Copy memory back
		cuda::check(cudaMemcpyAsync(
			h_iterations,
			raw_pointer(dp_iterations),
			image_size,
			cudaMemcpyDeviceToHost
		));

		cudaDeviceSynchronize();

		// Convert iterations to pixels on CPU
		#pragma omp parallel for
		for (int64_t i{}; i < image_size; ++i)
			h_pixels[i] = pfc::colors::g_color_map[h_iterations[i]];

		if (g_save_images) {
			std::string filename = make_filename_bmp(g_job_to_test, i++);
			output.to_file(filename);
		}

		dout << "Finished image " << i << " of job " << g_job_to_test << std::endl;
	}

	cuda::check(cudaFreeHost(h_iterations));
	cuda::check(cudaFree(dp_iterations));
}

void checked_main([[maybe_unused]] std::span <std::string_view const> const args) {
	int count{ 0 }; cuda::check(cudaGetDeviceCount(&count));

	if (0 < count) {
		for (int d{ 0 }; d < count; ++d) {
			cudaDeviceProp prop{}; cuda::check(cudaGetDeviceProperties(&prop, d));

			auto const cc{ std::format("{}.{}", prop.major, prop.minor) };
			pfc::byte const memory{ prop.totalGlobalMem };
			auto const threads{ prop.multiProcessorCount * prop.maxThreadsPerMultiProcessor };

			std::cout << std::format(
				"device #{}: '{}', cc {}, {:.1f} GiB, {} Threads\n", d, prop.name, cc, pfc::gibibyte{ memory }.count(), threads
			);
		}

		auto const [api, driver, runtime] {cuda::versions()};

		std::cout << std::format(
			"\n"
			"CUDA runtime version {}.\n"
			"Driver supports up to runtime version {}.\n"
			"CUDA API version {}.\n"
			"\n",
			api, driver, runtime) << std::flush;

		cuda::check(cudaSetDevice(0));

		auto const jobs = pfc::jobs<real_t>{ g_jbs_path + pfc::jobs<>::make_filename(g_job_to_test) };
		if (jobs.size() != g_job_to_test)
			throw std::runtime_error{ "Job size mismatch. File may not be found." };
		auto image = make_bitmap(g_width, jobs.aspect_ratio());

		auto const duration = pfc::timed_run([&jobs, &image] {
			process_jobs_with_cuda(jobs, image);
			cuda::check(cudaDeviceSynchronize()); // Wait for all kernels to finish
			});

		auto const in_seconds = pfc::to_seconds(duration);
		auto const mibs = g_job_total_size.count() / in_seconds;
		std::cout << "Job " << g_job_to_test << " result:\n"
			<< "Seconds: " << in_seconds << '\n'
			<< "MiB/s: " << mibs << '\n'
			<< "Speedup (best CPU): " << ((g_best_cpu_mibs > 0) ? mibs / g_best_cpu_mibs : -1) << '\n'
			<< "Speedup (best GPU): " << ((g_best_gpu_mibs > 0) ? mibs / g_best_gpu_mibs : -1) << '\n' << std::flush;
	}

	cuda::check(cudaDeviceReset()); // For profiling
}

int main(int const argc, char const* const* argv) {
	try {
		std::vector <std::string_view> const args(argv, argv + argc); checked_main(args);
	}
	catch (std::exception const& x) {
		std::cerr << "Error '" << x.what() << "'\n";
	}
	return 0;
}