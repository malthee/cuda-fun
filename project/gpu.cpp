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
constexpr double g_best_gpu_mibs{ 14467.9 };

std::ostream nout{ nullptr };
auto& dout{ debug_release(std::cout, nout) };
auto const g_bmp_path{ "./bitmaps/"s };
auto const g_jbs_path{ "./jobs/"s };

constexpr bool g_save_images{ false }; // For debugging, check if images still work
constexpr int g_block_size{ 16 };
constexpr int g_cuda_streams{ 6 };
constexpr int g_buffer_count{ 6 };

// -------------------------------------------------------------------------------------------------

std::string make_filename_bmp(std::size_t const t, std::size_t const n) {
	return std::format("{}fractal-test_{}-{:0{}}.bmp", g_bmp_path, t, n, 3);
}

// -------------------------------------------------------------------------------------------------

struct job_event_info {
	int job;
	bool is_job_complete; // To avoid overwriting the same buffer
	cudaEvent_t event;
};

struct cuda_resources {
	int64_t image_size; // Size of the image (width * height)
	uint16_t image_height; // Height of the image
	real_t   inv_height; // Inverse of the image height (to avoid division in the kernel)
	uint8_t* h_iterations[g_cuda_streams][g_buffer_count]; // Host memory for iterations (pinned), per stream, buffered
	uint8_t* dp_iterations[g_cuda_streams][g_buffer_count]; // Device memory for iterations, per stream, buffered
	pixel_t* h_pixels[g_cuda_streams][g_buffer_count]; // Host memory for pixels (pinned), per stream, buffered
	cudaStream_t streams[g_cuda_streams]; // CUDA streams
	job_event_info event_info[g_cuda_streams][g_buffer_count]; // Event info for each stream and buffer. Avoids overwriting the same buffer, keeps track of memcpy events.
};

static cuda_resources init_cuda_resources(uint16_t const height) {
	cuda_resources res{};
	res.image_height = height;
	res.image_size = static_cast<uint64_t>(g_width) * height;
	res.inv_height = static_cast<real_t>(1.0) / height;

	// Allocate for each stream, buffered
	for (int s{}; s < g_cuda_streams; ++s) {
		cuda::check(cudaStreamCreate(&res.streams[s]));

		for (int b{}; b < g_buffer_count; ++b) {
			// Allocate device memory
			cuda::check(cudaMalloc(&res.dp_iterations[s][b], res.image_size));

			// Allocate pinned host memory for iterations and pixels
			cuda::check(cudaMallocHost(&res.h_iterations[s][b], res.image_size));
			cuda::check(cudaMallocHost(&res.h_pixels[s][b], res.image_size * sizeof(pixel_t)));

			// Initialize event info
			cuda::check(cudaEventCreate(&res.event_info[s][b].event));
			res.event_info[s][b].job = -1;
			res.event_info[s][b].is_job_complete = true;
		}
	}

	return res;
}

static void free_cuda_resources(cuda_resources& res) {
	for (int s{}; s < g_cuda_streams; ++s) {
		for (int b{}; b < g_buffer_count; ++b) {
			// Free device memory
			cuda::check(cudaFree(res.dp_iterations[s][b]));

			// Free pinned host memory
			cuda::check(cudaFreeHost(res.h_iterations[s][b]));
			cuda::check(cudaFreeHost(res.h_pixels[s][b]));

			// Destroy the event
			cuda::check(cudaEventDestroy(res.event_info[s][b].event));
		}

		// Destroy the stream
		cuda::check(cudaStreamDestroy(res.streams[s]));
	}
}

static void process_buffer(cuda_resources& res, int stream, int buffer, int job) {
	auto const h_iterations = res.h_iterations[stream][buffer];
	pixel_t* h_pixels = res.h_pixels[stream][buffer];

	// Convert iterations to pixels parallel on CPU
	#pragma omp parallel for
	for (int64_t j = 0; j < res.image_size; ++j) {
		h_pixels[j] = pfc::colors::g_color_map[h_iterations[j]];
	}

	// Save image if required
	if (g_save_images) {
		std::string filename = make_filename_bmp(g_job_to_test, job);
		auto image = pfc::bitmap(g_width, res.image_height, std::span{ h_pixels, static_cast<size_t>(res.image_size) }, false);
		image.to_file(filename);
	}

	dout << "Finished image " << job << " of stream " << stream << std::endl;
}

static void process_jobs_with_cuda(const pfc::jobs<real_t>& jobs, cuda_resources& res) {
	// Set up kernel parameters depending on image size
	dim3 const blockSize(g_block_size, g_block_size);
	dim3 const gridSize((g_width + blockSize.x - 1) / blockSize.x,
		(res.image_height + blockSize.y - 1) / blockSize.y);

	// Process each job (image)
	for (int i{}; auto const& [ll, ur, cp, wh] : jobs) {
		auto s{ i % g_cuda_streams }; // Determine the stream index doing this job
		auto b{ (i / g_cuda_streams) % g_buffer_count }; // Determine the buffer index for this job
		auto& job_event { res.event_info[s][b] }; // Get the event info for this job

		// Ensure the buffer is ready for a new job
		if (!job_event.is_job_complete) {
			dout << "Waiting for buffer " << b << " of stream " << s << " to finish job " << job_event.job << std::endl;
			cuda::check(cudaEventSynchronize(job_event.event));
			process_buffer(res, s, b, job_event.job);
			job_event.is_job_complete = true;
		}

		// Update job number and mark the job as incomplete
		job_event.job = i;
		job_event.is_job_complete = false;

		// Calculate iterations (mandelbrot) index for color
		cuda::check(call_mandelbrot_kernel(gridSize,
			blockSize, 
			res.streams[s],
			res.dp_iterations[s][b], 
			res.image_height,
			res.inv_height,
			ll,
			ur
		));

		// MemcpyAsync result to host memory
		cuda::check(cudaMemcpyAsync(
			res.h_iterations[s][b],
			res.dp_iterations[s][b],
			res.image_size,
			cudaMemcpyDeviceToHost,
			res.streams[s]
		));	

		// Record copy event for this job
		cuda::check(cudaEventRecord(job_event.event, res.streams[s]));
		++i;
	}

	// Process events and buffers
	for (int s{}; s < g_cuda_streams; ++s) {
		for (int b{}; b < g_buffer_count; ++b) {
			auto& job_event{ res.event_info[s][b] };
			cuda::check(cudaEventSynchronize(job_event.event));
			process_buffer(res, s, b, job_event.job);
			job_event.is_job_complete = true;
		}
	}
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
			"\n"
			"Job {}.\n"
			"Stream count: {}.\n"
			"Buffer count: {}.\n"
			"Block size: {}.\n"
			"Save images: {}.\n"
			"\n"
			,api, driver, runtime, g_job_to_test, g_cuda_streams, g_buffer_count, g_block_size, g_save_images) << std::flush;

		cuda::check(cudaSetDevice(0));

		auto const jobs = pfc::jobs<real_t>{ g_jbs_path + pfc::jobs<>::make_filename(g_job_to_test) };
		if (jobs.size() != g_job_to_test)
			throw std::runtime_error{ "Job size mismatch. File may not be found." };
		auto const image_height = static_cast<uint16_t>(g_width / jobs.aspect_ratio());

		auto resources = init_cuda_resources(image_height);
		auto const duration = pfc::timed_run([&jobs, &resources] {
			process_jobs_with_cuda(jobs, resources);
			cuda::check(cudaDeviceSynchronize()); // Wait for all kernels to finish
			});
		free_cuda_resources(resources);

		auto const in_seconds = pfc::to_seconds(duration);
		auto const mibs = g_job_total_size.count() / in_seconds;
		std::cout << "Seconds: " << in_seconds << '\n'
			<< "MiB/s: " << mibs << '\n'
			<< "Speedup (best CPU): " << ((g_best_cpu_mibs > 0) ? mibs / g_best_cpu_mibs : -1) << '\n'
			<< "Speedup (best GPU): " << ((g_best_gpu_mibs > 0) ? mibs / g_best_gpu_mibs : -1) << '\n' << std::flush;
	}

	cuda::check(cudaDeviceReset()); // For profiling
	// Avoid closing instantly
	std::cout << "Press enter to exit." << std::endl;
	std::cin.get();
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