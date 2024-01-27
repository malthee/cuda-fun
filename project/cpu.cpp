#include "pfc/bitmap.h"
#include "pfc/jobs.h"
#include "pfc/chrono.h"

#include <cstdio>
#include <filesystem>
#include <format>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>
#include <future>
#include <cmath>

using namespace std::string_literals;


// -------------------------------------------------------------------------------------------------

template <typename T> concept is_numeric = std::is_integral_v <T> || std::is_floating_point_v <T>;

template <typename D, typename R> [[nodiscard]] constexpr auto&& debug_release([[maybe_unused]] D&& d, [[maybe_unused]] R&& r) noexcept {
#if defined _DEBUG
	return std::forward <D>(d);
#else
	return std::forward <R>(r);
#endif
}

pfc::bitmap make_bitmap(std::size_t const width, double const aspect_ratio) {
	return pfc::bitmap{ width, static_cast <std::size_t> (width / aspect_ratio) };
}

void throw_on_not_existent(std::string const& entity) {
	if (!std::filesystem::exists(entity))
		throw std::runtime_error{ "Filesystem entity '" + entity + "' does not exist." };
}

template <is_numeric T> pfc::byte_t to_byte(T const value) {
	return static_cast <pfc::byte_t> (value);
}

// -------------------------------------------------------------------------------------------------

using real_t = float;
using complex_t = pfc::complex<real_t>;

constinit std::size_t g_colors{ debug_release(64, 128) };
constinit real_t g_epsilon{ 0.00001f };
constinit real_t g_infinity{ 4 };
constinit std::size_t g_width{ debug_release(1024, 8192) };
constinit int g_batch_sizes[]{ 1, 2, 8 };
constinit std::size_t g_job_to_test{ 200 };
constinit auto g_job_image_size{ pfc::mebibyte{144} };
auto const g_job_total_size{ g_job_image_size * g_job_to_test };
std::ostream nout{ nullptr };
auto& dout{ debug_release(std::cout, nout) };

auto const g_bmp_path{ "./bitmaps/"s };
auto const g_jbs_path{ "./jobs/"s };

constinit int max_threads[]{ 1, 2, 4, 8, 16, 50, 100, 200 };
constinit int max_tasks[]{ 1, 2, 4, 8, 16, 50, 100, 200, 400, 600 };

// -------------------------------------------------------------------------------------------------

pfc::bmp::pixel_t iterate(complex_t const c) {
	std::size_t i{};
	complex_t   z{};

	while ((i++ < g_colors) && (z.norm() < g_infinity))
		z = pfc::square(z) + c;

	++i;

	return { .green = to_byte(1.0 * i / g_colors * 255) };
}

static inline void fractal(pfc::bitmap& bmp, std::size_t const start_height, std::size_t const stop_height, complex_t const ll, complex_t const ur) {
	auto const complex_width{ (ur - ll).real };
	auto const complex_height{ (ur - ll).imag };

	const auto width = bmp.width();
	const auto height = bmp.height();
	auto const dx{ complex_width / width };
	auto const dy{ complex_height / height };

	size_t y_width{ 0 };
	auto c{ ll };
	auto* pixel_data = bmp.data();

	for (std::size_t y{ start_height }; y < stop_height; c.imag += dy, ++y) {
		c.real = ll.real;

		for (std::size_t x{ 0 }; x < width; c.real += dx, ++x)
			*(pixel_data + y_width + x) = iterate(c);

		y_width += width;
	}
}

std::string make_filename_bmp(std::size_t const t, std::size_t const n) {
	return std::format("{}fractal-test_{}-{:0{}}.bmp", g_bmp_path, t, n, 3);
}

// -------------------------------------------------------------------------------------------------

static inline void measure_threads() {
	std::ofstream output("threads_results.csv");
	output << "batch_size;thread_count;execution_time;MiBs\n";

	auto jobs = pfc::jobs{ g_jbs_path + pfc::jobs<>::make_filename(g_job_to_test) };
	size_t job_size = jobs.size();
	std::size_t height = static_cast<std::size_t>(g_width / jobs.aspect_ratio());

	for (size_t current_batch_size : g_batch_sizes) {
		auto images = std::vector<pfc::bitmap>(current_batch_size);
		for (auto& bmp : images) {
			bmp = make_bitmap(g_width, jobs.aspect_ratio());
		}
		
		for (auto current_thread_count : max_threads) {
			if (current_thread_count < current_batch_size) { // We tried the other combinations already in the previous loop
				continue;
			}

			std::cout << "Batch size: " << current_batch_size << " Thread count: " << current_thread_count << "\n";

			auto duration = pfc::timed_run([&]() {
				for (std::size_t batch_start = 0; batch_start < job_size; batch_start += current_batch_size) {
					std::vector<std::thread> batch_threads;

					// Process a maximum of current_batch_size images at one time
					for (std::size_t i = batch_start; i < std::min(batch_start + current_batch_size, job_size); ++i) {
						batch_threads.emplace_back([&, i]() {
							auto& bmp = images[i % current_batch_size];
							auto [ll, ur, cp, wh] = jobs[i];
							std::vector<std::thread> subsection_threads;

							std::size_t height_per_thread = height / current_thread_count;
							std::size_t leftover = height % current_thread_count;
							std::size_t start_height = 0;

							for (std::size_t j = 0; j < current_thread_count; ++j) {
								std::size_t stop_height = start_height + height_per_thread + (j < leftover ? 1 : 0);
								subsection_threads.emplace_back([&bmp, start_height, stop_height, ll, ur]() {
									fractal(bmp, start_height, stop_height, ll, ur);
									});
								start_height = stop_height;
							}

							for (auto& t : subsection_threads) {
								if (t.joinable()) {
									t.join();
								}
							}
							});


							std::cout << "Batch start: " << batch_start << " Threads start: " << i << "\n";
					}

					for (auto& t : batch_threads) {
						if (t.joinable()) {
							t.join();
						}
					}
				}
				});
			auto in_seconds = pfc::to_seconds(duration);

			output << current_batch_size << ";" << current_thread_count << ";" 
				<< in_seconds << ";" << g_job_total_size.count() / in_seconds << "\n";
			output.flush();
		}
	}

	output.close();
}

static inline void measure_tasks() {
	std::ofstream output("tasks_results.csv");
	output << "batch_size;task_count;execution_time;MiBs\n";

	auto jobs = pfc::jobs{ g_jbs_path + pfc::jobs<>::make_filename(g_job_to_test) };
	size_t job_size = jobs.size();
	std::size_t height = static_cast<std::size_t>(g_width / jobs.aspect_ratio());

	for (size_t current_batch_size : g_batch_sizes) {
		auto images = std::vector<pfc::bitmap>(current_batch_size);
		for (auto& bmp : images) {
			bmp = make_bitmap(g_width, jobs.aspect_ratio());
		}

		for (auto current_task_count : max_tasks) {
			if (current_task_count < current_batch_size) { // We tried the other combinations already in the previous loop
				continue;
			}
			
			std::cout << "Batch size: " << current_batch_size << " Task count: " << current_task_count << "\n";

			auto duration = pfc::timed_run([&]() {
				std::vector<std::future<void>> futures;

				for (std::size_t batch_start = 0; batch_start < job_size; batch_start += current_batch_size) {
					// Process a maximum of current_batch_size images at one time
					for (std::size_t i = batch_start; i < std::min(batch_start + current_batch_size, job_size); ++i) {
						if (futures.size() >= current_task_count) {
							futures.front().wait();
							futures.erase(futures.begin());
						}
						
						futures.push_back(std::async(std::launch::async, [&, i]() {
							auto& bmp = images[i % current_batch_size];
							auto [ll, ur, cp, wh] = jobs[i];
							std::vector<std::future<void>> subsection_futures;

							std::size_t height_per_task = height / current_task_count;
							std::size_t leftover = height % current_task_count;
							std::size_t start_height = 0;

							for (std::size_t j = 0; j < current_task_count; ++j) {
								std::size_t stop_height = start_height + height_per_task + (j < leftover ? 1 : 0);
								subsection_futures.push_back(std::async(std::launch::async, [&bmp, start_height, stop_height, ll, ur]() {
									fractal(bmp, start_height, stop_height, ll, ur);
									}));
								start_height = stop_height;
							}

							for (auto& future : subsection_futures) {
								future.wait();
								future.get();
							}
							}));

						std::cout << "Batch start: " << batch_start << " Futures start: " << i << "\n";
					}
				}

				// Wait for all tasks to complete
				for (auto& future : futures) {
					future.wait();
					future.get();
				}
				});

			auto in_seconds = pfc::to_seconds(duration);
			output << current_batch_size << ";" << current_task_count << ";"
				<< in_seconds << ";" << g_job_total_size.count() / in_seconds << "\n";
			output.flush();
		}
	}

	output.close();
}

void checked_main([[maybe_unused]] std::span <std::string_view const> const args) {
	std::cout << std::format("I'm in {} mode ...\n", debug_release("debug", "release"));
	std::cout << "I'm using the following thread counts: ";
	for (auto const t : max_threads)
		std::cout << t << ' ';
	std::cout << '\n';
	std::cout << "I'm using the following task counts: ";
	for (auto const t : max_tasks)
		std::cout << t << ' ';
	std::cout << '\n';
	std::cout << std::format("I'm using job number {} ...\n", g_job_to_test);

	//measure_threads();
	measure_tasks();
}

int main(int const argc, char const* const* const argv) {
	try {
		checked_main(std::vector <std::string_view>(argv, argv + argc));
		return EXIT_SUCCESS;

	}
	catch (...) {
		std::cerr << "ERROR\n" << std::flush;
		return EXIT_FAILURE;
	}
}
