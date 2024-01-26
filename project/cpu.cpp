//       $Id: 101-fractals.cpp 47900 2023-12-01 09:40:55Z p20068 $
//      $URL: https://svn01.fh-hagenberg.at/bin/cepheiden/Inhalt/HPC/Threads/fractals/src/101-fractals.cpp $
// $Revision: 47900 $
//     $Date: 2023-12-01 10:40:55 +0100 (Fr., 01 Dez 2023) $
//   $Author: p20068 $
//   Creator: Peter Kulczycki
//  Creation: November 20, 2020
// Copyright: (c) 2021 Peter Kulczycki (peter.kulczycki<AT>fh-hagenberg.at)
//   License: This document contains proprietary information belonging to
//            University of Applied Sciences Upper Austria, Campus Hagenberg.
//            It is distributed under the Boost Software License (see
//            https://www.boost.org/users/license.html).

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
#include "threads_test.h"

using namespace std::string_literals;

// -------------------------------------------------------------------------------------------------

template <typename T> concept is_numeric = std::is_integral_v <T> || std::is_floating_point_v <T>;

template <typename D, typename R> [[nodiscard]] constexpr auto && debug_release ([[maybe_unused]] D && d, [[maybe_unused]] R && r) noexcept {
   #if defined _DEBUG
      return std::forward <D> (d);
   #else
      return std::forward <R> (r);
   #endif
}

pfc::bitmap make_bitmap (std::size_t const width, double const aspect_ratio) {
   return pfc::bitmap {width, static_cast <std::size_t> (width / aspect_ratio)};
}

void throw_on_not_existent (std::string const & entity) {
   if (!std::filesystem::exists (entity))
      throw std::runtime_error {"Filesystem entity '" + entity + "' does not exist."};
}

template <is_numeric T> pfc::byte_t to_byte (T const value) {
   return static_cast <pfc::byte_t> (value);
}

// -------------------------------------------------------------------------------------------------

using real_t    = float; // also check double
using complex_t = pfc::complex<real_t>;

constinit std::size_t g_colors   {debug_release (64, 128)};
constinit real_t      g_epsilon  {0.00001f};
constinit real_t      g_infinity {4};
constinit std::size_t g_width    {debug_release (1024, 8192)};
constinit auto job_to_test       {16};

std::ostream nout {nullptr};
auto &       dout {debug_release (std::cout, nout)};

auto const g_bmp_path {"./bitmaps/"s};
auto const g_jbs_path {"./jobs/"s};

auto const hardware_concurrency {std::thread::hardware_concurrency()};
auto const max_threads          {hardware_concurrency * 10};
auto const max_tasks            {max_threads * 2}; // Allow bit more tasks than threads

// -------------------------------------------------------------------------------------------------

pfc::bmp::pixel_t iterate (complex_t const c) {
   std::size_t i {};
   complex_t   z {};

   while ((i++ < g_colors) && (z.norm() < g_infinity))
      z = pfc::square(z) + c;

   ++i;   // gives a nice effect

   return {.green = to_byte (1.0 * i / g_colors * 255)}; // Optimize this lookup
}

static inline void fractal (pfc::bitmap& bmp, std::size_t const start_height, std::size_t const stop_height, complex_t const ll, complex_t const ur) {
   auto const complex_width  {(ur - ll).real};
   auto const complex_height {(ur - ll).imag};

   const auto width = bmp.width();
   const auto height = bmp.height();
   auto const dx {complex_width  / width };
   auto const dy {complex_height / height };

   size_t y_width {0};
   auto c {ll};
   auto* pixel_data = bmp.data();

   for (std::size_t y {start_height}; y < stop_height; c.imag += dy, ++y) {
       c.real = ll.real;

      for (std::size_t x {0}; x < width; c.real += dx, ++x)
          *(pixel_data + y_width + x) = iterate (c);
   
      y_width += width;
   }
}

std::string make_filename_bmp (std::size_t const t, std::size_t const n) {
   return std::format ("{}fractal-test_{}-{:0{}}.bmp", g_bmp_path, t, n, 3);
}

// -------------------------------------------------------------------------------------------------

static inline void measure_threads(int job_nr) {
    std::ofstream output("threads_results.csv");
    output << "batch_size;thread_count;execution_time\n";

    auto jobs = pfc::jobs{ g_jbs_path + pfc::jobs<>::make_filename(job_nr) };
    size_t job_size = jobs.size();
    auto images = std::vector<pfc::bitmap>(job_size);
    std::size_t height = static_cast<std::size_t>(g_width / jobs.aspect_ratio());

    // Allocate images
    for (auto& bmp : images) {
        bmp = make_bitmap(g_width, jobs.aspect_ratio());
    }

    for (unsigned int current_batch_size = 1; current_batch_size <= job_size; current_batch_size = std::min(current_batch_size * 2, static_cast<unsigned int>(job_size + 1))) {
        for (unsigned int current_thread_count = 1; current_thread_count <= max_threads; current_thread_count = std::ceil(current_thread_count * 1.1)) {
            if (current_thread_count < current_batch_size) {
				continue;
			}
            std::cout << "Batch size: " << current_batch_size << " Thread count: " << current_thread_count << "\n";
            
            auto duration = pfc::timed_run([&]() {
                for (std::size_t batch_start = 0; batch_start < job_size; batch_start += current_batch_size) {
                    std::vector<std::thread> batch_threads;

                    // Process a maximum of current_batch_size images at one time
                    for (std::size_t i = batch_start; i < std::min(batch_start + current_batch_size, job_size); ++i) {
                        batch_threads.emplace_back([&, i]() {
                            auto& bmp = images[i];
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
                    }

                    for (auto& t : batch_threads) {
                        if (t.joinable()) {
                            t.join();
                        }
                    }
                }
                });

            output << current_batch_size << ";" << current_thread_count << ";" << duration.count() << "\n";
        }
    }

    output.close();
}

static inline void measure_tasks(int job_nr) {
    std::ofstream output("tasks_results.csv");
    output << "batch_size;task_count;execution_time\n";

    auto jobs = pfc::jobs{ g_jbs_path + pfc::jobs<>::make_filename(job_nr) };
    size_t job_size = jobs.size();
    auto images = std::vector<pfc::bitmap>(job_size);
    std::size_t height = static_cast<std::size_t>(g_width / jobs.aspect_ratio());

    for (auto& bmp : images) {
        bmp = make_bitmap(g_width, jobs.aspect_ratio());
    }

    for (unsigned int current_batch_size = 1; current_batch_size <= job_size; current_batch_size = std::min(current_batch_size * 2, static_cast<unsigned int>(job_size + 1))) {
        for (unsigned int current_task_count = 115; current_task_count <= max_tasks; current_task_count = std::ceil(current_task_count * 1.1)) {
            if (current_task_count < current_batch_size) {
                continue;
            }
            std::cout << "Batch size: " << current_batch_size << " Task count: " << current_task_count << "\n";
            
            auto duration = pfc::timed_run([&]() {
                for (std::size_t batch_start = 0; batch_start < job_size; batch_start += current_batch_size) {
                    std::vector<std::future<void>> batch_futures;

                    for (std::size_t i = batch_start; i < std::min(batch_start + current_batch_size, job_size); ++i) {
                        batch_futures.push_back(std::async(std::launch::async, [&, i]() {
                            auto& bmp = images[i];
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
                            }
                            }));
                    }

                    for (auto& future : batch_futures) {
                        future.wait();
                    }
                }
                });

            output << current_batch_size << ";" << current_task_count << ";" << duration.count() << "\n";
            output.flush();
        }
    }

    output.close();
}

void checked_main ([[maybe_unused]] std::span <std::string_view const> const args) {
   std::cout << std::format ("I'm in {} mode ...\n", debug_release ("debug", "release"));
   std::cout << std::format ("I'm using a maximum of {} threads ...\n", max_threads);
   std::cout << std::format ("I'm using a maximum of {} tasks ...\n", max_tasks);
   std::cout << std::format ("I'm using job number {} ...\n", job_to_test);
   
   measure_threads(job_to_test);
   measure_tasks(job_to_test);
}

int main (int const argc, char const * const * const argv) {
   try {
      checked_main (std::vector <std::string_view> (argv, argv + argc));
      return EXIT_SUCCESS;

   } catch (...) {
      std::cerr << "ERROR\n" << std::flush;
      return EXIT_FAILURE;
   }
}
