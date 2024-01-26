#include "pfc/bitmap.h"
#include "pfc/jobs.h"
#include "pfc/chrono.h"
#include "pfc/cuda_helper.h"
#include "pfc/config.h"

#include <cstdio>
#include <filesystem>
#include <format>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>
#include <cmath>
#include "gpu_kernel.cuh"

using namespace config;

// -------------------------------------------------------------------------------------------------

template <typename U = std::ratio <1>, typename S = std::size_t> using memory_size = std::chrono::duration <S, U>;

using kibi = std::ratio <1024>;
using mebi = std::ratio <1024 * kibi::num>;
using gibi = std::ratio <1024 * mebi::num>;
using tebi = std::ratio <1024 * gibi::num>;
using pebi = std::ratio <1024 * tebi::num>;
using exbi = std::ratio <1024 * pebi::num>;

using     byte = memory_size <>;
using kibibyte = memory_size <kibi, float>;
using mebibyte = memory_size <mebi, float>;
using gibibyte = memory_size <gibi, float>;
using tebibyte = memory_size <tebi, float>;
using pebibyte = memory_size <pebi, float>;
using exbibyte = memory_size <exbi, float>;

auto operator "" _byte(unsigned long long const v) {
    return byte{ v };
}

auto operator "" _kibibyte(long double const v) {
    return kibibyte{ v };
}

auto operator "" _mebibyte(long double const v) {
    return mebibyte{ v };
}

auto operator "" _gibibyte(long double const v) {
    return gibibyte{ v };
}

// -------------------------------------------------------------------------------------------------

std::string make_filename_bmp (std::size_t const t, std::size_t const n) {
   return std::format ("{}fractal-test_{}-{:0{}}.bmp", g_bmp_path, t, n, 3);
}

pfc::bitmap make_bitmap(std::size_t const width, double const aspect_ratio) {
    return pfc::bitmap{ width, static_cast <std::size_t> (width / aspect_ratio) };
}

void throw_on_not_existent(std::string const& entity) {
    if (!std::filesystem::exists(entity))
        throw std::runtime_error{ "Filesystem entity '" + entity + "' does not exist." };
}

// -------------------------------------------------------------------------------------------------

void process_jobs_with_cuda(bool save_images) {
    using pixel_t = pfc::bmp::pixel_t;
    const auto jobs = pfc::jobs{ g_jbs_path + pfc::jobs<>::make_filename(job_to_test) };

    // ausgehen jedes image im job same size? yes
    // blocksize gridsize optimieren? -< occupancy in nsight
    // vergleichen smart pointers? nah
    // prerechnen von colors? -> on cpu return not pixels but iterations
    // zurück zu host kopy in measurement?
    // calc color on cpu
    // loop unrolling
    // streams (async copy and calculate same time)


    for (std::size_t i{}; auto const& [ll, ur, cp, wh] : jobs) {
        auto image = make_bitmap(g_width, jobs.aspect_ratio());
        auto image_size = image.size_bytes();
        auto height = image.height();
        pixel_t* h_pixels = image.data();

#if !defined USE_SMART_POINTERS_ON_DEVICE
        pixel_t* dp_pixels{ nullptr }; cuda::check(cudaMalloc(&dp_pixels, image_size));
#else
        auto dp_pixels { cuda::make_unique<pixel_t>(image_size) };
#endif

        // Set up kernel parameters
        dim3 blockSize(16, 16);
        dim3 gridSize((g_width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

        cuda::check(call_mandelbrot_kernel(gridSize, blockSize, raw_pointer(dp_pixels), g_width, height, ll, ur));
        std::cout << "Finished image " << i << " of job " << job_to_test-1 << std::endl;
    
        cuda::check(cudaMemcpy(
            h_pixels,
            raw_pointer(dp_pixels),
            image_size,
            cudaMemcpyDeviceToHost
        ));

#if !defined USE_SMART_POINTERS_ON_DEVICE
        cuda::check(cudaFree(dp_pixels));
#endif

        if (save_images) {
            std::string filename = make_filename_bmp(job_to_test, i++);
            image.to_file(filename);
        }
    }

    cuda::check(cudaDeviceReset());
}

void checked_main([[maybe_unused]] std::span <std::string_view const> const args) {
    int count{ 0 }; cuda::check(cudaGetDeviceCount(&count));

    if (0 < count) {
        for (int d{ 0 }; d < count; ++d) {
            cudaDeviceProp prop{}; cuda::check(cudaGetDeviceProperties(&prop, d));

            auto const cc{ std::format("{}.{}", prop.major, prop.minor) };
            byte const memory{ prop.totalGlobalMem };
            auto const threads{ prop.multiProcessorCount * prop.maxThreadsPerMultiProcessor };

            std::cout << std::format(
                "device #{}: '{}', cc {}, {:.1f} GiB, {} Threads\n", d, prop.name, cc, gibibyte{ memory }.count(), threads
            );
        }

        auto const [api, driver, runtime] {cuda::versions()};

        std::cout << std::format(
            "\n"
            "CUDA runtime version {}.\n"
            "Driver supports up to runtime version {}.\n"
            "CUDA API version {}.\n"
            "\n",
            api, driver, runtime);

        cuda::check(cudaSetDevice(0));

        process_jobs_with_cuda(true);
    }

    cuda::check(cudaDeviceReset());
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