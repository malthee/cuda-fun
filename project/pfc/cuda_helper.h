#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdexcept>
#include <format>


namespace cuda {

    void check(cudaError_t const e) {
        if (e != cudaSuccess)
            throw std::runtime_error{ cudaGetErrorName(e) };
    }

    auto versions() noexcept {
        auto const to_string{ [](std::integral auto const v) {
           return std::format("{}.{}", v / 1000, v % 1000 / 10);
        } };

        int version_driver{}; check(cudaDriverGetVersion(&version_driver));
        int version_runtime{}; check(cudaRuntimeGetVersion(&version_runtime));

        return std::make_tuple(
            to_string(CUDART_VERSION),
            to_string(version_driver),
            to_string(version_runtime)
        );
    }

    void free(auto*& dp) {
        if (dp)
            check(cudaFree(dp));

        dp = nullptr;
    }

    template <typename T> [[nodiscard]] T* malloc(std::size_t const size) {
        T* dp{}; check(cudaMalloc(&dp, sizeof(T) * size)); return dp;
    }

    template <typename T> [[nodiscard]] auto make_unique(std::size_t const size) {
        return std::unique_ptr <T[], decltype (&free <T>)> {malloc <T>(size), free <T>};
    }
}