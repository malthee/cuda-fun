/// General configuration and utility functions for mandelbrot.

#pragma once

#include "complex.h"
#include "bitmap.h"

#include <ratio>
#include <filesystem>
#include <chrono>
#include <string>
#include <stdexcept>
#include <string_view>
#include <iostream>

#undef  USE_SMART_POINTERS_ON_DEVICE
#define USE_SMART_POINTERS_ON_DEVICE

using namespace std::string_literals;

namespace config {

template <typename P> constexpr auto raw_pointer(P& p) noexcept {
    if constexpr (std::is_pointer_v <P>)
        return p;
    else
        return p.get();
}

// -------------------------------------------------------------------------------------------------

template <typename D, typename R> [[nodiscard]] constexpr auto&& debug_release([[maybe_unused]] D&& d, [[maybe_unused]] R&& r) noexcept {
#if defined _debug
    return std::forward <D>(d);
#else
    return std::forward <R>(r);
#endif
}

// -------------------------------------------------------------------------------------------------

using real_t = float; 
using complex_t = pfc::complex<real_t>;

__constant__ constexpr  std::size_t g_colors{ debug_release<std::size_t, std::size_t>(64, 128) };
__constant__ constexpr  real_t      g_epsilon{ 0.00001f };
__constant__ constexpr  real_t      g_infinity{ 4 };
__constant__ constexpr  std::size_t g_width{ debug_release<std::size_t, std::size_t>(1024, 8192) };
__constant__ constexpr  int job_to_test{ 16 };

auto const g_bmp_path{ "./bitmaps/"s };
auto const g_jbs_path{ "./jobs/"s };

};