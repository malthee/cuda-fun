/// General configuration and utility functions for mandelbrot used both on the host and the device.

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
#include <array>

//#define USE_SMART_POINTERS_ON_DEVICE // -> Seems to be slower, disabled
//#define _debug

using namespace std::string_literals;

namespace shared {

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
using pixel_t = pfc::bmp::pixel_t;

__constant__ constexpr  size_t g_colors{ debug_release<size_t,size_t>(64, 128) };
__constant__ constexpr  real_t g_epsilon{ 0.00001f };
__constant__ constexpr  real_t g_infinity{ 4 };
__constant__ constexpr  uint16_t g_width{ debug_release<uint16_t, uint16_t>(1024, 8192) };

};