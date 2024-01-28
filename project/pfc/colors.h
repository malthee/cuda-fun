// Compile time calculation of all possible colors

#pragma once
#include "pfc/bitmap.h"
#include "pfc/shared.h"

#include <array>

namespace pfc { namespace colors {

using color_t = pfc::bmp::pixel_t;
using byte_t = pfc::byte_t;

constexpr color_t compute_color(std::size_t iteration, std::size_t max_iters) {
    return { .green = static_cast<byte_t>(1.0 * iteration / max_iters * 255) };
}

template<std::size_t N>
constexpr std::array<color_t, N> generate_color_map() {
    std::array<color_t, N> colormap{};
    for (std::size_t i = 0; i < N; ++i) {
        colormap[i] = compute_color(i, N);
    }
    return colormap;
}

constexpr auto g_color_map = generate_color_map<shared::g_colors>();

}
}