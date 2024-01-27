//       $Id: HW.cpp 46868 2023-03-05 19:11:14Z p20068 $
//      $URL: https://svn01.fh-hagenberg.at/bin/cepheiden/Inhalt/HPC/GPU/HW/src/HW.cpp $
// $Revision: 46868 $
//     $Date: 2023-03-05 20:11:14 +0100 (So., 05 Mär 2023) $
//   $Author: p20068 $
//   Creator: Peter Kulczycki
//  Creation: October 28, 2020
// Copyright: (c) 2021 Peter Kulczycki (peter.kulczycki<AT>fh-hagenberg.at)
//   License: This document contains proprietary information belonging to
//            University of Applied Sciences Upper Austria, Campus Hagenberg.
//            It is distributed under the Boost Software License (see
//            https://www.boost.org/users/license.html).

#if __has_include (<vld.h>)
   #include <vld.h>
#endif

#include "./HWKernel.h"

#include <chrono>
#include <concepts>
#include <format>
#include <iostream>
#include <memory>
#include <ratio>
#include <span>
#include <stdexcept>
#include <string_view>
#include <string>
#include <type_traits>
#include <vector>

using namespace std::string_view_literals;

#undef  USE_SMART_POINTERS_ON_DEVICE
#define USE_SMART_POINTERS_ON_DEVICE

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

auto operator "" _byte (unsigned long long const v) {
   return byte {v};
}

auto operator "" _kibibyte (long double const v) {
   return kibibyte {v};
}

auto operator "" _mebibyte (long double const v) {
   return mebibyte {v};
}

auto operator "" _gibibyte (long double const v) {
   return gibibyte {v};
}

// -------------------------------------------------------------------------------------------------

template <typename P> constexpr auto raw_pointer (P & p) noexcept {
   if constexpr (std::is_pointer_v <P>)
      return p;
   else
      return p.get ();
}

// -------------------------------------------------------------------------------------------------

namespace cuda {

void check (cudaError_t const e) {
   if (e != cudaSuccess)
      throw std::runtime_error {cudaGetErrorName (e)};
}

auto versions () noexcept {
   auto const to_string {[] (std::integral auto const v) {
      return std::format ("{}.{}", v / 1000, v % 1000 / 10);
   }};

   int version_driver  {}; check (cudaDriverGetVersion  (&version_driver));
   int version_runtime {}; check (cudaRuntimeGetVersion (&version_runtime));

   return std::make_tuple (
      to_string (CUDART_VERSION),
      to_string (version_driver),
      to_string (version_runtime)
   );
}

// -------------------------------------------------------------------------------------------------

#if defined USE_SMART_POINTERS_ON_DEVICE

void free (auto * & dp) {
   if (dp)
      check (cudaFree (dp));

   dp = nullptr;
}

template <typename T> [[nodiscard]] T * malloc (std::size_t const size) {
   T * dp {}; check (cudaMalloc (&dp, sizeof (T) * size)); return dp;
}

template <typename T> [[nodiscard]] auto make_unique (std::size_t const size) {
   return std::unique_ptr <T [], decltype (&free <T>)> {malloc <T> (size), free <T>};
}

#endif   // USE_SMART_POINTERS_ON_DEVICE

// -------------------------------------------------------------------------------------------------

}   // namespace cuda

void checked_main ([[maybe_unused]] std::span <std::string_view const> const args) {
   int count {0}; cuda::check (cudaGetDeviceCount (&count));

   if (0 < count) {
      for (int d {0}; d < count; ++d) {
         cudaDeviceProp prop {}; cuda::check (cudaGetDeviceProperties (&prop, d));

         auto const cc      {std::format ("{}.{}", prop.major, prop.minor)};
         byte const memory  {prop.totalGlobalMem};
         auto const threads {prop.multiProcessorCount * prop.maxThreadsPerMultiProcessor};

         std::cout << std::format (
            "device #{}: '{}', cc {}, {:.1f} GiB, {} Threads\n", d, prop.name, cc, gibibyte {memory}.count (), threads
         );
      }

      auto const [api, driver, runtime] {cuda::versions()};

      std::cout << std::format (
         "\n"
         "CUDA runtime version {}.\n"
         "Driver supports up to runtime version {}.\n"
         "CUDA API version {}.\n"
         "\n",
      api, driver, runtime);

      cuda::check (cudaSetDevice (0));

      auto const src    {"Lorem ipsum dolor sit amet, consetetur sadipscing elitr ..."sv};
      auto const size   {std::size (src) + 1};
      auto const hp_dst {std::make_unique <char []> (size)};

      std::cout << "source: '" << src << "'\n"
                   "copy:   ";

      #if !defined USE_SMART_POINTERS_ON_DEVICE
         char * dp_src {nullptr}; cuda::check (cudaMalloc (&dp_src, size));
         char * dp_dst {nullptr}; cuda::check (cudaMalloc (&dp_dst, size));
      #else
         auto dp_src {cuda::make_unique <char> (size)};
         auto dp_dst {cuda::make_unique <char> (size)};
      #endif

      cuda::check (cudaMemcpy (
         raw_pointer (dp_src),
         std::data (src),
         size,
         cudaMemcpyHostToDevice
      ));

      auto const tib {32u};
      auto const big {static_cast <unsigned> ((size + tib - 1) / tib)};

      cuda::check (call_copy_string_kernel (
         big,
         tib,
         raw_pointer (dp_dst),
         raw_pointer (dp_src),
         static_cast <unsigned> (size)
      ));

      cuda::check (cudaMemcpy (
         raw_pointer (hp_dst),
         raw_pointer (dp_dst),
         size,
         cudaMemcpyDeviceToHost
      ));

      #if !defined USE_SMART_POINTERS_ON_DEVICE
         cuda::check (cudaFree (dp_dst));
         cuda::check (cudaFree (dp_src));
      #endif

      std::cout << "'" << hp_dst << "'\n\n";
   }

   cuda::check (cudaDeviceReset ());
}

int main_o (int const argc, char const * const * argv) {
   try {
      std::vector <std::string_view> const args (argv, argv + argc); checked_main (args);

   } catch (std::exception const & x) {
      std::cerr << "Error '" << x.what () << "'\n";
   }
   return 0;
}

/*
device #0: 'Quadro RTX 5000', cc 7.5, 16.0 GiB, 49152 Threads
device #1: 'Quadro RTX 5000', cc 7.5, 16.0 GiB, 49152 Threads

CUDA runtime version 12.1.
Driver supports up to runtime version 12.1.
CUDA API version 12.1.

source: 'Lorem ipsum dolor sit amet, consetetur sadipscing elitr ...'
copy:   'Lorem ipsum dolor sit amet, consetetur sadipscing elitr ...'
*/
