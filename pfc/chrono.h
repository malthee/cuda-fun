//       $Id: chrono.h 47889 2023-11-29 15:13:44Z p20068 $
//      $URL: https://svn01.fh-hagenberg.at/bin/cepheiden/pfc/trunk/pfc/inc/pfc/chrono.h $
// $Revision: 47889 $
//     $Date: 2023-11-29 16:13:44 +0100 (Mi., 29 Nov 2023) $
//   $Author: p20068 $
//   Creator: Peter Kulczycki
//  Creation: March 15, 2021
// Copyright: (c) 2023 Peter Kulczycki (peter.kulczycki<AT>fh-hagenberg.at)
//   License: This document contains proprietary information belonging to
//            University of Applied Sciences Upper Austria, Campus Hagenberg.
//            It is distributed under the Boost Software License (see
//            https://www.boost.org/users/license.html).

#pragma once

#undef  PFC_CHRONO_VERSION
#define PFC_CHRONO_VERSION "1.0.5"

#if defined __CUDACC__
   #pragma message ("PFC: Compiling 'chrono.h' with nvcc does not have any effect")
#else

#include <chrono>
#include <concepts>
#include <functional>
#include <type_traits>

namespace pfc { namespace details {

template <typename D> concept std_chrono_duration = requires {
   typename D::rep;
   typename D::period;
} && std::same_as <D, std::chrono::duration <typename D::rep, typename D::period>>;

}   // namespace details

using default_clock = std::chrono::high_resolution_clock;

template <typename SI> constexpr double to (std::chrono::duration <double, SI> const duration) {
   return duration.count ();
}

constexpr double to_minutes (std::chrono::duration <double, std::ratio <60>> const duration) {
   return duration.count ();
}

constexpr double to_seconds (std::chrono::duration <double> const duration) {   // period defaults to std::ratio <1>
   return duration.count ();
}

template <typename C = default_clock, typename ...A> auto timed_run (std::invocable <A...> auto && f, A && ...a) {
   auto const start {C::now ()};

   std::invoke (std::forward <std::remove_cvref_t <decltype (f)>> (f), std::forward <A> (a)...);

   return C::now () - start;
}

}   // namespace pfc

#endif   // __CUDACC__
