//       $Id: random.h 47889 2023-11-29 15:13:44Z p20068 $
//      $URL: https://svn01.fh-hagenberg.at/bin/cepheiden/pfc/trunk/pfc/inc/pfc/random.h $
// $Revision: 47889 $
//     $Date: 2023-11-29 16:13:44 +0100 (Mi., 29 Nov 2023) $
//   $Author: p20068 $
//   Creator: Peter Kulczycki
//  Creation: March 15, 2021
// Copyright: (c) 2021 Peter Kulczycki (peter.kulczycki<AT>fh-hagenberg.at)
//   License: This document contains proprietary information belonging to
//            University of Applied Sciences Upper Austria, Campus Hagenberg.
//            It is distributed under the Boost Software License (see
//            https://www.boost.org/users/license.html).

#pragma once

#undef  PFC_RANDOM_VERSION
#define PFC_RANDOM_VERSION "1.1.2"

#if defined __CUDACC__
   #pragma message ("PFC: Compiling 'random.h' with nvcc does not have any effect")
#else

#include <chrono>
#include <concepts>
#include <random>
#include <type_traits>

namespace pfc { namespace details {

template <typename T> constexpr bool is_std_random_engine_v {
   std::is_same_v <T, std::knuth_b>       ||
   std::is_same_v <T, std::minstd_rand>   ||
   std::is_same_v <T, std::minstd_rand0>  ||
   std::is_same_v <T, std::mt19937>       ||
   std::is_same_v <T, std::mt19937_64>    ||
   std::is_same_v <T, std::ranlux24>      ||
   std::is_same_v <T, std::ranlux24_base> ||
   std::is_same_v <T, std::ranlux48>      ||
   std::is_same_v <T, std::ranlux48_base>
};

template <typename T> concept arithmetic        = std::is_integral_v <T> || std::is_floating_point_v <T>;
template <typename T> concept std_random_engine = is_std_random_engine_v <T>;

template <std::integral I> auto make_uniform_distribution (I const a, I const b) {
   return std::uniform_int_distribution {a, b};
}

template <std::floating_point F> auto make_uniform_distribution (F const a, F const b) {
   return std::uniform_real_distribution {a, b};
}

}   // namespace details

using defaut_random_engine = std::mt19937_64;

template <details::std_random_engine E = defaut_random_engine> auto get_random_normal (std::floating_point auto const m, std::floating_point auto const s) {
   using F = std::common_type_t <std::remove_cvref_t <decltype (m)>, std::remove_cvref_t <decltype (s)>>;

   static auto engine       {E {std::random_device {} ()}};
   static auto distribution {std::normal_distribution <F> (m, s)};

   return distribution (engine);
}

template <details::std_random_engine E = defaut_random_engine> auto get_random_uniform (details::arithmetic auto const a, details::arithmetic auto const b) {
   using A = std::common_type_t <std::remove_cvref_t <decltype (a)>, std::remove_cvref_t <decltype (b)>>;

   static auto engine       {E {std::random_device {} ()}};
   static auto distribution {details::make_uniform_distribution <A> (a, b)};

   return distribution (engine);
}

template <details::std_random_engine E = defaut_random_engine, typename ...T> auto get_random_normal (std::chrono::duration <T...> const a, std::chrono::duration <T...> const b) {
   return std::chrono::duration <T...> {get_random_normal <E> (a.count (), b.count ())};
}

template <details::std_random_engine E = defaut_random_engine, typename ...T> auto get_random_uniform (std::chrono::duration <T...> const a, std::chrono::duration <T...> const b) {
   return std::chrono::duration <T...> {get_random_uniform <E> (a.count (), b.count ())};
}

}   // namespace pfc

#endif   // __CUDACC__
