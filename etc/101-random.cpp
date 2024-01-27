//       $Id: 101-random.cpp 47923 2023-12-06 11:27:55Z p20068 $
//      $URL: https://svn01.fh-hagenberg.at/bin/cepheiden/pfc/trunk/101/src/101-random.cpp $
// $Revision: 47923 $
//     $Date: 2023-12-06 12:27:55 +0100 (Mi., 06 Dez 2023) $
//   $Author: p20068 $
//   Creator: Peter Kulczycki
//  Creation: November 6, 2020
// Copyright: (c) 2021 Peter Kulczycki (peter.kulczycki<AT>fh-hagenberg.at)
//   License: This document contains proprietary information belonging to
//            University of Applied Sciences Upper Austria, Campus Hagenberg.
//            It is distributed under the Boost Software License (see
//            https://www.boost.org/users/license.html).

// template <typename E = std::mt19937_64> int get_random_uniform (int const l, int const u) {
//    static E engine {std::random_device {} ()}; return std::uniform_int_distribution {l, u} (engine);
// }

#include "pfc/random.h"

#include <iostream>

using namespace std::chrono_literals;

int main_c () {
   using sec = std::chrono::duration <double>;

   for (std::size_t i {0}; i < 10; ++i)
      std::cout << "pfc::get_random_uniform (1, 6):             " << pfc::get_random_uniform (1, 6)             << '\n'
                << "pfc::get_random_uniform (0ms, 100ms):       " << pfc::get_random_uniform (0ms, 100ms)       << '\n'
                << "pfc::get_random_normal (0.0, 0.1):          " << pfc::get_random_normal (0.0, 0.1)          << '\n'
                << "pfc::get_random_normal (sec {0}, sec {10}): " << pfc::get_random_normal (sec {0}, sec {10}) << "\n\n";
   return 0;
}
