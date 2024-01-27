//       $Id: 101-chrono.cpp 47889 2023-11-29 15:13:44Z p20068 $
//      $URL: https://svn01.fh-hagenberg.at/bin/cepheiden/pfc/trunk/101/src/101-chrono.cpp $
// $Revision: 47889 $
//     $Date: 2023-11-29 16:13:44 +0100 (Mi., 29 Nov 2023) $
//   $Author: p20068 $
//   Creator: Peter Kulczycki
//  Creation: November 7, 2020
// Copyright: (c) 2023 Peter Kulczycki (peter.kulczycki<AT>fh-hagenberg.at)
//   License: This document contains proprietary information belonging to
//            University of Applied Sciences Upper Austria, Campus Hagenberg.
//            It is distributed under the Boost Software License (see
//            https://www.boost.org/users/license.html).

#include "pfc/chrono.h"

#include <iostream>
#include <thread>

using namespace std::chrono_literals;

void calculate_something (std::chrono::milliseconds const duration) {
   std::this_thread::sleep_for (duration);
}

void print_elapsed (pfc::details::std_chrono_duration auto const elapsed) {
   std::cout
      << elapsed << '\n'
      << std::chrono::duration_cast <std::chrono::milliseconds> (elapsed) << '\n'
      << std::chrono::duration_cast <std::chrono::seconds> (elapsed) << '\n'
      << pfc::to <std::milli> (elapsed) << " ms\n"
      << pfc::to <std::kilo> (elapsed) << " ks\n"
      << pfc::to_seconds (elapsed) << " s\n"
      << pfc::to_minutes (elapsed) << " m\n";
}

int main_c () {
   print_elapsed (pfc::timed_run (calculate_something, 1s));
   return 0;
}
