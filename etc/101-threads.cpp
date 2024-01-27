//       $Id: 101-threads.cpp 47889 2023-11-29 15:13:44Z p20068 $
//      $URL: https://svn01.fh-hagenberg.at/bin/cepheiden/pfc/trunk/101/src/101-threads.cpp $
// $Revision: 47889 $
//     $Date: 2023-11-29 16:13:44 +0100 (Mi., 29 Nov 2023) $
//   $Author: p20068 $
//   Creator: Peter Kulczycki
//  Creation: November 6, 2020
// Copyright: (c) 2023 Peter Kulczycki (peter.kulczycki<AT>fh-hagenberg.at)
//   License: This document contains proprietary information belonging to
//            University of Applied Sciences Upper Austria, Campus Hagenberg.
//            It is distributed under the Boost Software License (see
//            https://www.boost.org/users/license.html).

#include <chrono>
#include <format>
#include <future>
#include <iostream>
#include <mutex>
#include <thread>

#define SYNCHRONIZEDx

int some_work_to_do (char const c) {
   using namespace std::chrono_literals;

   #if defined SYNCHRONIZED
      static std::mutex mutex {};        // since C++11, the creation of static objects is thread safe
      std::unique_lock  lock  {mutex};   //
   #endif

   for (int i {0}; i < 10; ++i) {
      std::cout << c; std::this_thread::sleep_for (100ms);
   }

   return c;
}

void test_with_tasks () {
   auto f1 {std::async (std::launch::async, some_work_to_do, 'A')};
   auto f2 {std::async (std::launch::async, some_work_to_do, 'B')};
   auto f3 {std::async (std::launch::async, some_work_to_do, 'C')};

   std::cout << std::format ("\nf3: {}\nf2: {}\nf1: {}\n", f3.get (), f2.get (), f1.get ());
}

void test_with_threads () {
   auto f {[] (char const c) {
      return some_work_to_do (c);
   }};

   f ('x'); std::cout << '\n' << std::flush;

   std::vector <std::thread> pool;

   pool.emplace_back (f, '1');
   pool.emplace_back (some_work_to_do, '2');
   pool.emplace_back ([] {
      return some_work_to_do ('3');
   });

   for (auto & t : pool)
      if (t.joinable ())
         t.join ();

   std::cout << '\n' << std::flush;
}

int main_c () {
   test_with_threads ();
   test_with_tasks ();
   return 0;
}
