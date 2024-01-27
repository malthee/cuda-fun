//       $Id: futures.cpp 47889 2023-11-29 15:13:44Z p20068 $
//      $URL: https://svn01.fh-hagenberg.at/bin/cepheiden/Inhalt/CPP/futures/src/futures.cpp $
// $Revision: 47889 $
//     $Date: 2023-11-29 16:13:44 +0100 (Mi., 29 Nov 2023) $
//   $Author: p20068 $
//   Creator: Peter Kulczycki
//  Creation: October 19, 2021
// Copyright: (c) 2023 Peter Kulczycki (peter.kulczycki<AT>fh-hagenberg.at)
//   License: This document contains proprietary information belonging to
//            University of Applied Sciences Upper Austria, Campus Hagenberg.
//            It is distributed under the Boost Software License (see
//            https://www.boost.org/users/license.html).

#include <chrono>
#include <format>
#include <future>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <type_traits>

using namespace std::chrono_literals;

// -------------------------------------------------------------------------------------------------

void sleep_for (std::chrono::milliseconds const duration) {
   std::this_thread::sleep_for (duration);
}

void sleep () {
   sleep_for (3s);
}

auto thread_id () {
   return std::this_thread::get_id ();
}

// -------------------------------------------------------------------------------------------------

template <typename T> consteval T make_value () {
   using U = std::remove_cvref_t <T>;

   if constexpr (std::is_same_v <U, int>)
      return 42;

   else if constexpr (std::is_same_v <U, double>)
      return 3.14;

   else if constexpr (std::is_same_v <U, char const *>)
      return "hello world";
}

// -------------------------------------------------------------------------------------------------

template <typename T> void consume_value (std::future <T> & future) {
   try {
      std::cout << std::format ("thread {}: consuming ...\n", thread_id ());

      auto const value {future.get ()};

      std::cout << std::format ("thread {}: consumed value '{}'\n", thread_id (), value);

   } catch (std::exception const & x) {
      std::cerr << std::format ("thread {}: some exception '{}' was thrown\n", thread_id (), x.what ());

   } catch (...) {
      std::cerr << std::format ("thread {}: some unknown exception was thrown\n", thread_id ());
   }
}

template <typename T> void produce_value (std::promise <T> & promise) {
   try {
      auto const value {make_value <T> ()};

      std::cout << std::format ("thread {}: producing value '{}' ...\n", thread_id (), value);

      sleep ();

//    throw std::runtime_error {std::format ("oops in thread {}", thread_id ())};

      promise.set_value (value);

   } catch (...) {
      try {
         promise.set_exception (std::current_exception ());

      } catch (...) {   // set_exception may throw
      }
   }
}

// -------------------------------------------------------------------------------------------------

int main_c () {
   std::cout << std::format ("thread {}: main thread\n", thread_id ());

   std::promise <char const *> promise {};
   auto                        future  {promise.get_future ()};

   std::jthread consumer {[&future] {
      consume_value (future);
   }};

   sleep ();

   std::jthread producer {[&promise] {
      produce_value (promise);
   }};
}
