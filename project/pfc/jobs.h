//       $Id: jobs.h 46519 2022-12-04 12:29:04Z p20068 $
//      $URL: https://svn01.fh-hagenberg.at/bin/cepheiden/pfc/trunk/pfc/inc/pfc/jobs.h $
// $Revision: 46519 $
//     $Date: 2022-12-04 13:29:04 +0100 (So., 04 Dez 2022) $
//   $Author: p20068 $
//   Creator: Peter Kulczycki
//  Creation: September 29, 2020
// Copyright: (c) 2021 Peter Kulczycki (peter.kulczycki<AT>fh-hagenberg.at)
//   License: This document contains proprietary information belonging to
//            University of Applied Sciences Upper Austria, Campus Hagenberg.
//            It is distributed under the Boost Software License (see
//            https://www.boost.org/users/license.html).

#pragma once

#undef  PFC_JOBS_VERSION
#define PFC_JOBS_VERSION "1.3.6"

#if defined __CUDACC__
   #pragma message ("PFC: Compiling 'jobs.h' with nvcc does not have any effect")
#else

#include "./complex.h"

#include <concepts>
#include <cstdio>
#include <format>
#include <fstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

// -------------------------------------------------------------------------------------------------

namespace pfc {

template <std::floating_point T = double> class jobs final {
   public:
      using real_t    = T;
      using complex_t = pfc::complex <real_t>;
      using pair_t    = std::pair <real_t, real_t>;

      using job_t = std::tuple <
         complex_t,   // lower left corner (in the complex plane)
         complex_t,   // upper right corner (in the complex plane)
         complex_t,   // center point (in the complex plane)
         pair_t       // complex image size (width and height)
      >;

      static char const * version () {
         return PFC_JOBS_VERSION;
      }

      static std::string make_filename (std::size_t const images) {
         return std::format ("jobs-{:03}.txt", images);
      }

      jobs () = default;

      explicit jobs (std::size_t const images) : jobs {make_filename (images)} {
      }

      explicit jobs (std::string const & filename) {
         read_jobs (std::ifstream {filename});
      }

      auto const & operator [] (std::size_t const i) const {
         return m_jobs[i];
      }

      auto const & aspect_ratio () const {
         return m_ratio;
      }

      auto const & at (std::size_t const i) const {
         return m_jobs[i];
      }

      auto const & get_center (std::size_t const i) const {
         return std::get <2> (m_jobs[i]);
      }

      auto const & get_lower_left (std::size_t const i) const {
         return std::get <0> (m_jobs[i]);
      }

      auto const & get_size (std::size_t const i) const {
         return std::get <3> (m_jobs[i]);
      }

      auto const & get_upper_right (std::size_t const i) const {
         return std::get <1> (m_jobs[i]);
      }

      auto begin () const {
         return std::cbegin (m_jobs);
      }

      auto end () const {
         return std::cend (m_jobs);
      }

      auto size () const {
         return std::size (m_jobs);
      }

   private:
      static complex_t read_complex (std::istream & in) {
         char c; real_t r, i; in >> c >> c >> r >> c >> i >> c; return {r, i};
      }

      static std::istream & turn_on_exceptions (std::istream & in) {
         in.exceptions (std::ios::badbit /*| std::ios::eofbit*/ | std::ios::failbit); return in;
      }

      void read_jobs (std::istream && in) {
         if (in) {
            auto const xflags {in.exceptions ()}; turn_on_exceptions (in);

            if (std::size_t images {}; (in >> images >> m_ratio) && (images > 0) && (m_ratio > 0)) {
               m_jobs.resize (images);

               for (auto & [ll, ur, cp, wh] : m_jobs) {
                  char        c {};
                  std::size_t i {};

                  in >> i;   // image # (i.e. zoom step #)

                  ll = read_complex (in);   // lower left corner (in the complex plane)
                  ur = read_complex (in);   // upper right corner (in the complex plane)
                  cp = read_complex (in);   // center point (in the complex plane)

                  in >> c >> c >> wh.first >> c >> wh.second >> c;   // image size (width and height)
               }
            }

            in.exceptions (xflags);
         }
      }

      std::vector <job_t> m_jobs  {};
      real_t              m_ratio {};
};

}   // namespace pfc

#endif   // __CUDACC__
