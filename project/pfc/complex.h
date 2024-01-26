//       $Id: complex.h 48073 2024-01-11 08:11:16Z p20068 $
//      $URL: https://svn01.fh-hagenberg.at/bin/cepheiden/pfc/trunk/pfc/inc/pfc/complex.h $
// $Revision: 48073 $
//     $Date: 2024-01-11 09:11:16 +0100 (Do., 11 Jän 2024) $
//   $Author: p20068 $
//   Creator: Peter Kulczycki
//  Creation: November 7, 2020
// Copyright: (c) 2021 Peter Kulczycki (peter.kulczycki<AT>fh-hagenberg.at)
//   License: This document contains proprietary information belonging to
//            University of Applied Sciences Upper Austria, Campus Hagenberg.
//            It is distributed under the Boost Software License (see
//            https://www.boost.org/users/license.html).

#pragma once

#undef  PFC_COMPLEX_VERSION
#define PFC_COMPLEX_VERSION "1.10.0"

#undef PFC_COMPLEX_HAVE_CUCOMPLEX
#undef PFC_COMPLEX_HAVE_STDCOMPLEX

#if __has_include (<cucomplex.h>)
   #include <cucomplex.h>

   #define PFC_COMPLEX_HAVE_CUCOMPLEX
#endif

#if !defined __CUDACC__
   #include <complex>
   #include <iostream>

   #define PFC_COMPLEX_HAVE_STDCOMPLEX
#endif

#include <type_traits>

#undef PFC_GPU_ENABLED

#if defined __CUDACC__
   #define PFC_GPU_ENABLED __host__ __device__
#else
   #define PFC_GPU_ENABLED
#endif

namespace pfc {

template <typename T = double> class complex final {
   static_assert (
      std::is_integral_v <T> || std::is_floating_point_v <T>, "PFC: T must be an integral or a floating-point type"
   );

   public:
      using imag_t  = T;
      using real_t  = T;
      using value_t = T;

      PFC_GPU_ENABLED static char const * version () {
         return PFC_COMPLEX_VERSION;
      }

      constexpr complex () = default;

      template <typename U> PFC_GPU_ENABLED constexpr complex (U const r) : real {static_cast <value_t> (r)} {
      }

      template <typename U> PFC_GPU_ENABLED constexpr complex (U const r, U const i)
         : real {static_cast <value_t> (r)}
         , imag {static_cast <value_t> (i)} {
      }

      template <typename U> PFC_GPU_ENABLED constexpr complex (complex <U> const & c)
         : real {static_cast <value_t> (c.real)}
         , imag {static_cast <value_t> (c.imag)} {
      }

      #if defined PFC_COMPLEX_HAVE_CUCOMPLEX
         PFC_GPU_ENABLED constexpr complex (cuDoubleComplex const & c)
            : real {static_cast <value_t> (cuCreal (c))}
            , imag {static_cast <value_t> (cuCimag (c))} {
         }

         PFC_GPU_ENABLED constexpr complex (cuFloatComplex const & c)
            : real {static_cast <value_t> (cuCrealf (c))}
            , imag {static_cast <value_t> (cuCimagf (c))} {
         }
      #endif

      #if defined PFC_COMPLEX_HAVE_STDCOMPLEX
         template <typename U> PFC_GPU_ENABLED constexpr complex (std::complex <U> const & c)
            : real {static_cast <value_t> (c.real ())}
            , imag {static_cast <value_t> (c.imag ())} {
         }
      #endif

      constexpr complex (complex const &) = default;
      constexpr complex (complex &&) = default;

      constexpr complex & operator = (complex const &) = default;
      constexpr complex & operator = (complex &&) = default;

      #if defined PFC_COMPLEX_HAVE_CUCOMPLEX
         PFC_GPU_ENABLED constexpr operator cuDoubleComplex () const {
            return {static_cast <double> (real), static_cast <double> (imag)};
         }

         PFC_GPU_ENABLED constexpr operator cuFloatComplex () const {
            return {static_cast <float> (real), static_cast <float> (imag)};
         }
      #endif

      #if defined PFC_COMPLEX_HAVE_STDCOMPLEX
         PFC_GPU_ENABLED constexpr operator std::complex <value_t> () const {
            return {real, imag};
         }
      #endif

      friend PFC_GPU_ENABLED constexpr auto operator + (complex lhs, complex const & rhs) {
         return lhs += rhs;
      }

      friend PFC_GPU_ENABLED constexpr auto operator - (complex const & rhs) {
         return complex {} -= rhs;
      }

      friend PFC_GPU_ENABLED constexpr auto operator - (complex lhs, complex const & rhs) {
         return lhs -= rhs;
      }

      friend PFC_GPU_ENABLED constexpr auto operator * (complex lhs, complex const & rhs) {
         return lhs *= rhs;
      }

      friend PFC_GPU_ENABLED constexpr auto operator / (complex lhs, value_t const rhs) {
         return lhs /= rhs;
      }

      #if !defined __CUDACC__
         friend auto & operator << (std::ostream & lhs, complex const & rhs) {
            return lhs << '{' << rhs.real << ',' << rhs.imag << '}';
         }
      #endif

      PFC_GPU_ENABLED constexpr complex & operator += (complex const & rhs) {
         real += rhs.real; imag += rhs.imag; return *this;
      }

      PFC_GPU_ENABLED constexpr complex & operator -= (complex const & rhs) {
         real -= rhs.real; imag -= rhs.imag; return *this;
      }

      PFC_GPU_ENABLED constexpr complex & operator *= (complex const & rhs) {
         auto const r {real * rhs.real - imag * rhs.imag};
         auto const i {real * rhs.imag + imag * rhs.real};

         real = r; imag = i; return *this;
      }

      PFC_GPU_ENABLED constexpr complex & operator /= (value_t const rhs) {
         real /= rhs; imag /= rhs; return *this;
      }

      PFC_GPU_ENABLED constexpr value_t norm () const {
         return real * real + imag * imag;
      }

      PFC_GPU_ENABLED constexpr complex & square () {
         auto const r {real * real - imag * imag};

         imag *= real * 2;
         real  = r;

         return *this;
      }

      value_t real {};
      value_t imag {};
};

#if defined PFC_COMPLEX_HAVE_CUCOMPLEX

PFC_GPU_ENABLED constexpr cuDoubleComplex to_cuDoubleComplex (complex <double> const & x) {
   return x;
}

PFC_GPU_ENABLED constexpr cuFloatComplex to_cuFloatComplex (complex <float> const & x) {
   return x;
}

#endif   // PFC_COMPLEX_HAVE_CUCOMPLEX

#if defined PFC_COMPLEX_HAVE_STDCOMPLEX

template <typename T> PFC_GPU_ENABLED constexpr std::complex <T> to_std_complex (complex <T> const & x) {
   return x;
}

#endif   // PFC_COMPLEX_HAVE_STDCOMPLEX

template <typename T> PFC_GPU_ENABLED constexpr auto norm (complex <T> const & x) {
   return x.norm ();
}

template <typename T> PFC_GPU_ENABLED constexpr auto & square (complex <T> & x) {
   return x.square ();
}

namespace literals {

PFC_GPU_ENABLED inline constexpr auto operator "" _imag_f (long double const literal) {
   return pfc::complex <float> {0.0f, static_cast <float> (literal)};
}

PFC_GPU_ENABLED inline constexpr auto operator "" _imag (unsigned long long const literal) {
   return pfc::complex <double> {0.0, static_cast <double> (literal)};
}

PFC_GPU_ENABLED inline constexpr auto operator "" _imag (long double const literal) {
   return pfc::complex <double> {0.0, static_cast <double> (literal)};
}

PFC_GPU_ENABLED inline constexpr auto operator "" _imag_l (long double const literal) {
   return pfc::complex <long double> {0.0l, literal};
}

PFC_GPU_ENABLED inline constexpr auto operator "" _real_f (long double const literal) {
   return pfc::complex <float> {static_cast <float> (literal)};
}

PFC_GPU_ENABLED inline constexpr auto operator "" _real (unsigned long long const literal) {
   return pfc::complex <double> {static_cast <double> (literal)};
}

PFC_GPU_ENABLED inline constexpr auto operator "" _real (long double const literal) {
   return pfc::complex <double> {static_cast <double> (literal)};
}

PFC_GPU_ENABLED inline constexpr auto operator "" _real_l (long double const literal) {
   return pfc::complex <long double> {literal};
}

} }   // namespace pfc::literals

#undef PFC_GPU_ENABLED
