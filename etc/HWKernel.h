//       $Id: HWKernel.h 45253 2021-12-23 10:48:53Z p20068 $
//      $URL: https://svn01.fh-hagenberg.at/bin/cepheiden/Inhalt/HPC/GPU/HW/src/HWKernel.h $
// $Revision: 45253 $
//     $Date: 2021-12-23 11:48:53 +0100 (Thu, 23 Dec 2021) $
//   $Author: p20068 $
//   Creator: Peter Kulczycki
//  Creation: October 28, 2020
// Copyright: (c) 2021 Peter Kulczycki (peter.kulczycki<AT>fh-hagenberg.at)
//   License: This document contains proprietary information belonging to
//            University of Applied Sciences Upper Austria, Campus Hagenberg.
//            It is distributed under the Boost Software License (see
//            https://www.boost.org/users/license.html).

#pragma once

#if __has_include (<pfc/compiler.h>)
   #include <pfc/compiler.h>
#endif

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cstddef>

cudaError_t call_copy_string_kernel (dim3 big, dim3 tib, char * dp_dst, char const * dp_src, unsigned size);
