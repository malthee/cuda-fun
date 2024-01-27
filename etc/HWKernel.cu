//       $Id: HWKernel.cu 45253 2021-12-23 10:48:53Z p20068 $
//      $URL: https://svn01.fh-hagenberg.at/bin/cepheiden/Inhalt/HPC/GPU/HW/src/HWKernel.cu $
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

#include "./HWKernel.h"

template <typename T> __global__ void copy_string_kernel (T * const dp_dst, T const * const dp_src, unsigned const size) {
   auto const t {blockIdx.x * blockDim.x + threadIdx.x};

   if (t < size)
      dp_dst[t] = dp_src[t];
}

cudaError_t call_copy_string_kernel (dim3 const big, dim3 const tib, char * const dp_dst, char const * const dp_src, unsigned const size) {
   copy_string_kernel <<<big, tib>>> (dp_dst, dp_src, size); return cudaGetLastError ();
}
