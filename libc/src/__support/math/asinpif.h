//===-- Implementation header for asinpif -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_MATH_ASINPIF_H
#define LLVM_LIBC_SRC___SUPPORT_MATH_ASINPIF_H

#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/FPUtil/PolyEval.h"
#include "src/__support/FPUtil/cast.h"
#include "src/__support/FPUtil/multiply_add.h"
#include "src/__support/FPUtil/sqrt.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/optimization.h"
#include "src/__support/macros/properties/types.h"

#include "hdr/errno_macros.h"
#include "hdr/fenv_macros.h"
#include "src/__support/FPUtil/FEnvImpl.h"
#include "src/__support/FPUtil/except_value_utils.h"

namespace LIBC_NAMESPACE_DECL {
namespace math {

LIBC_INLINE constexpr float asinpif(float x) {
  using FPBits = fputil::FPBits<float>;

  FPBits xbits(x);
  bool is_neg = xbits.is_neg();
  double x_abs = fputil::cast<double>(xbits.abs().get_val());

  auto signed_result = [is_neg](auto r) -> auto { return is_neg ? -r : r; };

  if (LIBC_UNLIKELY(x_abs > 1.0)) {
    if (xbits.is_nan()) {
      if (xbits.is_signaling_nan()) {
        fputil::raise_except_if_required(FE_INVALID);
        return FPBits::quiet_nan().get_val();
      }
      return x;
    }

    fputil::raise_except_if_required(FE_INVALID);
    fputil::set_errno_if_required(EDOM);
    return FPBits::quiet_nan().get_val();
  }

  // the coefficients for the polynomial approximation of asin(x)/pi in the
  // range [0, 0.5] extracted using Sollya
  //
  // Sollya code:
  // prec = 200;
  // display = hexadecimal;
  //
  // g = asin(sqrt(x)) / (pi * sqrt(x));
  // P = fpminimax(g, [|0,1,2,3,4,5,6,7,8,9,10,11,12,13|], [|D...|], [1e-30;
  // 0.25]);
  //
  // for i from 0 to degree(P) do {
  //   if (i < degree(P)) then
  //     print(coeff(P, i))
  //   else
  //     print(coeff(P, i));
  // };
  //
  // OUTPUT:
  // 0x1.45f306dc9c883p-2
  // 0x1.b2995e7b7b0ccp-5
  // 0x1.8723a1d5e3397p-6
  // 0x1.d1a452c4985d2p-7
  // 0x1.3ce5309393d12p-7
  // 0x1.d2b226cde1b71p-8
  // 0x1.6a0d7274a2995p-8
  // 0x1.22beb6d533d2fp-8
  // 0x1.e905390792fc8p-9
  // 0x1.7c84982aec05ap-9
  // 0x1.8d9c5583cbaffp-9
  // 0x1.5feeeacc6c4a6p-9
  // -0x1.1237966112762p-10
  // 0x1.cd80e8b6681dp-8
  constexpr double ASINPI_POLY_COEFFS[] = {
      0x1.45f306dc9c883p-2,   // c1
      0x1.b2995e7b7b0ccp-5,   // c3
      0x1.8723a1d5e3397p-6,   // c5
      0x1.d1a452c4985d2p-7,   // c7
      0x1.3ce5309393d12p-7,   // c9
      0x1.d2b226cde1b71p-8,   // c11
      0x1.6a0d7274a2995p-8,   // c13
      0x1.22beb6d533d2fp-8,   // c15
      0x1.e905390792fc8p-9,   // c17
      0x1.7c84982aec05ap-9,   // c19
      0x1.8d9c5583cbaffp-9,   // c21
      0x1.5feeeacc6c4a6p-9,   // c23
      -0x1.1237966112762p-10, // c25
      0x1.cd80e8b6681dp-8     // c27
  };

  // polynomial evaluation using horner's method
  // work only for |x| in [0, 0.5]
  // Returns v * P(v2) where P(v2) = c0 + c1*v2 + c2*v2^2 + ...
  auto asinpi_polyeval = [&](double v, double v2) -> double {
    return v * fputil::polyeval(v2, ASINPI_POLY_COEFFS[0],
                                ASINPI_POLY_COEFFS[1], ASINPI_POLY_COEFFS[2],
                                ASINPI_POLY_COEFFS[3], ASINPI_POLY_COEFFS[4],
                                ASINPI_POLY_COEFFS[5], ASINPI_POLY_COEFFS[6],
                                ASINPI_POLY_COEFFS[7], ASINPI_POLY_COEFFS[8],
                                ASINPI_POLY_COEFFS[9], ASINPI_POLY_COEFFS[10],
                                ASINPI_POLY_COEFFS[11], ASINPI_POLY_COEFFS[12],
                                ASINPI_POLY_COEFFS[13]);
  };

  // Returns P(v2) - c0 = c1*v2 + c2*v2^2 + ...
  // This is the "tail" of the polynomial, used to avoid cancellation
  // in the range reduction path.
  auto asinpi_polyeval_tail = [&](double v2) -> double {
    return v2 * fputil::polyeval(v2, ASINPI_POLY_COEFFS[1],
                                 ASINPI_POLY_COEFFS[2], ASINPI_POLY_COEFFS[3],
                                 ASINPI_POLY_COEFFS[4], ASINPI_POLY_COEFFS[5],
                                 ASINPI_POLY_COEFFS[6], ASINPI_POLY_COEFFS[7],
                                 ASINPI_POLY_COEFFS[8], ASINPI_POLY_COEFFS[9],
                                 ASINPI_POLY_COEFFS[10], ASINPI_POLY_COEFFS[11],
                                 ASINPI_POLY_COEFFS[12],
                                 ASINPI_POLY_COEFFS[13]);
  };

  // if |x| <= 0.5:
  if (LIBC_UNLIKELY(x_abs <= 0.5)) {
    double x_d = fputil::cast<double>(x);
    double result = asinpi_polyeval(x_d, x_d * x_d);
    return fputil::cast<float>(result);
  }

  // If |x| > 0.5, we need to use the range reduction method:
  //    y = asin(x) => x = sin(y)
  //      because: sin(a) = cos(pi/2 - a)
  //      therefore:
  //    x = cos(pi/2 - y)
  //      let z = pi/2 - y,
  //    x = cos(z)
  //      because: cos(2a) = 1 - 2 * sin^2(a), z = 2a, a = z/2
  //      therefore:
  //    cos(z) = 1 - 2 * sin^2(z/2)
  //    sin(z/2) = sqrt((1 - cos(z))/2)
  //    sin(z/2) = sqrt((1 - x)/2)
  //      let u = (1 - x)/2
  //      then:
  //    sin(z/2) = sqrt(u)
  //    z/2 = asin(sqrt(u))
  //    z = 2 * asin(sqrt(u))
  //    pi/2 - y = 2 * asin(sqrt(u))
  //    y = pi/2 - 2 * asin(sqrt(u))
  //    y/pi = 1/2 - 2 * asin(sqrt(u))/pi
  //
  // Finally, we can write:
  //   asinpi(x) = 1/2 - 2 * asinpi(sqrt(u))
  //     where u = (1 - x) /2
  //             = 0.5 - 0.5 * x
  //             = multiply_add(-0.5, x, 0.5)
  //
  // asinpi(x) = 1/2 - 2 * sqrt(u) * P(u)
  //           = 1/2 - 2 * sqrt(u) * (c0 + (P(u) - c0))
  //           = 1/2 - 2 * sqrt(u) * c0 - 2 * sqrt(u) * (P(u) - c0)
  //           = (1/2 - 2 * sqrt(u) / pi) - 2 * sqrt(u) * tail(u)
  //
  // We use the high-precision constant 1/pi split into two parts:
  //   ONE_OVER_PI_HI + ONE_OVER_PI_LO ≈ 1/π
  // so that:
  //   1/2 - 2*sqrt(u)*c0 = 1/2 - 2*sqrt(u)*ONE_OVER_PI_HI
  //                         - 2*sqrt(u)*ONE_OVER_PI_LO
  // The first two terms are computed exactly via multiply_add.

  constexpr double ONE_OVER_PI_HI = ASINPI_POLY_COEFFS[0];
  constexpr double ONE_OVER_PI_LO = -0x1.6b01ec5417056p-56;

  double u = fputil::multiply_add(-0.5, x_abs, 0.5);
  double sqrt_u = fputil::sqrt<double>(u);

  // compute the tail: P(u) - c0
  double tail = asinpi_polyeval_tail(u);

  // compute: 0.5 - 2*sqrt(u)/pi - 2*sqrt(u)*tail
  //        = 0.5 - 2*sqrt(u)*(1/pi + tail)
  double neg2_sqrt_u = -2.0 * sqrt_u;
  // 0.5 + neg2_sqrt_u * ONE_OVER_PI_HI
  double result_hi = fputil::multiply_add(neg2_sqrt_u, ONE_OVER_PI_HI, 0.5);
  // Add the low part of 1/pi and the tail
  double result = result_hi + neg2_sqrt_u * (ONE_OVER_PI_LO + tail);

  return fputil::cast<float>(signed_result(result));
}

} // namespace math
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_MATH_ASINPIF_H
