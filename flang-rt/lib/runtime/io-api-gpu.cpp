//===-- lib/runtime/io-api-gpu.cpp ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Implements the subset of the I/O statement API needed for basic
// list-directed output (PRINT *) of intrinsic types for the GPU.
//
// The RPC interface forwards each runtime call from the client to the server
// using a shared buffer. These calls are buffered on the server, so only the
// return value from 'BeginExternalListOutput' and 'EndIoStatement' are
// meaningful.

#include "io-api-gpu.h"
#include "flang/Runtime/io-api.h"

#include <shared/rpc.h>
#include <shared/rpc_dispatch.h>

namespace Fortran::runtime::io {
// A weak reference to the RPC client used to submit calls to the server.
[[gnu::weak, gnu::visibility("protected")]] rpc::Client client asm(
    "__llvm_rpc_client");

RT_EXT_API_GROUP_BEGIN

Cookie IONAME(BeginExternalListOutput)(
    ExternalUnit unitNumber, const char *sourceFile, int sourceLine) {
  return rpc::dispatch<BeginExternalListOutput_Opcode>(client,
      IONAME(BeginExternalListOutput), unitNumber, sourceFile, sourceLine);
}

enum Iostat IONAME(EndIoStatement)(Cookie cookie) {
  return rpc::dispatch<EndIoStatement_Opcode>(
      client, IONAME(EndIoStatement), cookie);
}

bool IONAME(OutputInteger8)(Cookie cookie, std::int8_t n) {
  return rpc::dispatch<OutputInteger8_Opcode>(
      client, IONAME(OutputInteger8), cookie, n);
}

bool IONAME(OutputInteger16)(Cookie cookie, std::int16_t n) {
  return rpc::dispatch<OutputInteger16_Opcode>(
      client, IONAME(OutputInteger16), cookie, n);
}

bool IONAME(OutputInteger32)(Cookie cookie, std::int32_t n) {
  return rpc::dispatch<OutputInteger32_Opcode>(
      client, IONAME(OutputInteger32), cookie, n);
}

bool IONAME(OutputInteger64)(Cookie cookie, std::int64_t n) {
  return rpc::dispatch<OutputInteger64_Opcode>(
      client, IONAME(OutputInteger64), cookie, n);
}

#ifdef __SIZEOF_INT128__
bool IONAME(OutputInteger128)(Cookie cookie, common::int128_t n) {
  return rpc::dispatch<OutputInteger128_Opcode>(
      client, IONAME(OutputInteger128), cookie, n);
}
#endif

bool IONAME(OutputReal32)(Cookie cookie, float x) {
  return rpc::dispatch<OutputReal32_Opcode>(
      client, IONAME(OutputReal32), cookie, x);
}

bool IONAME(OutputReal64)(Cookie cookie, double x) {
  return rpc::dispatch<OutputReal64_Opcode>(
      client, IONAME(OutputReal64), cookie, x);
}

bool IONAME(OutputComplex32)(Cookie cookie, float re, float im) {
  return rpc::dispatch<OutputComplex32_Opcode>(
      client, IONAME(OutputComplex32), cookie, re, im);
}

bool IONAME(OutputComplex64)(Cookie cookie, double re, double im) {
  return rpc::dispatch<OutputComplex64_Opcode>(
      client, IONAME(OutputComplex64), cookie, re, im);
}

bool IONAME(OutputAscii)(Cookie cookie, const char *x, std::size_t length) {
  return rpc::dispatch<OutputAscii_Opcode>(
      client, IONAME(OutputAscii), cookie, x, length);
}

bool IONAME(OutputLogical)(Cookie cookie, bool truth) {
  return rpc::dispatch<OutputLogical_Opcode>(
      client, IONAME(OutputLogical), cookie, truth);
}

RT_EXT_API_GROUP_END
} // namespace Fortran::runtime::io
