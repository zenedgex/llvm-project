//===- SROA.h - Scalar Replacement Of Aggregates ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This file provides the interface for LLVM's Scalar Replacement of
/// Aggregates pass. This pass provides both aggregate splitting and the
/// primary SSA formation used in the compiler.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_SCALAR_SROA_H
#define LLVM_TRANSFORMS_SCALAR_SROA_H

#include "llvm/IR/PassManager.h"
#include "llvm/Support/Compiler.h"
#include <optional>

namespace llvm {

class Function;

/// Options for the SROA pass pipeline configuration.
struct SROAPassOptions {
  /// Whether to preserve the CFG (no modifications allowed).
  /// Default is false (modify-cfg) to match the original SROA behavior.
  bool PreserveCFG = false;
  /// Maximum size in bytes of a homogeneous struct to convert to a vector.
  /// If nullopt, defaults to 16 bytes.
  std::optional<unsigned> MaxStructToVectorSize = std::nullopt;
};

class SROAPass : public PassInfoMixin<SROAPass> {
  SROAPassOptions Options;

public:
  /// Construct SROA pass with the given options.
  SROAPass(SROAPassOptions Options = {}) : Options(Options) {}

  /// Run the pass over the function.
  LLVM_ABI PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);

  LLVM_ABI void
  printPipeline(raw_ostream &OS,
                function_ref<StringRef(StringRef)> MapClassName2PassName);
};

} // end namespace llvm

#endif // LLVM_TRANSFORMS_SCALAR_SROA_H
