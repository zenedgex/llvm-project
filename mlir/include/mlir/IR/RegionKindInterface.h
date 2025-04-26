//===- RegionKindInterface.h - Region Kind Interfaces -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the definitions of the infer op interfaces defined in
// `RegionKindInterface.td`.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_REGIONKINDINTERFACE_H_
#define MLIR_IR_REGIONKINDINTERFACE_H_

#include "mlir/IR/OpDefinition.h"

namespace mlir {

/// The kinds of regions contained in an operation. SSACFG regions
/// require the SSA-Dominance property to hold. Graph regions do not
/// require SSA-Dominance. If a registered operation does not implement
/// RegionKindInterface, then any regions it contains are assumed to be
/// SSACFG regions.
enum class RegionKind {
  SSACFG,
  Graph,
};

namespace OpTrait {
/// A trait that specifies that an operation only defines graph regions.
template <typename ConcreteType>
class HasOnlyGraphRegion : public TraitBase<ConcreteType, HasOnlyGraphRegion> {
public:
  static RegionKind getRegionKind(unsigned index) { return RegionKind::Graph; }
  static bool hasSSADominance(unsigned index) { return false; }
};

/// Indicates that this operation may break control flow, by propagating the
/// control flow break from a nested region.
template <typename ConcreteType>
class PropagateControlFlowBreak
    : public TraitBase<ConcreteType, PropagateControlFlowBreak> {
public:
  static LogicalResult verifyTrait(Operation *op) {
    // Verify the operation has regions and can handle breaking control flow
    if (op->getNumRegions() == 0)
      return op->emitOpError(
          "operation with PropagateControlFlowBreak trait must have regions");
    return success();
  }
};

/// Indicates that this operation terminates the region execution, transfering
/// control flow to a parent operation, potentially not the directly enclosing
/// one.
template <typename ConcreteType>
class RegionTerminator : public TraitBase<ConcreteType, RegionTerminator> {
public:
  static LogicalResult verifyTrait(Operation *op) {
    if (op->getNumBreakingControlRegions() == 0)
      return op->emitOpError("operation with region terminator trait must have "
                             "breaking control regions > 0");
    if (!op->hasTrait<OpTrait::IsTerminator>())
      return op->emitOpError(
          "operation with region terminator trait must be a terminator");
    return success();
  }
};

} // namespace OpTrait

/// Return "true" if the given region may have SSA dominance. This function also
/// returns "true" in case the owner op is an unregistered op or an op that does
/// not implement the RegionKindInterface.
bool mayHaveSSADominance(Region &region);

/// Return "true" if the given region may be a graph region without SSA
/// dominance. This function returns "true" in case the owner op is an
/// unregistered op. It returns "false" if it is a registered op that does not
/// implement the RegionKindInterface.
bool mayBeGraphRegion(Region &region);

bool hasNestedPredecessors(Operation *op);

/// Return "true" if the given operation may break control flow and contains
/// nested operations that have a successor above this operation.
bool hasBreakingControlFlowOps(Operation *op);

void collectAllNestedPredecessors(Operation *op,
                                  SmallVector<Operation *> &predecessors);

namespace detail {
void visitNestedBreakingControlFlowOpsImpl(
    Operation *op,
    function_ref<WalkResult(Operation *, int nestedLevel)> callback);
}

template <typename CallbackT>
std::enable_if_t<
    std::is_same_v<decltype(std::declval<CallbackT>()(
                       std::declval<Operation *>(), std::declval<int>())),
                   WalkResult>>
visitNestedBreakingControlFlowOps(Operation *op, CallbackT &&callback) {
  detail::visitNestedBreakingControlFlowOpsImpl(op, callback);
}

template <typename CallbackT>
std::enable_if_t<
    std::is_same_v<decltype(std::declval<CallbackT>()(
                       std::declval<Operation *>(), std::declval<int>())),
                   void>>
visitNestedBreakingControlFlowOps(Operation *op, CallbackT &&callback) {
  detail::visitNestedBreakingControlFlowOpsImpl(
      op, [&](Operation *visitedOp, int nestedLevel) {
        callback(visitedOp, nestedLevel);
        return WalkResult::advance();
      });
}

template <typename CallbackT>
std::enable_if_t<
    std::is_same_v<decltype(std::declval<CallbackT>()(
                       std::declval<Operation *>(), std::declval<int>())),
                   WalkResult>>
visitNestedBreakingControlFlowOps(Region &region, CallbackT &&callback) {
  for (Operation &op : region.getOps())
    detail::visitNestedBreakingControlFlowOpsImpl(&op, callback);
}
template <typename CallbackT>
std::enable_if_t<
    std::is_same_v<decltype(std::declval<CallbackT>()(
                       std::declval<Operation *>(), std::declval<int>())),
                   void>>
visitNestedBreakingControlFlowOps(Region &region, CallbackT &&callback) {
  for (Operation &op : region.getOps())
    detail::visitNestedBreakingControlFlowOpsImpl(
        &op, [&](Operation *visitedOp, int nestedLevel) {
          callback(visitedOp, nestedLevel);
          return WalkResult::advance();
        });
}

} // namespace mlir

#include "mlir/IR/RegionKindInterface.h.inc"

namespace mlir {

// Return true if the given region may break control flow.
inline bool hasBreakingControlFlow(Region *region) {
  return region->getParentOp()
             ->hasTrait<OpTrait::PropagateControlFlowBreak>() ||
         isa<HasBreakingControlFlowOpInterface>(region->getParentOp());
}

} // namespace mlir

#endif // MLIR_IR_REGIONKINDINTERFACE_H_
