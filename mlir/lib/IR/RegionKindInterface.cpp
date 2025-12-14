//===- RegionKindInterface.cpp - Region Kind Interfaces ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the definitions of the region kind interfaces defined in
// `RegionKindInterface.td`.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/RegionKindInterface.h"
#include "mlir/Support/WalkResult.h"

#include "llvm/Support/DebugLog.h"

#define DEBUG_TYPE "region-kind-interface"

using namespace mlir;

#include "mlir/IR/RegionKindInterface.cpp.inc"

bool mlir::mayHaveSSADominance(Region &region) {
  auto regionKindOp = dyn_cast<RegionKindInterface>(region.getParentOp());
  if (!regionKindOp)
    return true;
  return regionKindOp.hasSSADominance(region.getRegionNumber());
}

bool mlir::mayBeGraphRegion(Region &region) {
  if (!region.getParentOp()->isRegistered())
    return true;
  auto regionKindOp = dyn_cast<RegionKindInterface>(region.getParentOp());
  if (!regionKindOp)
    return false;
  return !regionKindOp.hasSSADominance(region.getRegionNumber());
}

namespace {
// Iterator on all reachable operations in the region.
// Also keep track if we visited the nested regions of the current op
// already to drive the traversal.
struct NestedOpIterator {
  NestedOpIterator(Region *region, int nestedLevel)
      : region(region), nestedLevel(nestedLevel) {
    regionIt = region->begin();
    blockIt = regionIt->end();
    if (regionIt != region->end())
      blockIt = regionIt->begin();
  }
  // Advance the iterator to the next reachable operation.
  void advance() {
    assert(regionIt != region->end());
    if (blockIt == regionIt->end()) {
      ++regionIt;
      if (regionIt != region->end())
        blockIt = regionIt->begin();
      return;
    }
    ++blockIt;
    if (blockIt != regionIt->end()) {
      LDBG() << this << " - Incrementing block iterator, next op: "
             << OpWithFlags(&*blockIt, OpPrintingFlags().skipRegions());
    }
  }

  // The region we're iterating over.
  Region *region;
  // The Block currently being iterated over.
  Region::iterator regionIt;
  // The Operation currently being iterated over.
  Block::iterator blockIt;
  // The nested level of the current region relative to the starting region.
  int nestedLevel = 0;
};
} // namespace

/// Recursive walk that calls the callback only for terminator operation which
/// are breaking control flow.
static void walk(Operation *rootOp,
                 function_ref<WalkResult(Operation *, int)> callback) {
  // Worklist of regions to visit to drive the traversal.
  SmallVector<NestedOpIterator> worklist;

  // Perform a traversal of the regions, visiting each
  // reachable operation.
  for (Region &region : rootOp->getRegions()) {
    if (region.empty())
      continue;
    worklist.push_back({&region, 1});
  }
  while (!worklist.empty()) {
    NestedOpIterator &it = worklist.back();
    if (it.regionIt == it.region->end()) {
      // We're done with this region.
      worklist.pop_back();
      continue;
    }
    if (it.blockIt == it.regionIt->end()) {
      // We're done with this block.
      it.advance();
      continue;
    }
    Operation *op = &*it.blockIt;

    // Only call the callback if we're at the end of the block.
    if (std::next(it.blockIt) == it.regionIt->end() &&
        callback(op, it.nestedLevel).wasInterrupted())
      return;

    // Advance before pushing nested regions to avoid reference invalidation.
    int currentNestedLevel = it.nestedLevel;
    it.advance();

    // Recursively visit the nested regions.
    for (Region &nestedRegion : op->getRegions()) {
      if (nestedRegion.empty())
        continue;
      worklist.push_back({&nestedRegion, currentNestedLevel + 1});
    }
  }
}

bool mlir::hasNestedPredecessors(Operation *op) {
  bool found = false;
  walk(op, [&](Operation *visitedOp, int nestedLevel) {
    if (nestedLevel > 1 &&
        nestedLevel ==
            static_cast<int>(visitedOp->getNumBreakingControlRegions()))
      found = true;
    return found ? WalkResult::interrupt() : WalkResult::advance();
  });
  return found;
}

bool mlir::hasBreakingControlFlowOps(Operation *op) {
  bool found = false;
  walk(op, [&](Operation *visitedOp, int nestedLevel) {
    if (nestedLevel >
        static_cast<int>(visitedOp->getNumBreakingControlRegions()))
      found = true;
    return found ? WalkResult::interrupt() : WalkResult::advance();
  });
  return found;
}

void mlir::detail::visitNestedBreakingControlFlowOpsImpl(
    Operation *op,
    function_ref<WalkResult(Operation *, int nestedLevel)> callback) {
  ::walk(op, [&](Operation *visitedOp, int nestedLevel) {
    if (nestedLevel <=
        static_cast<int>(visitedOp->getNumBreakingControlRegions()))
      return callback(visitedOp, nestedLevel);
    return WalkResult::advance();
  });
}

void mlir::collectAllNestedPredecessors(
    Operation *op, SmallVector<Operation *> &predecessors) {
  visitNestedBreakingControlFlowOps(
      op, [&](Operation *visitedOp, int nestedLevel) {
        if (nestedLevel ==
            static_cast<int>(visitedOp->getNumBreakingControlRegions()))
          predecessors.push_back(visitedOp);
        return WalkResult::advance();
      });
}