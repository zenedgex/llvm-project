// RUN: not mlir-opt %s -split-input-file 2>&1 | FileCheck %s

func.func @crash_repro() {
  %0 = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
  %1 = amx.tile_zero : !amx.tile<16x16xi32>
  
  // CHECK: Cannot build binary Linalg operation: expects allComplex, allFloatingPoint, or allInteger
  %2 = linalg.batch_reduce_matmul ins(%0, %0 : tensor<1xi32>, tensor<1xi32>) outs(%1 : !amx.tile<16x16xi32>) -> !amx.tile<16x16xi32>
  return
}