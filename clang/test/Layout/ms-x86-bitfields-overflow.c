// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fms-extensions -fdump-record-layouts -fsyntax-only %s 2>/dev/null \
// RUN:            | FileCheck %s

// Test that large _BitInt fields do not overflow the internal storage unit tracker (previously unsigned char).
// If the bug exists, this struct splits into two 256-bit units.
// If fixed, f2 and f3 are packed into a single 256-bit unit.

#pragma ms_struct on

struct __attribute__((packed, aligned(1))) A {
  _BitInt(250) f2 : 2;
  _BitInt(250) f3 : 2;
};

// CHECK-LABEL:   0 | struct A{{$}}
// CHECK-NEXT:0:0-1 |   _BitInt(250) f2
// CHECK-NEXT:0:2-3 |   _BitInt(250) f3
// CHECK-NEXT:      | [sizeof=32, align=32]

// Force the compiler to layout the struct
int x[sizeof(struct A)];
