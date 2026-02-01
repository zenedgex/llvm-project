// REQUIRES: x86-registered-target
// REQUIRES: amdgpu-registered-target

// Tests that the module dependency scan and the module dependency graph
// generation are correct for offloading driver command-lines.

// BUG: It seems like there are too many duplicate modules are generated on the
// offloading side. The ScanCompilerInvocation might be affected by offloading
// flags.

// RUN: split-file %s %t

// RUN: %clang -std=c++23 -nostdlib -fmodules \
// RUN:   -fmodules-driver -Rmodules-driver \
// RUN:   -fmodule-map-file=%t/module.modulemap %t/main.cpp \
// RUN:   -fmodules-cache-path=%t/modules-cache \
// RUN:   -fopenmp=libomp -fopenmp-targets=amdgcn-amd-amdhsa -nogpulib -nogpuinc \
// RUN:   --target=x86_64-linux-gnu \
// RUN:   %t/A.cpp %t/A-B.cpp %t/A-C.cpp %t/B.cpp -### 2>&1 \
// RUN:   | sed 's:\\\\\?:/:g' \
// RUN:   | FileCheck -DPREFIX=%/t %s

// CHECK:       clang: remark: standard modules manifest file not found; import of standard library modules not supported [-Rmodules-driver]
// CHECK:       clang: remark: printing module dependency graph [-Rmodules-driver]
// CHECK-NEXT:  digraph "Module Dependency Graph" {
//
// CHECK:        "[[PREFIX]]/main.cpp-[[HOST_ARCH:.*]]" [ fillcolor=3, label="{ Filename: [[PREFIX]]/main.cpp | Triple: [[HOST_ARCH]] }"];
// CHECK-NEXT:   "[[PREFIX]]/main.cpp-[[OFFLOADING_ARCH:.*]]" [ fillcolor=3, label="{ Filename: [[PREFIX]]/main.cpp | Triple: [[OFFLOADING_ARCH]] }"];
// CHECK-NEXT:   "A-[[HOST_ARCH]]" [ fillcolor=2, label="{ Module type: Named module | Module name: A | Triple: [[HOST_ARCH]] }"];
// CHECK-NEXT:   "A-[[OFFLOADING_ARCH]]" [ fillcolor=2, label="{ Module type: Named module | Module name: A | Triple: [[OFFLOADING_ARCH]] }"];
// CHECK-NEXT:   "A:B-[[HOST_ARCH]]" [ fillcolor=2, label="{ Module type: Named module | Module name: A:B | Triple: [[HOST_ARCH]] }"];
// CHECK-NEXT:   "A:B-[[OFFLOADING_ARCH]]" [ fillcolor=2, label="{ Module type: Named module | Module name: A:B | Triple: [[OFFLOADING_ARCH]] }"];
// CHECK-NEXT:   "A:C-[[HOST_ARCH]]" [ fillcolor=2, label="{ Module type: Named module | Module name: A:C | Triple: [[HOST_ARCH]] }"];
// CHECK-NEXT:   "A:C-[[OFFLOADING_ARCH]]" [ fillcolor=2, label="{ Module type: Named module | Module name: A:C | Triple: [[OFFLOADING_ARCH]] }"];
// CHECK-NEXT:   "B-[[HOST_ARCH]]" [ fillcolor=2, label="{ Module type: Named module | Module name: B | Triple: [[HOST_ARCH]] }"];
// CHECK-NEXT:   "B-[[OFFLOADING_ARCH]]" [ fillcolor=2, label="{ Module type: Named module | Module name: B | Triple: [[OFFLOADING_ARCH]] }"];
// CHECK-NEXT:   "transitive1-[[TRANSITIVE1_HASH1:.*]]" [ fillcolor=1, label="{ Module type: Clang module | Module name: transitive1 | Hash: [[TRANSITIVE1_HASH1]] }"];
// CHECK-NEXT:   "transitive2-[[TRANSITIVE2_HASH1:.*]]" [ fillcolor=1, label="{ Module type: Clang module | Module name: transitive2 | Hash: [[TRANSITIVE2_HASH1]] }"];
// CHECK-NEXT:   "direct1-[[DIRECT1_HASH1:.*]]" [ fillcolor=1, label="{ Module type: Clang module | Module name: direct1 | Hash: [[DIRECT1_HASH1]] }"];
// CHECK-NEXT:   "direct2-[[DIRECT2_HASH1:.*]]" [ fillcolor=1, label="{ Module type: Clang module | Module name: direct2 | Hash: [[DIRECT2_HASH1]] }"];
// CHECK-NEXT:   "root-[[ROOT_HASH1:.*]]" [ fillcolor=1, label="{ Module type: Clang module | Module name: root | Hash: [[ROOT_HASH1]] }"];
// CHECK-NEXT:   "transitive1-[[TRANSITIVE1_HASH2:.*]]" [ fillcolor=1, label="{ Module type: Clang module | Module name: transitive1 | Hash: [[TRANSITIVE1_HASH2]] }"];
// CHECK-NEXT:   "transitive2-[[TRANSITIVE2_HASH2:.*]]" [ fillcolor=1, label="{ Module type: Clang module | Module name: transitive2 | Hash: [[TRANSITIVE2_HASH2]] }"];
// CHECK-NEXT:   "direct1-[[DIRECT1_HASH2:.*]]" [ fillcolor=1, label="{ Module type: Clang module | Module name: direct1 | Hash: [[DIRECT1_HASH2]] }"];
// CHECK-NEXT:   "direct2-[[DIRECT2_HASH2:.*]]" [ fillcolor=1, label="{ Module type: Clang module | Module name: direct2 | Hash: [[DIRECT2_HASH2]] }"];
// CHECK-NEXT:   "root-[[ROOT_HASH2:.*]]" [ fillcolor=1, label="{ Module type: Clang module | Module name: root | Hash: [[ROOT_HASH2]] }"];
// CHECK-NEXT:   "transitive1-[[TRANSITIVE1_HASH3:.*]]" [ fillcolor=1, label="{ Module type: Clang module | Module name: transitive1 | Hash: [[TRANSITIVE1_HASH3]] }"];
// CHECK-NEXT:   "transitive2-[[TRANSITIVE2_HASH3:.*]]" [ fillcolor=1, label="{ Module type: Clang module | Module name: transitive2 | Hash: [[TRANSITIVE2_HASH3]] }"];
// CHECK-NEXT:   "direct1-[[DIRECT1_HASH3:.*]]" [ fillcolor=1, label="{ Module type: Clang module | Module name: direct1 | Hash: [[DIRECT1_HASH3]] }"];
// CHECK-NEXT:   "transitive1-[[TRANSITIVE1_HASH4:.*]]" [ fillcolor=1, label="{ Module type: Clang module | Module name: transitive1 | Hash: [[TRANSITIVE1_HASH4]] }"];
// CHECK-NEXT:   "transitive2-[[TRANSITIVE2_HASH4:.*]]" [ fillcolor=1, label="{ Module type: Clang module | Module name: transitive2 | Hash: [[TRANSITIVE2_HASH4]] }"];
// CHECK-NEXT:   "direct1-[[DIRECT1_HASH4:.*]]" [ fillcolor=1, label="{ Module type: Clang module | Module name: direct1 | Hash: [[DIRECT1_HASH4]] }"];
// CHECK-NEXT:   "direct2-[[DIRECT2_HASH4:.*]]" [ fillcolor=1, label="{ Module type: Clang module | Module name: direct2 | Hash: [[DIRECT2_HASH4]] }"];
// CHECK-NEXT:   "root-[[ROOT_HASH3:.*]]" [ fillcolor=1, label="{ Module type: Clang module | Module name: root | Hash: [[ROOT_HASH3]] }"];
//
// CHECK:        "A-[[HOST_ARCH]]" -> "[[PREFIX]]/main.cpp-[[HOST_ARCH]]";
// CHECK-NEXT:   "A-[[HOST_ARCH]]" -> "B-[[HOST_ARCH]]";
// CHECK-NEXT:   "A-[[OFFLOADING_ARCH]]" -> "[[PREFIX]]/main.cpp-[[OFFLOADING_ARCH]]";
// CHECK-NEXT:   "A-[[OFFLOADING_ARCH]]" -> "B-[[OFFLOADING_ARCH]]";
// CHECK-NEXT:   "A:B-[[HOST_ARCH]]" -> "A-[[HOST_ARCH]]";
// CHECK-NEXT:   "A:B-[[OFFLOADING_ARCH]]" -> "A-[[OFFLOADING_ARCH]]";
// CHECK-NEXT:   "A:C-[[HOST_ARCH]]" -> "A-[[HOST_ARCH]]";
// CHECK-NEXT:   "A:C-[[OFFLOADING_ARCH]]" -> "A-[[OFFLOADING_ARCH]]";
// CHECK-NEXT:   "B-[[HOST_ARCH]]" -> "[[PREFIX]]/main.cpp-[[HOST_ARCH]]";
// CHECK-NEXT:   "B-[[OFFLOADING_ARCH]]" -> "[[PREFIX]]/main.cpp-[[OFFLOADING_ARCH]]";
// CHECK-NEXT:   "transitive1-[[TRANSITIVE1_HASH1]]" -> "direct1-[[DIRECT1_HASH1]]";
// CHECK-NEXT:   "transitive1-[[TRANSITIVE1_HASH1]]" -> "direct2-[[DIRECT2_HASH1]]";
// CHECK-NEXT:   "transitive2-[[TRANSITIVE2_HASH1]]" -> "direct1-[[DIRECT1_HASH1]]";
// CHECK-NEXT:   "direct1-[[DIRECT1_HASH1]]" -> "root-[[ROOT_HASH1]]";
// CHECK-NEXT:   "direct1-[[DIRECT1_HASH1]]" -> "A:B-[[HOST_ARCH]]";
// CHECK-NEXT:   "direct2-[[DIRECT2_HASH1]]" -> "root-[[ROOT_HASH1]]";
// CHECK-NEXT:   "root-[[ROOT_HASH1]]" -> "[[PREFIX]]/main.cpp-[[HOST_ARCH]]";
// CHECK-NEXT:   "root-[[ROOT_HASH1]]" -> "B-[[HOST_ARCH]]";
// CHECK-NEXT:   "transitive1-[[TRANSITIVE1_HASH2]]" -> "direct1-[[DIRECT1_HASH2]]";
// CHECK-NEXT:   "transitive1-[[TRANSITIVE1_HASH2]]" -> "direct2-[[DIRECT2_HASH2]]";
// CHECK-NEXT:   "transitive2-[[TRANSITIVE2_HASH2]]" -> "direct1-[[DIRECT1_HASH2]]";
// CHECK-NEXT:   "direct1-[[DIRECT1_HASH2]]" -> "root-[[ROOT_HASH2]]";
// CHECK-NEXT:   "direct2-[[DIRECT2_HASH2]]" -> "root-[[ROOT_HASH2]]";
// CHECK-NEXT:   "root-[[ROOT_HASH2]]" -> "[[PREFIX]]/main.cpp-[[OFFLOADING_ARCH]]";
// CHECK-NEXT:   "transitive1-[[TRANSITIVE1_HASH3]]" -> "direct1-[[DIRECT1_HASH3]]";
// CHECK-NEXT:   "transitive2-[[TRANSITIVE2_HASH3]]" -> "direct1-[[DIRECT1_HASH3]]";
// CHECK-NEXT:   "direct1-[[DIRECT1_HASH3]]" -> "A:B-[[OFFLOADING_ARCH]]";
// CHECK-NEXT:   "transitive1-[[TRANSITIVE1_HASH4]]" -> "direct1-[[DIRECT1_HASH4]]";
// CHECK-NEXT:   "transitive1-[[TRANSITIVE1_HASH4]]" -> "direct2-[[DIRECT2_HASH4]]";
// CHECK-NEXT:   "transitive2-[[TRANSITIVE2_HASH4]]" -> "direct1-[[DIRECT1_HASH4]]";
// CHECK-NEXT:   "direct1-[[DIRECT1_HASH4]]" -> "root-[[ROOT_HASH3]]";
// CHECK-NEXT:   "direct2-[[DIRECT2_HASH4]]" -> "root-[[ROOT_HASH3]]";
// CHECK-NEXT:   "root-[[ROOT_HASH3]]" -> "B-[[OFFLOADING_ARCH]]";
// CHECK-NEXT: }

//--- module.modulemap
module root { header "root.h" }
module direct1 { header "direct1.h" }
module direct2 { header "direct2.h" }
module transitive1 { header "transitive1.h" }
module transitive2 { header "transitive2.h" }

//--- root.h
#include "direct1.h"
#include "direct2.h"

//--- direct1.h
#include "transitive1.h"
#include "transitive2.h"

//--- direct2.h
#include "transitive1.h"

//--- transitive1.h
// empty

//--- transitive2.h
// empty

//--- A.cpp
export module A;
export import :B;
import :C;

//--- A-B.cpp
module;
#include "direct1.h"
export module A:B;

//--- A-C.cpp
export module A:C;

//--- B.cpp
module;
#include "root.h"
export module B;
import A;

//--- main.cpp
#include "root.h"
import A;
import B;
