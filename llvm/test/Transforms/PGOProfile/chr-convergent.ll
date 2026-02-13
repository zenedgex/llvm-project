; Test that CHR does not transform regions containing convergent or
; noduplicate calls, following the same guard as SimplifyCFG.
;
; CHR (Control Height Reduction) merges multiple biased branches into a
; single speculative check, cloning the region into hot/cold paths. On GPU
; targets, this merged branch may be divergent (per-thread), splitting the
; wavefront: some threads take the hot path, others the cold path.
;
; A convergent call like ds_bpermute (a cross-lane operation on AMDGPU)
; requires a specific set of threads to be active — when thread X reads
; from thread Y via ds_bpermute, thread Y must be active and participating
; in the same call. After CHR cloning, thread Y may have gone to the cold
; path while thread X is on the hot path, so the hot-path ds_bpermute reads
; a stale register value from thread Y instead of the intended value.
;
; Similarly, noduplicate calls must not be duplicated by definition.
;
; RUN: opt < %s -passes='require<profile-summary>,function(chr)' -S | FileCheck %s

target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"
target triple = "amdgcn-amd-amdhsa"

declare i32 @llvm.amdgcn.workitem.id.x()
declare i32 @llvm.amdgcn.ds.bpermute(i32, i32) #0

; Two biased divergent branches where the first region contains a convergent
; cross-lane operation (ds_bpermute). CHR must not clone this region.
;
; Original code (should be preserved as-is):
;   if (val > 0)           // Biased true, per-thread condition
;     result = bpermute()  // Cross-lane read: thread X reads from thread Y
;   if (val < 100)         // Biased true, per-thread condition
;     output[tid] = result
;
; Without this fix, CHR would transform to:
;   if (val > 0 && val < 100) {  // Merged speculative branch (hot path)
;     result = bpermute()        // BUG: thread Y may not be on hot path,
;                                //   so thread X reads stale value from Y
;     output[tid] = result
;   } else {                     // Cold path (.nonchr clone)
;     if (val > 0) result = bpermute()
;     if (val < 100) output[tid] = result
;   }
;
; The merged branch splits the wavefront differently than the original
; branches, changing which threads are active at the bpermute call site.
;
define amdgpu_kernel void @test_chr_convergent(
    ptr addrspace(1) %input,
    ptr addrspace(1) %output) !prof !14 {
; CHECK-LABEL: @test_chr_convergent(
; CHECK-NOT: nonchr
; CHECK: entry:
; CHECK:   %cond1 = icmp sgt i32 %val, 0
; CHECK:   br i1 %cond1, label %bb1, label %merge1
; CHECK: bb1:
; CHECK:   %perm = call i32 @llvm.amdgcn.ds.bpermute(i32 %lane_idx, i32 %val)
; CHECK: merge1:
; CHECK:   %cond2 = icmp slt i32 %val, 100
; CHECK:   br i1 %cond2, label %bb2, label %merge2
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep_in = getelementptr inbounds i32, ptr addrspace(1) %input, i32 %tid
  %val = load i32, ptr addrspace(1) %gep_in, align 4
  %cond1 = icmp sgt i32 %val, 0
  br i1 %cond1, label %bb1, label %merge1, !prof !15

bb1:
  %lane_idx = shl i32 %tid, 2
  %perm = call i32 @llvm.amdgcn.ds.bpermute(i32 %lane_idx, i32 %val)
  br label %merge1

merge1:
  %result = phi i32 [ %perm, %bb1 ], [ 0, %entry ]
  %cond2 = icmp slt i32 %val, 100
  br i1 %cond2, label %bb2, label %merge2, !prof !15

bb2:
  %gep_out = getelementptr inbounds i32, ptr addrspace(1) %output, i32 %tid
  store i32 %result, ptr addrspace(1) %gep_out, align 4
  br label %merge2

merge2:
  ret void
}

; Same pattern but with a noduplicate call instead of convergent.
; CHR must also skip this region.
declare void @noduplicate_callee() #1

define amdgpu_kernel void @test_chr_noduplicate(
    ptr addrspace(1) %input,
    ptr addrspace(1) %output) !prof !14 {
; CHECK-LABEL: @test_chr_noduplicate(
; CHECK-NOT: nonchr
; CHECK: entry:
; CHECK:   br i1 %cond1, label %bb1, label %merge1
; CHECK: bb1:
; CHECK:   call void @noduplicate_callee()
; CHECK: merge1:
; CHECK:   br i1 %cond2, label %bb2, label %merge2
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep_in = getelementptr inbounds i32, ptr addrspace(1) %input, i32 %tid
  %val = load i32, ptr addrspace(1) %gep_in, align 4
  %cond1 = icmp sgt i32 %val, 0
  br i1 %cond1, label %bb1, label %merge1, !prof !15

bb1:
  call void @noduplicate_callee()
  br label %merge1

merge1:
  %cond2 = icmp slt i32 %val, 100
  br i1 %cond2, label %bb2, label %merge2, !prof !15

bb2:
  %gep_out = getelementptr inbounds i32, ptr addrspace(1) %output, i32 %tid
  store i32 %val, ptr addrspace(1) %gep_out, align 4
  br label %merge2

merge2:
  ret void
}

; A case without convergent or noduplicate calls — CHR should still transform.
; This verifies the fix is targeted (not overly broad).
define amdgpu_kernel void @test_chr_no_convergent(
    ptr addrspace(1) %input,
    ptr addrspace(1) %output) !prof !14 {
; CHECK-LABEL: @test_chr_no_convergent(
; CHECK: entry.split:
; CHECK: entry.split.nonchr:
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep_in = getelementptr inbounds i32, ptr addrspace(1) %input, i32 %tid
  %val = load i32, ptr addrspace(1) %gep_in, align 4
  %cond1 = icmp sgt i32 %val, 0
  br i1 %cond1, label %bb1, label %merge1, !prof !15

bb1:
  %doubled = mul i32 %val, 2
  br label %merge1

merge1:
  %result = phi i32 [ %doubled, %bb1 ], [ 0, %entry ]
  %cond2 = icmp slt i32 %val, 100
  br i1 %cond2, label %bb2, label %merge2, !prof !15

bb2:
  %gep_out = getelementptr inbounds i32, ptr addrspace(1) %output, i32 %tid
  store i32 %result, ptr addrspace(1) %gep_out, align 4
  br label %merge2

merge2:
  ret void
}

attributes #0 = { convergent nounwind readnone willreturn }
attributes #1 = { noduplicate nounwind }

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"ProfileSummary", !1}
!1 = !{!2, !3, !4, !5, !6, !7, !8, !9}
!2 = !{!"ProfileFormat", !"InstrProf"}
!3 = !{!"TotalCount", i64 10000}
!4 = !{!"MaxCount", i64 10}
!5 = !{!"MaxInternalCount", i64 1}
!6 = !{!"MaxFunctionCount", i64 1000}
!7 = !{!"NumCounts", i64 3}
!8 = !{!"NumFunctions", i64 3}
!9 = !{!"DetailedSummary", !10}
!10 = !{!11, !12, !13}
!11 = !{i32 10000, i64 100, i32 1}
!12 = !{i32 999000, i64 100, i32 1}
!13 = !{i32 999999, i64 1, i32 2}

!14 = !{!"function_entry_count", i64 100}
!15 = !{!"branch_weights", i32 999, i32 1}
