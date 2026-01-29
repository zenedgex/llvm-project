//===---------------------- AMDGPUNextUseAnalysis.cpp ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AMDGPUNextUseAnalysis.h"
#include "AMDGPU.h"
#include "GCNRegPressure.h"
#include "GCNSubtarget.h"

#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/IR/ModuleSlotTracker.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/ToolOutputFile.h"

#include <cmath>
#include <limits>
#include <queue>

using namespace llvm;

#define DEBUG_TYPE "amdgpu-next-use-analysis"

static cl::opt<bool>
    DumpNextUseDistance("amdgpu-next-use-analysis-dump-distance",
                        cl::init(false), cl::Hidden);

static cl::opt<std::string>
    DumpNextUseDistanceAsJson("amdgpu-next-use-analysis-dump-distance-as-json",
                              cl::Hidden);
static cl::opt<bool>
    DumpNextUseDistanceVerbose("amdgpu-next-use-analysis-dump-distance-verbose",
                               cl::init(false), cl::Hidden);

static cl::opt<AMDGPUNextUseAnalysis::CompatibilityMode> CompatModeOpt(
    "amdgpu-next-use-analysis-compatibility-mode", cl::Hidden,
    cl::init(AMDGPUNextUseAnalysis::CompatibilityMode::Graphics),
    cl::values(
        clEnumValN(AMDGPUNextUseAnalysis::CompatibilityMode::Graphics,
                   "graphics", "TBD"),
        clEnumValN(AMDGPUNextUseAnalysis::CompatibilityMode::MachineLearning,
                   "machine-learning", "TBD")));

namespace {
double encodeLoopDepth(double Depth) {
  constexpr double LoopWeight = 1000.0;
  return std::pow(LoopWeight, Depth);
}

/// Represents a distance to a machine basic block.
/// Used for returning both the distance and the target block together.
struct MBBDistPair {
  double Distance;
  const MachineBasicBlock *MBB;
  constexpr MBBDistPair()
      : Distance(std::numeric_limits<double>::max()), MBB(nullptr) {}
  MBBDistPair(double D, const MachineBasicBlock *B) : Distance(D), MBB(B) {}
};

/// Represents a live register use with its distance.
/// Used for tracking and sorting register uses by distance.
struct LiveRegUse {
  const MachineOperand *Use = nullptr;
  double Dist = 0.0;
  LiveRegUse() = default;
  LiveRegUse(const MachineOperand *Use, double Dist) : Use(Use), Dist(Dist) {}

  bool valid() { return Use; }

  Register getReg() const { return Use->getReg(); }
  LaneBitmask getLaneMask(const SIRegisterInfo *TRI) const {
    return TRI->getSubRegIndexLaneMask(Use->getSubReg());
  }

  bool operator<(const LiveRegUse &Other) const {
    if (!Use || Dist < Other.Dist)
      return true;
    // Ensure deterministic results (that match v1)
    return Dist == Other.Dist && Other.getReg() < getReg();
  }
};

/// Helper struct to hold parsed instruction information for JSON output.
struct InstructionInfo {
  std::string MIStr; // Backing storage for StringRefs
  StringRef DefName;
  StringRef DefType;
  StringRef Instr;
};

std::string Quote(StringRef S) { return "\"" + S.str() + "\""; }
std::string Sep(bool Final) { return std::string(Final ? "" : ","); }
llvm::format_object<double> Fmt(double Dist) { return format("%.1f", Dist); }
} // namespace

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// AMDGPUNextUseAnalysisImpl
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class llvm::AMDGPUNextUseAnalysisImpl {
  using CompatibilityMode = AMDGPUNextUseAnalysis::CompatibilityMode;
  const MachineFunction *MF = nullptr;
  const SIRegisterInfo *TRI = nullptr;
  const SIInstrInfo *TII = nullptr;
  const MachineLoopInfo *MLI = nullptr;
  const MachineDominatorTree *DT = nullptr;
  const MachineRegisterInfo *MRI = nullptr;
  DenseMap<const MachineInstr *, double> InstrToId;
  CompatibilityMode CompatMode;

  void initializeTables() {
    for (const MachineBasicBlock &BB : *MF)
      calcInstrIds(&BB, InstrToId);
  }

  void clearTables() {
    InstrToId.clear();
    RegUseMap.clear();
    Paths.clear();
  }

  bool mlMode() const {
    return CompatMode == CompatibilityMode::MachineLearning;
  }

  bool gfxMode() const { return CompatMode == CompatibilityMode::Graphics; }

  //----------------------------------------------------------------------------
  // Instruction Ids
  //----------------------------------------------------------------------------
  void calcInstrIds(const MachineBasicBlock *BB,
                    DenseMap<const MachineInstr *, double> &InstrToId) const;

  /// Returns MI's instruction Id. It renumbers (part of) the BB if MI is not
  /// found in the map.
  double getInstrId(const MachineInstr *MI) const;
  double getInstrId(MachineBasicBlock::const_instr_iterator I) const {
    return getInstrId(&*I);
  }

  // Length of the segment from MI (inclusive) to the first instruction of the
  // basic block.
  double getHeadLen(const MachineInstr *MI) const {
    const MachineBasicBlock *MBB = MI->getParent();
    return getInstrId(MI) + getInstrId(&MBB->instr_front()) + 1;
  }

  // Length of the segment from MI (exclusive) to the last instruction of the
  // basic block.
  double getTailLen(const MachineInstr *MI) const {
    const MachineBasicBlock *MBB = MI->getParent();
    return getInstrId(&MBB->instr_back()) - getInstrId(MI);
  }

  // Length of the segment from 'From' to 'To' (exclusive). Both instructions
  // must in the same basic block.
  double getDistance(const MachineInstr *From, const MachineInstr *To) const {
    assert(From->getParent() == To->getParent());
    return getInstrId(To) - getInstrId(From);
  }

  //----------------------------------------------------------------------------
  // RegUses
  //----------------------------------------------------------------------------
private:
  DenseMap<unsigned, SmallVector<const MachineOperand *>> RegUseMap;

  const SmallVector<const MachineOperand *> &getRegisterUses(unsigned Reg) {
    auto I = RegUseMap.find(Reg);
    if (I != RegUseMap.end())
      return I->second;

    SmallVector<const MachineOperand *> &Uses = RegUseMap[Reg];
    for (const MachineOperand &UseMO : MRI->use_nodbg_operands(Reg))
      Uses.push_back(&UseMO);
    return Uses;
  }

  //----------------------------------------------------------------------------
  // Paths
  //----------------------------------------------------------------------------
private:
  class Path {
  private:
    const MachineBasicBlock *Src;
    const MachineBasicBlock *Dst;

  public:
    Path(const MachineBasicBlock *Src, const MachineBasicBlock *Dst)
        : Src(Src), Dst(Dst) {}
    Path() : Src(nullptr), Dst(nullptr) {}
    Path(const Path &Other) = default;
    Path &operator=(const Path &Other) = default;

    bool operator==(const Path &Other) const {
      return Src == Other.Src && Dst == Other.Dst;
    }
    bool operator!=(const Path &Other) const {
      return !this->operator==(Other);
    }

    const MachineBasicBlock *src() const { return Src; }
    const MachineBasicBlock *dst() const { return Dst; }
  };

  struct PathDenseMapInfo {
    using MBBPtrInfo = DenseMapInfo<MachineBasicBlock *>;

    static inline Path getEmptyKey() {
      return Path(MBBPtrInfo::getEmptyKey(), MBBPtrInfo::getEmptyKey());
    }
    static inline Path getTombstoneKey() {
      return Path(MBBPtrInfo::getTombstoneKey(), MBBPtrInfo::getTombstoneKey());
    }

    static unsigned getHashValue(const Path &Val) {
      return detail::combineHashValue(MBBPtrInfo::getHashValue(Val.src()),
                                      MBBPtrInfo::getHashValue(Val.dst()));
    }
    static bool isEqual(const Path &LHS, const Path &RHS) { return LHS == RHS; }
  };

  struct PathInfo {
    bool Edge;
    bool Backedge;
    bool Reachable;
    double LoopWeight;
    std::optional<double> ShortestDistance;
    std::optional<double> ShortestUnweightedDistance;
    double Size;
  };
  DenseMap<Path, PathInfo, PathDenseMapInfo> Paths;

  PathInfo &mutPathInfoFor(const MachineBasicBlock *From,
                           const MachineBasicBlock *To) const;
  const PathInfo &pathInfoFor(const MachineBasicBlock *From,
                              const MachineBasicBlock *To) const {
    return mutPathInfoFor(From, To);
  }

  //----------------------------------------------------------------------------
  // Calculate features
  //----------------------------------------------------------------------------
  double calcSize(const MachineBasicBlock *BB) const;
  double calcWeightedSize(const MachineBasicBlock *From,
                          const MachineBasicBlock *To) const;
  double calcLoopWeight(const MachineBasicBlock *From,
                        const MachineBasicBlock *To) const;
  bool calcIsReachable(const MachineBasicBlock *From,
                       const MachineBasicBlock *To) const;
  bool calcIsBackedge(const MachineBasicBlock *From,
                      const MachineBasicBlock *To) const;

  double calcShortestPath(const MachineBasicBlock *From,
                          const MachineBasicBlock *To, bool Unweighted) const;

  /// If the path from \p MI to \p UseMI does not cross any loops, then this
  /// \returns the shortest instruction distance between them.
  double calcShortestDistance(const MachineInstr *MI,
                              const MachineInstr *UseMI) const;
  double calcShortestUnweightedDistance(const MachineInstr *MI,
                                        const MachineInstr *UseMI) const;

  //----------------------------------------------------------------------------
  // Feature getters. Use cached results if available. If not calculate.
  //----------------------------------------------------------------------------
  double getSize(const MachineBasicBlock *BB) const {
    return pathInfoFor(BB, BB).Size;
  }

  bool isReachable(const MachineBasicBlock *From,
                   const MachineBasicBlock *To) const {
    return pathInfoFor(From, To).Reachable;
  }

  bool isDistanceFinite(const MachineBasicBlock *From,
                        const MachineBasicBlock *To) const {
    if (From == To)
      return false;
    return getShortestPath(From, To) != std::numeric_limits<double>::max();
  }

  // Can be used as a substitute for DT->dominates(A, B) if A and B are in the
  // same basic block.
  bool instrsAreInOrder(const MachineInstr *A, const MachineInstr *B) const {
    assert(A->getParent() == B->getParent() &&
           "instructions must be in the same basic block!");
    if (A == B || getInstrId(A) < getInstrId(B))
      return true;
    if (!A->isPHI())
      return false;
    if (!B->isPHI())
      return true;
    for (auto &PHI : A->getParent()->phis()) {
      if (&PHI == A)
        return true;
      if (&PHI == B)
        return false;
    }
    return false;
  }

  double getLoopWeight(const MachineBasicBlock *From,
                       const MachineBasicBlock *To) const {
    return pathInfoFor(From, To).LoopWeight;
  }

  /// Calculates the shortest distance and caches it.
  double getShortestPath(const MachineBasicBlock *From,
                         const MachineBasicBlock *To) const {
    std::optional<double> &D = mutPathInfoFor(From, To).ShortestDistance;
    if (!D.has_value())
      D = calcShortestPath(From, To, /* Unweighted */ false);
    return D.value();
  }

  double getShortestUnweightedPath(const MachineBasicBlock *From,
                                   const MachineBasicBlock *To) const {
    std::optional<double> &D =
        mutPathInfoFor(From, To).ShortestUnweightedDistance;
    if (!D.has_value())
      D = calcShortestPath(From, To, /* Unweighted */ true);
    return D.value();
  }

  //----------------------------------------------------------------------------
  // CFG Helpers
  //----------------------------------------------------------------------------
  /// /Returns the shortest distance between a given basic block \p CurMBB and
  /// its closest exiting latch of \p CurLoop.
  MBBDistPair
  calcShortestDistanceToExitingLatch(const MachineBasicBlock *CurMBB,
                                     const MachineLoop *CurLoop) const;

  /// Helper function that finds the shortest instruction path in \p CurMMB's
  /// loop that includes \p CurMBB and starts from the loop header and ends at
  /// the earliest loop latch. \Returns the path cost and the earliest latch
  /// MBB.
  MBBDistPair
  calcLoopDistanceAndExitingLatch(const MachineBasicBlock *CurMBB) const;

  MBBDistPair calcLoopWeightedDistanceAndExitingLatch(
      const MachineBasicBlock *CurMBB) const {
    MBBDistPair Exit = calcLoopDistanceAndExitingLatch(CurMBB);
    Exit.Distance *= encodeLoopDepth(1);
    return Exit;
  }

  MBBDistPair
  calcWeightedLoopDistance(MachineLoop *ML, const MachineBasicBlock *UseMBB,
                           bool IsUseOutsideOfTheCurrentLoopNest) const;

  /// Helper function for calculating the minimum instruction distance from the
  /// outer loop header to the outer loop latch.
  MBBDistPair calcNestedLoopDistanceAndExitingLatchToOutsideUse(
      const MachineBasicBlock *CurMBB, const MachineBasicBlock *UseMBB) const;
  MBBDistPair calcNestedLoopDistanceAndExitingLatchToParentUse(
      const MachineBasicBlock *CurMBB, const MachineBasicBlock *UseMBB) const;

  /// Given \p DefMI in a loop and \p UseMI outside the loop, this function
  /// returns the minimum instruction path between \p DefMI and \p UseMI.
  /// Please note that since \p DefMI is in a loop we don't care about the
  /// exact position of the instruction in the block because we are making a
  /// rough estimate of the dynamic instruction path length, given that the loop
  /// iterates multiple times.
  double calcInsideToOutsideLoopDistance(Register DefReg,
                                         LaneBitmask DefLaneMask,
                                         const MachineInstr *DefMI,
                                         const MachineInstr *UseMI) const;

  /// \Returns the shortest path distance from \p CurMI to the end of the loop
  /// latch plus the distance from the top of the loop header to the PHI use.
  double calcBackedgeDistance(const MachineInstr *CurMI,
                              const MachineInstr *UseMI) const;

  /// \Returns true if the use of \p DefReg (\p UseMI) is a PHI in the loop
  /// header, i.e., DefReg is flowing through the back-edge.
  bool isIncomingValFromBackedge(const MachineInstr *CurMI,
                                 const MachineInstr *UseMI, Register DefReg,
                                 LaneBitmask DefLaneMask) const;

  /// Helper to calculate distance from current instruction to a single use.
  double calcDistanceToUse(Register LiveReg, LaneBitmask LaneMask,
                           const MachineInstr &CurMI,
                           const MachineOperand *UseMO) const;

  //----------------------------------------------------------------------------
  // Debug/Developer Helpers
  //----------------------------------------------------------------------------

  /// Goes over all MBB pairs in \p MF, calculates the shortest path between
  /// them and fills in \p ShortestPathTable.
  void populatePathTable();
  void dumpShortestPaths() const;
  void printAllDistances();

  //----------------------------------------------------------------------------
  // Helper methods for printFurthestDistancesAsJson
  //----------------------------------------------------------------------------
  InstructionInfo parseInstructionString(const MachineInstr &MI,
                                         ModuleSlotTracker &MST) const;
  void collectDefinedRegisters(const MachineInstr &MI,
                               SmallSet<unsigned, 4> &Defs) const;
  void computeLiveRegUses(
      const MachineInstr &MI, const GCNRPTracker::LiveRegSet &LiveRegs,
      const SmallSet<unsigned, 4> &Defs,
      DenseMap<const MachineOperand *, LiveRegUse> &RelevantUses,
      LiveRegUse &Furthest, LiveRegUse *FurthestSubreg = nullptr);
  void printInstructionHeader(raw_ostream &OS, const MachineInstr &MI,
                              ModuleSlotTracker &MST) const;
  void printDistances(
      raw_ostream &OS,
      const DenseMap<const MachineOperand *, LiveRegUse> &Uses) const;
  void printFurthestUse(raw_ostream &OS, const LiveRegUse &Furthest,
                        bool Subreg = false, bool Last = false) const;

  //----------------------------------------------------------------------------
  // Misc Helpers
  //----------------------------------------------------------------------------
  bool machineOperandCoveredBy(const MachineOperand &MO,
                               LaneBitmask LaneMask) const {
    LaneBitmask Mask = TRI->getSubRegIndexLaneMask(MO.getSubReg());
    return (Mask & LaneMask) == Mask;
  }

public:
  AMDGPUNextUseAnalysisImpl() = default;
  ~AMDGPUNextUseAnalysisImpl() { clearTables(); }

  void initialize(const MachineFunction *, const MachineLoopInfo *,
                  const MachineDominatorTree *);

  CompatibilityMode getCompatibilityMode() { return CompatMode; }

  void setCompatibilityMode(CompatibilityMode Mode) {
    CompatMode = Mode;
    clearTables();
    initializeTables();
  }

  /// \Returns the next-use distance for \p DefReg.
  std::optional<double>
  getNextUseDistance(Register LiveReg, LaneBitmask LaneMask,
                     const MachineInstr &FromMI,
                     const SmallVector<const MachineOperand *> &Uses,
                     SmallVector<double> *Distances = nullptr,
                     const MachineOperand **UseOut = nullptr);

  std::optional<double>
  getNextUseDistance(Register LiveReg, const MachineInstr &FromMI,
                     const SmallVector<const MachineOperand *> &Uses,
                     SmallVector<double> *Distances = nullptr,
                     const MachineOperand **UseOut = nullptr) {
    return getNextUseDistance(
        LiveReg,
        MRI->getMaxLaneMaskForVReg(
            LiveReg), // FIXME: would LaneBitmask::all() work?
        FromMI, Uses, Distances, UseOut);
  }

  void getUses(unsigned Register, LaneBitmask LaneMask, const MachineInstr &MI,
               SmallVector<const MachineOperand *> &Uses);

  void printFurthestDistancesAsJson(raw_ostream &OS, const LiveIntervals *LIS);
};

void AMDGPUNextUseAnalysisImpl::calcInstrIds(
    const MachineBasicBlock *BB,
    DenseMap<const MachineInstr *, double> &InstrToId) const {
  double Id = 0.0;
  for (auto &MI : BB->instrs()) {
    InstrToId[&MI] = Id;
    if (!mlMode() || !MI.isPHI())
      ++Id;
  }
}

/// Returns MI's instruction Id. It renumbers (part of) the BB if MI is not
/// found in the map.
double AMDGPUNextUseAnalysisImpl::getInstrId(const MachineInstr *MI) const {
  auto It = InstrToId.find(MI);
  if (It != InstrToId.end())
    return It->second;

  // Renumber the MBB.
  // TODO: Renumber from MI onwards.
  auto MutInstrToId =
      const_cast<DenseMap<const MachineInstr *, double> &>(InstrToId);
  calcInstrIds(MI->getParent(), MutInstrToId);
  return InstrToId.find(MI)->second;
}

AMDGPUNextUseAnalysisImpl::PathInfo &
AMDGPUNextUseAnalysisImpl::mutPathInfoFor(const MachineBasicBlock *From,
                                          const MachineBasicBlock *To) const {
  auto &MutPaths = const_cast<AMDGPUNextUseAnalysisImpl *>(this)->Paths;
  Path P(From, To);
  auto I = MutPaths.find(P);
  if (I != MutPaths.end())
    return I->second;

  PathInfo &Slot = MutPaths[P];
  Slot.Edge = From->isSuccessor(To);
  Slot.Backedge = calcIsBackedge(From, To);
  Slot.Reachable = calcIsReachable(From, To);
  Slot.LoopWeight = calcLoopWeight(From, To);
  Slot.Size = From == To ? calcSize(From) : std::numeric_limits<double>::max();
  return Slot;
}

double AMDGPUNextUseAnalysisImpl::calcSize(const MachineBasicBlock *BB) const {
  double Size = BB->size();
  if (mlMode())
    Size -= std::distance(BB->begin(), BB->getFirstNonPHI());
  return Size;
}

double
AMDGPUNextUseAnalysisImpl::calcWeightedSize(const MachineBasicBlock *From,
                                            const MachineBasicBlock *To) const {
  double LoopWeight = getLoopWeight(From, To);
  if (LoopWeight == 0.0)
    LoopWeight = 1.0;
  return getSize(From) * LoopWeight;
}

static double getEffectiveLoopDepth(MachineLoop *Loop,
                                    const MachineBasicBlock *To,
                                    const MachineLoopInfo *MLI) {
  double LoopDepth = 0.0;
  MachineLoop *const End = Loop->getOutermostLoop()->getParentLoop();
  for (MachineLoop *TmpLoop = Loop; TmpLoop != End;
       TmpLoop = TmpLoop->getParentLoop()) {
    if (TmpLoop->contains(To))
      continue;
    LoopDepth++;
  }
  return LoopDepth;
}

double
AMDGPUNextUseAnalysisImpl::calcLoopWeight(const MachineBasicBlock *From,
                                          const MachineBasicBlock *To) const {
  MachineLoop *LoopFrom = MLI->getLoopFor(From);
  MachineLoop *LoopTo = MLI->getLoopFor(To);

  if (!LoopFrom)
    return 0.0;

  if (!LoopTo)
    return encodeLoopDepth(getEffectiveLoopDepth(LoopFrom, To, MLI));

  if (LoopFrom->contains(LoopTo)) // covers LoopFrom == LoopTo
    return 1.0;

  if (LoopTo->contains(LoopFrom))
    return encodeLoopDepth(MLI->getLoopDepth(From) - MLI->getLoopDepth(To));

  return encodeLoopDepth(getEffectiveLoopDepth(LoopFrom, To, MLI));
}

bool AMDGPUNextUseAnalysisImpl::calcIsReachable(
    const MachineBasicBlock *From, const MachineBasicBlock *To) const {
  std::queue<const MachineBasicBlock *> Queue;
  DenseSet<const MachineBasicBlock *> Visited;

  Queue.push(From);
  Visited.insert(From);

  while (!Queue.empty()) {
    const MachineBasicBlock *Current = Queue.front();
    Queue.pop();

    for (const MachineBasicBlock *Succ : Current->successors()) {
      if (Succ == To)
        return true;

      Path P(Succ, To);
      auto I = Paths.find(P);
      if (I != Paths.end()) {
        if (I->second.Reachable)
          return true;
        continue;
      }

      if (Visited.insert(Succ).second)
        Queue.push(Succ);
    }
  }
  return false;
}

bool AMDGPUNextUseAnalysisImpl::calcIsBackedge(
    const MachineBasicBlock *From, const MachineBasicBlock *To) const {
  if (!From->isSuccessor(To))
    return false;
  MachineLoop *Loop1 = MLI->getLoopFor(From);
  MachineLoop *Loop2 = MLI->getLoopFor(To);
  if (!Loop1 || !Loop2 || Loop1 != Loop2)
    return false;
  MachineBasicBlock *LoopHeader = Loop1->getHeader();
  if (To != LoopHeader)
    return false;
  SmallVector<MachineBasicBlock *, 2> Latches;
  Loop1->getLoopLatches(Latches);
  auto It = llvm::find(Latches, From);
  return It != Latches.end();
}

double
AMDGPUNextUseAnalysisImpl::calcShortestPath(const MachineBasicBlock *FromMBB,
                                            const MachineBasicBlock *ToMBB,
                                            bool Unweighted) const {

  assert(FromMBB != ToMBB && "The basic blocks should be different.");
  DenseSet<const MachineBasicBlock *> Visited;
  struct Data {
    const MachineBasicBlock *BestPred = nullptr;
    double ShortestDistance = std::numeric_limits<double>::max();
  };
  DenseMap<const MachineBasicBlock *, Data> MBBData;

  auto Cmp = [&MBBData](const MachineBasicBlock *MBB1,
                        const MachineBasicBlock *MBB2) {
    return MBBData[MBB1].ShortestDistance > MBBData[MBB2].ShortestDistance;
  };
  std::priority_queue<const MachineBasicBlock *,
                      std::vector<const MachineBasicBlock *>, decltype(Cmp)>
      Worklist(Cmp);

  Worklist.push(FromMBB);
  MBBData[FromMBB] = {nullptr, 0.0};

  while (!Worklist.empty()) {
    const MachineBasicBlock *CurMBB = Worklist.top();
    Worklist.pop();

    if (!Visited.insert(CurMBB).second)
      continue;

    if (CurMBB == ToMBB) {
      auto *Pred = MBBData[CurMBB].BestPred;
      return MBBData[Pred].ShortestDistance - MBBData[FromMBB].ShortestDistance;
    }

    auto Pair = MBBData.try_emplace(
        CurMBB, Data{nullptr, std::numeric_limits<double>::max()});
    double CurrMBBDist = Pair.first->second.ShortestDistance;

    for (MachineBasicBlock *Succ : CurMBB->successors()) {
      const AMDGPUNextUseAnalysisImpl::PathInfo &PI = pathInfoFor(CurMBB, Succ);
      if (PI.Backedge)
        if (gfxMode())
          continue;

      double AB = Unweighted ? getSize(Succ) : calcWeightedSize(Succ, ToMBB);
      double NewSuccDist = CurrMBBDist + AB;

      auto &[SuccPred, SuccDist] = MBBData[Succ];
      if (NewSuccDist < SuccDist) {
        // We found a better path to Succ, update best predecessor and distance
        SuccPred = CurMBB;
        SuccDist = NewSuccDist;
      }

      Worklist.push(Succ);
    }
  }
  return std::numeric_limits<double>::max();
}

double AMDGPUNextUseAnalysisImpl::calcShortestDistance(
    const MachineInstr *CurMI, const MachineInstr *UseMI) const {
  const MachineBasicBlock *CurMBB = CurMI->getParent();
  const MachineBasicBlock *UseMBB = UseMI->getParent();

  if (CurMBB == UseMBB)
    return getDistance(CurMI, UseMI);

  double CurMITailLen = getTailLen(CurMI);
  double UseHeadLen = getHeadLen(UseMI);
  double Dst = getShortestPath(CurMBB, UseMBB);
  assert(Dst != std::numeric_limits<double>::max() &&
         "calcShortestDistance called for instructions in non-reachable"
         " basic blocks!");
  return CurMITailLen + Dst + UseHeadLen;
}

double AMDGPUNextUseAnalysisImpl::calcShortestUnweightedDistance(
    const MachineInstr *CurMI, const MachineInstr *UseMI) const {
  const MachineBasicBlock *CurMBB = CurMI->getParent();
  const MachineBasicBlock *UseMBB = UseMI->getParent();

  if (CurMBB == UseMBB)
    return getDistance(CurMI, UseMI);

  double CurMITailLen = getTailLen(CurMI);
  double UseHeadLen = getHeadLen(UseMI);
  double Dst = getShortestUnweightedPath(CurMBB, UseMBB);
  assert(Dst != std::numeric_limits<double>::max() &&
         "calcShortestUnweightedDistance called for instructions in"
         " non-reachable basic blocks!");
  return CurMITailLen + Dst + UseHeadLen;
}

MBBDistPair AMDGPUNextUseAnalysisImpl::calcShortestDistanceToExitingLatch(
    const MachineBasicBlock *CurMBB, const MachineLoop *CurLoop) const {

  SmallVector<MachineBasicBlock *, 2> Latches;
  CurLoop->getLoopLatches(Latches);
  MBBDistPair Exit;

  for (MachineBasicBlock *LMBB : Latches) {
    if (LMBB == CurMBB)
      return {0.0, CurMBB};

    double Dst = getShortestPath(CurMBB, LMBB);
    if (Exit.Distance > Dst) {
      Exit.Distance = Dst;
      Exit.MBB = LMBB;
    }
  }
  return Exit;
}

MBBDistPair AMDGPUNextUseAnalysisImpl::calcLoopDistanceAndExitingLatch(
    const MachineBasicBlock *CurMBB) const {

  // This is a hot spot. Check it before doing anything else.
  MachineLoop *CurLoop = MLI->getLoopFor(CurMBB);
  if (CurLoop->getNumBlocks() == 1)
    return {getSize(CurMBB), CurMBB};

  MachineBasicBlock *LoopHeader = CurLoop->getHeader();
  SmallVector<MachineBasicBlock *, 2> Latches;
  CurLoop->getLoopLatches(Latches);
  bool IsCurLoopLatch = llvm::any_of(
      Latches, [&](MachineBasicBlock *LMBB) { return CurMBB == LMBB; });

  if (CurMBB == LoopHeader) {
    MBBDistPair Exit = calcShortestDistanceToExitingLatch(CurMBB, CurLoop);
    Exit.Distance += getSize(LoopHeader) + getSize(Exit.MBB);
    return Exit;
  }

  if (IsCurLoopLatch)
    return {getSize(LoopHeader) + getShortestPath(LoopHeader, CurMBB) +
                getSize(CurMBB),
            CurMBB};

  double LoopHeaderToCurMBBDistance = getShortestPath(LoopHeader, CurMBB);

  MBBDistPair Exit = calcShortestDistanceToExitingLatch(CurMBB, CurLoop);
  Exit.Distance += getSize(LoopHeader) + LoopHeaderToCurMBBDistance +
                   getSize(CurMBB) + getSize(Exit.MBB);
  return Exit;
}

MBBDistPair AMDGPUNextUseAnalysisImpl::calcWeightedLoopDistance(
    MachineLoop *ML, const MachineBasicBlock *UseMBB,
    bool IsUseOutsideOfTheCurrentLoopNest) const {
  if (ML->getNumBlocks() == 1) {
    double UseLoopDepth = IsUseOutsideOfTheCurrentLoopNest
                              ? 0.0
                              : static_cast<double>(MLI->getLoopDepth(UseMBB));
    return {
        getSize(ML->getHeader()) *
            encodeLoopDepth(MLI->getLoopDepth(ML->getHeader()) - UseLoopDepth),
        ML->getLoopLatch()};
  }

  return calcLoopWeightedDistanceAndExitingLatch(ML->getHeader());
}

// Calculates the overhead of a loop nest for three cases: 1. the use is outside
// of the current loop, but they share the same loop nest 2. the use is
// outside of the current loop nest and 3. the use is in a parent loop of the
// current loop nest.
MBBDistPair
AMDGPUNextUseAnalysisImpl::calcNestedLoopDistanceAndExitingLatchToOutsideUse(
    const MachineBasicBlock *CurMBB, const MachineBasicBlock *UseMBB) const {

  MachineLoop *CurLoop = MLI->getLoopFor(CurMBB);
  MachineLoop *UseLoop = MLI->getLoopFor(UseMBB);

  MachineLoop *OutermostLoop = CurLoop->getOutermostLoop();
  if (OutermostLoop->contains(UseLoop)) {

    // The CurLoop and the UseLoop are independent and they are in the same
    // loop nest.
    if (mlMode() && MLI->getLoopDepth(CurMBB) <= MLI->getLoopDepth(UseMBB) &&
        (CurLoop->getNumBlocks() == 1))
      return {getSize(CurLoop->getHeader()) * encodeLoopDepth(1),
              CurLoop->getLoopLatch()};

    if (MLI->getLoopDepth(CurMBB) <= MLI->getLoopDepth(UseMBB))
      return calcWeightedLoopDistance(CurLoop, UseMBB, true);

    assert(CurLoop != OutermostLoop && "The loop cannot be the outermost.");
    MachineLoop *OuterLoopOfCurLoop = CurLoop;
    while (OutermostLoop != OuterLoopOfCurLoop &&
           MLI->getLoopDepth(OuterLoopOfCurLoop->getHeader()) !=
               MLI->getLoopDepth(UseMBB)) {
      OuterLoopOfCurLoop = OuterLoopOfCurLoop->getParentLoop();
    }
    return calcWeightedLoopDistance(OuterLoopOfCurLoop, UseMBB, true);
  }

  // We should take into consideration the whole loop nest in the
  // calculation of the distance because we will reach the use after
  // executing the whole loop nest.
  return calcWeightedLoopDistance(OutermostLoop, UseMBB, true);
}

MBBDistPair
AMDGPUNextUseAnalysisImpl::calcNestedLoopDistanceAndExitingLatchToParentUse(
    const MachineBasicBlock *CurMBB, const MachineBasicBlock *UseMBB) const {
  MachineLoop *CurLoop = MLI->getLoopFor(CurMBB);
  MachineLoop *UseLoop = MLI->getLoopFor(UseMBB);

  MachineLoop *UseLoopSubLoop = nullptr;
  for (MachineLoop *ML : UseLoop->getSubLoopsVector()) {
    // All the sub-loops of the UseLoop will be executed before the use.
    // Hence, we should take this into consideration in distance calculation.
    if (ML->contains(CurLoop)) {
      UseLoopSubLoop = ML;
      break;
    }
  }
  return calcWeightedLoopDistance(UseLoopSubLoop, UseMBB, false);
}

static bool isUseOutsideOfTheCurrentLoop(const MachineLoop *UseLoop,
                                         const MachineLoop *CurLoop) {

  if (CurLoop && !UseLoop)
    return true;

  if (!CurLoop || !UseLoop)
    return false;

  if (!UseLoop->contains(CurLoop) && !CurLoop->contains(UseLoop))
    return true;

  return UseLoop->contains(CurLoop) && UseLoop != CurLoop;
}

static bool isUseOutsideOfTheCurrentLoopNest(const MachineLoop *UseLoop,
                                             const MachineLoop *CurLoop) {
  if (CurLoop && !UseLoop)
    return true;

  if (!CurLoop || !UseLoop)
    return false;

  return !UseLoop->contains(CurLoop) && !CurLoop->contains(UseLoop);
}

static bool isUseInParentLoop(const MachineLoop *UseLoop,
                              const MachineLoop *CurLoop) {
  if (!CurLoop || !UseLoop)
    return false;

  return UseLoop->contains(CurLoop) && UseLoop != CurLoop;
}

double AMDGPUNextUseAnalysisImpl::calcInsideToOutsideLoopDistance(
    Register DefReg, LaneBitmask DefLaneMask, const MachineInstr *CurMI,
    const MachineInstr *UseMI) const {
  const MachineBasicBlock *CurMBB = CurMI->getParent();
  const MachineBasicBlock *UseMBB = UseMI->getParent();
  const MachineLoop *CurLoop = MLI->getLoopFor(CurMBB);
  const MachineLoop *UseLoop = MLI->getLoopFor(UseMBB);

  MBBDistPair Exit;

  if (isUseOutsideOfTheCurrentLoopNest(UseLoop, CurLoop)) {
    if (CurLoop->getSubLoops().empty() && CurLoop->isOutermost())
      Exit = calcLoopWeightedDistanceAndExitingLatch(CurMBB);
    else
      Exit = calcNestedLoopDistanceAndExitingLatchToOutsideUse(CurMBB, UseMBB);
  } else if (isUseInParentLoop(UseLoop, CurLoop)) {
    assert(MLI->getLoopDepth(UseMBB) < MLI->getLoopDepth(CurMBB) &&
           "The loop depth of the current instruction must be bigger than "
           "these.\n");
    if (isIncomingValFromBackedge(CurMI, UseMI, DefReg, DefLaneMask))
      return calcBackedgeDistance(CurMI, UseMI);

    //  Get the loop distance of all the inner loops of UseLoop.
    Exit = calcNestedLoopDistanceAndExitingLatchToParentUse(CurMBB, UseMBB);
  }

  return Exit.Distance + getShortestPath(Exit.MBB, UseMBB) + getHeadLen(UseMI);
}

double AMDGPUNextUseAnalysisImpl::calcBackedgeDistance(
    const MachineInstr *CurMI, const MachineInstr *UseMI) const {
  const MachineBasicBlock *CurMBB = CurMI->getParent();
  const MachineBasicBlock *UseMBB = UseMI->getParent();
  MachineLoop *CurLoop = MLI->getLoopFor(CurMBB);
  MachineLoop *UseLoop = MLI->getLoopFor(UseMBB);

  assert(UseLoop && "There is no backedge.");
  double CurMITailLen = getTailLen(CurMI);
  double UseHeadLen = getHeadLen(UseMI);

  if (!CurLoop)
    return CurMITailLen + getShortestPath(CurMBB, UseMBB) + UseHeadLen;

  if (CurLoop == UseLoop) {
    MBBDistPair Exit = calcShortestDistanceToExitingLatch(CurMBB, CurLoop);
    if (Exit.MBB == CurMBB)
      return CurMITailLen + UseHeadLen;
    return UseHeadLen + CurMITailLen + Exit.Distance + getSize(Exit.MBB);
  }

  if (!CurLoop->contains(UseLoop) && !UseLoop->contains(CurLoop)) {
    MBBDistPair Exit = calcLoopDistanceAndExitingLatch(CurMBB);
    return Exit.Distance + getShortestPath(Exit.MBB, UseMBB) + UseHeadLen;
  }

  if (!CurLoop->contains(UseLoop)) {
    MBBDistPair InnerLoopExit =
        calcNestedLoopDistanceAndExitingLatchToParentUse(CurMBB, UseMBB);
    MBBDistPair Exit =
        calcShortestDistanceToExitingLatch(InnerLoopExit.MBB, UseLoop);
    return InnerLoopExit.Distance + Exit.Distance + getSize(Exit.MBB) +
           UseHeadLen;
  }

  llvm_unreachable("The backedge distance has not been calculated!");
}

bool AMDGPUNextUseAnalysisImpl::isIncomingValFromBackedge(
    const MachineInstr *CurMI, const MachineInstr *UseMI, Register DefReg,
    LaneBitmask DefLaneMask) const {
  if (!UseMI->isPHI())
    return false;

  MachineLoop *CurLoop = MLI->getLoopFor(CurMI->getParent());
  MachineLoop *UseLoop = MLI->getLoopFor(UseMI->getParent());

  if (!UseLoop || (CurLoop && !UseLoop->contains(CurLoop)) ||
      UseMI->getParent() != UseLoop->getHeader())
    return false;

  SmallVector<MachineBasicBlock *, 2> Latches;
  UseLoop->getLoopLatches(Latches);

  bool IsNotIncomingValFromLatch = false;
  bool IsIncomingValFromLatch = false;
  auto Ops = UseMI->operands();
  for (auto It = std::next(Ops.begin()), ItE = Ops.end(); It != ItE;
       It = std::next(It, 2)) {
    auto &RegMO = *It;
    auto &MBBMO = *std::next(It);
    assert(RegMO.isReg() && "Expected register operand of PHI");
    assert(MBBMO.isMBB() && "Expected MBB operand of PHI");
    if (RegMO.getReg() == DefReg &&
        machineOperandCoveredBy(RegMO, DefLaneMask)) {
      MachineBasicBlock *IncomingBB = MBBMO.getMBB();
      auto It = llvm::find(Latches, IncomingBB);
      if (It == Latches.end())
        IsNotIncomingValFromLatch = true;
      else
        IsIncomingValFromLatch = true;
    }
  }
  return IsIncomingValFromLatch && !IsNotIncomingValFromLatch;
}

double AMDGPUNextUseAnalysisImpl::calcDistanceToUse(
    Register LiveReg, LaneBitmask LiveLaneMask, const MachineInstr &CurMI,
    const MachineOperand *UseMO) const {

  const MachineInstr *UseMI = UseMO->getParent();
  const MachineBasicBlock *CurMBB = CurMI.getParent();
  const MachineBasicBlock *UseMBB = UseMI->getParent();
  MachineLoop *CurLoop = MLI->getLoopFor(CurMBB);
  MachineLoop *UseLoop = MLI->getLoopFor(UseMBB);

  // Helpers
  auto depth = [](MachineLoop *L) { return L ? L->getLoopDepth() : 0; };
  auto findTopLoop = [](MachineLoop *CurLoop, MachineLoop *UseLoop) {
    MachineLoop *L = UseLoop;
    for (;;) {
      MachineLoop *P = L->getParentLoop();
      if (!P || P == CurLoop)
        return L;
      L = P;
    }
  };

  if (isUseOutsideOfTheCurrentLoop(UseLoop, CurLoop))
    return calcInsideToOutsideLoopDistance(LiveReg, LiveLaneMask, &CurMI,
                                           UseMI);

  if (isIncomingValFromBackedge(&CurMI, UseMI, LiveReg, LiveLaneMask))
    return calcBackedgeDistance(&CurMI, UseMI);

  if (mlMode()) {
    if (depth(CurLoop) < depth(UseLoop)) {
      // 2. Inside-loop uses (< LoopTag): reset to preheader position
      //    This models: if spilled before loop, reload at preheader
      assert(UseLoop);
      MachineLoop *TopLoop = findTopLoop(CurLoop, UseLoop);
      const MachineBasicBlock *PreHdr = TopLoop->getLoopPreheader();
      return calcShortestUnweightedDistance(&CurMI, &PreHdr->back());
    }

    if (!UseLoop && CurMBB != UseMBB)
      return calcShortestUnweightedDistance(&CurMI, UseMI);

    if (CurLoop == UseLoop && CurMBB != UseMBB)
      return calcShortestDistance(&CurMI, UseMI);

    if (UseLoop && CurMBB == UseMBB && !instrsAreInOrder(&CurMI, UseMI) &&
        !UseMI->isPHI()) {
      // use is in the next loop iteration
      double CurTailLen = getTailLen(&CurMI);
      double UseHeadLen = getHeadLen(UseMI);
      MBBDistPair Exit = calcShortestDistanceToExitingLatch(UseMBB, UseLoop);
      const MachineBasicBlock *HdrMBB = UseLoop->getHeader();
      double HdrSize = getSize(HdrMBB);
      double Dst = UseMBB == HdrMBB ? 0.0 : getShortestPath(HdrMBB, UseMBB);
      return CurTailLen + Exit.Distance + HdrSize + Dst + UseHeadLen;
    }
  }

  double D = calcShortestDistance(&CurMI, UseMI);
  assert(D >= 0);
  return D;
}

void AMDGPUNextUseAnalysisImpl::populatePathTable() {
  for (const MachineBasicBlock &MBB1 : *MF) {
    for (const MachineBasicBlock &MBB2 : *MF) {
      if (&MBB1 == &MBB2)
        continue;
      getShortestPath(&MBB1, &MBB2);
    }
  }
}

void AMDGPUNextUseAnalysisImpl::dumpShortestPaths() const {
  for (const auto &P : Paths) {
    const MachineBasicBlock *From = P.first.src();
    const MachineBasicBlock *To = P.first.dst();
    std::optional<double> Dist = P.second.ShortestDistance;
    errs() << "From: " << From->getName() << "-> To:" << To->getName() << " = "
           << Dist.value_or(-1.0) << "\n";
  }
}

void AMDGPUNextUseAnalysisImpl::printAllDistances() {
  auto getRegNextUseDistance =
      [this](Register DefReg) -> std::optional<double> {
    const MachineInstr &DefMI = *MRI->def_instr_begin(DefReg);

    SmallVector<const MachineOperand *> Uses;
    for (MachineOperand &UseMO : MRI->use_nodbg_operands(DefReg))
      Uses.push_back(&UseMO);

    return getNextUseDistance(DefReg, DefMI, Uses);
  };

  for (const MachineBasicBlock &MBB : *MF) {
    for (const MachineInstr &MI : *&MBB) {
      for (const MachineOperand &MO : MI.operands()) {
        if (!MO.isReg() || MO.isUse())
          continue;

        Register Reg = MO.getReg();
        if (Reg.isPhysical() || TRI->isAGPR(*MRI, Reg))
          continue;

        std::optional<double> NextUseDistance = getRegNextUseDistance(Reg);
        errs() << "Next-use distance of Register " << printReg(Reg, TRI)
               << " = ";
        if (NextUseDistance)
          errs() << Fmt(*NextUseDistance);
        else
          errs() << "null";
        errs() << "\n";
      }
    }
  }
}

std::optional<double> AMDGPUNextUseAnalysisImpl::getNextUseDistance(
    Register LiveReg, LaneBitmask LaneMask, const MachineInstr &CurMI,
    const SmallVector<const MachineOperand *> &Uses,
    SmallVector<double> *Distances, const MachineOperand **UseOut) {

  assert(!LiveReg.isPhysical() && !TRI->isAGPR(*MRI, LiveReg) &&
         "Next-use distance is calculated for SGPRs and VGPRs");
  const MachineOperand *NextUse = nullptr;
  double NextUseDistance = std::numeric_limits<double>::max();

  if (Distances) {
    Distances->clear();
    Distances->reserve(Uses.size());
  }
  for (auto *UseMO : Uses) {
    double D = calcDistanceToUse(LiveReg, LaneMask, CurMI, UseMO);
    if (D < NextUseDistance) {
      NextUseDistance = D;
      NextUse = UseMO;
    }
    if (Distances)
      Distances->push_back(D);
  }
  if (UseOut)
    *UseOut = NextUse;
  return NextUseDistance != std::numeric_limits<double>::max()
             ? std::optional<double>(NextUseDistance)
             : std::nullopt;
}

void AMDGPUNextUseAnalysisImpl::initialize(const MachineFunction *MF,
                                           const MachineLoopInfo *ML,
                                           const MachineDominatorTree *DT) {

  this->MF = MF;
  this->MLI = ML;
  this->DT = DT;

  const GCNSubtarget &ST = MF->getSubtarget<GCNSubtarget>();
  TII = ST.getInstrInfo();
  TRI = &TII->getRegisterInfo();
  MRI = &MF->getRegInfo();

  if (CompatModeOpt.getNumOccurrences()) {
    CompatMode = CompatModeOpt;
  } else {
    // TODO: Set default based on subtarget?
    CompatMode = CompatibilityMode::Graphics;
  }

  initializeTables();

  if (DumpNextUseDistance) {
    populatePathTable();
    MF->print(errs());
    printAllDistances();
  }
}

void AMDGPUNextUseAnalysisImpl::getUses(
    unsigned Reg, LaneBitmask LaneMask, const MachineInstr &MI,
    SmallVector<const MachineOperand *> &Uses) {

  const bool CheckMask = LaneMask != LaneBitmask::getAll() &&
                         LaneMask != MRI->getMaxLaneMaskForVReg(Reg);
  const MachineBasicBlock *MBB = MI.getParent();

  for (const MachineOperand *UseMO : getRegisterUses(Reg)) {
    if (CheckMask && !machineOperandCoveredBy(*UseMO, LaneMask))
      continue;

    const MachineInstr *UseMI = UseMO->getParent();
    const MachineBasicBlock *UseMBB = UseMI->getParent();

    bool reachable;
    if (mlMode()) {
      static auto mbbFor = [](const MachineOperand *MO) {
        return MO->getParent()->getOperand(MO->getOperandNo() + 1).getMBB();
      };

      if (MBB == UseMBB) {
        reachable = instrsAreInOrder(&MI, UseMI);
        if (!reachable)
          reachable = !UseMI->isPHI() && MLI->getLoopFor(UseMBB);
        if (reachable && UseMI->isPHI()) {
          const MachineBasicBlock *EdgeSrc = mbbFor(UseMO);
          reachable = isReachable(UseMBB, EdgeSrc);
        }

      } else if (UseMI->isPHI()) {
        // allow direct backedge
        reachable = (MBB == mbbFor(UseMO)) && pathInfoFor(MBB, UseMBB).Backedge;
      } else {
        reachable = isReachable(MBB, UseMBB);
      }

    } else if (MBB == UseMBB) {
      reachable = instrsAreInOrder(&MI, UseMI);
    } else {
      reachable = isDistanceFinite(MBB, UseMBB);
    }

    if (reachable)
      Uses.push_back(UseMO);
  }
}

static std::string nameForMBB(const llvm::MachineBasicBlock &BB,
                              ModuleSlotTracker &MST) {
  std::string S;
  llvm::raw_string_ostream OS(S);
  BB.printName(OS, llvm::MachineBasicBlock::PrintNameIr, &MST);
  return OS.str();
}

static std::string printRegToString(Register Reg, LaneBitmask LaneMask,
                                    const MachineRegisterInfo *MRI,
                                    const SIRegisterInfo *TRI) {
  unsigned SubRegIdx = 0;
  if (!Reg.isVirtual() || LaneMask != MRI->getMaxLaneMaskForVReg(Reg))
    SubRegIdx = TRI->getSubRegIndexForLaneMask(LaneMask);
  std::string S;
  llvm::raw_string_ostream OS(S);
  OS << printReg(Reg, TRI, SubRegIdx, MRI);
  return OS.str();
}

template <typename T> inline std::string printToString(T &X) {
  std::string S;
  llvm::raw_string_ostream OS(S);
  X.print(OS);
  return StringRef(OS.str()).trim().str();
}

template <typename T> inline std::string printToString(T *X) {
  return X ? printToString(*X) : "null";
}

inline std::string printToString(const llvm::MachineInstr &MI,
                                 llvm::ModuleSlotTracker &MST) {
  std::string S;
  llvm::raw_string_ostream OS(S);
  MI.print(OS, MST,
           /* IsStandalone    */ false,
           /* SkipOpers       */ false,
           /* SkipDebugLoc    */ false,
           /* AddNewLine      */ false,
           /* TargetInstrInfo */ nullptr);
  return StringRef(OS.str()).trim().str();
}

InstructionInfo AMDGPUNextUseAnalysisImpl::parseInstructionString(
    const MachineInstr &MI, ModuleSlotTracker &MST) const {
  InstructionInfo Info;
  Info.MIStr = printToString(MI, MST);
  StringRef MIRef(Info.MIStr);
  StringRef Def;
  std::tie(Def, Info.Instr) = MIRef.split('=');
  if (Info.Instr.empty()) {
    Def = "%void:void";
    Info.Instr = MIRef;
  }
  Info.Instr = Info.Instr.trim();
  std::tie(Info.DefName, Info.DefType) = Def.trim().split(":");
  return Info;
}

void AMDGPUNextUseAnalysisImpl::collectDefinedRegisters(
    const MachineInstr &MI, SmallSet<unsigned, 4> &Defs) const {
  for (const MachineOperand &MO : MI.all_defs())
    if (MO.isReg() && MO.getReg().isValid())
      Defs.insert(MO.getReg());
}

void AMDGPUNextUseAnalysisImpl::computeLiveRegUses(
    const MachineInstr &MI, const GCNRPTracker::LiveRegSet &LiveRegs,
    const SmallSet<unsigned, 4> &Defs,
    DenseMap<const MachineOperand *, LiveRegUse> &RelevantUses,
    LiveRegUse &Furthest, LiveRegUse *FurthestSubreg) {

  SmallVector<const MachineOperand *> Uses;
  SmallVector<double> Distances;
  std::map<LaneBitmask, SmallVector<LiveRegUse>> UsesByMask;

  for (auto &KV : LiveRegs) {
    const unsigned Reg = KV.first;
    const LaneBitmask LaneMask = KV.second;
    if (Defs.contains(Reg))
      continue;

    Uses.clear();
    UsesByMask.clear();

    this->getUses(Reg, LaneMask, MI, Uses);
    if (Uses.empty())
      continue;

    const MachineOperand *NextUse = nullptr;
    std::optional<double> Dist;
    Dist =
        this->getNextUseDistance(Reg, LaneMask, MI, Uses, &Distances, &NextUse);
    if (!Dist.has_value())
      continue;

    LiveRegUse U{NextUse, Dist.value()};
    RelevantUses.try_emplace(NextUse, U);
    if (Furthest < U)
      Furthest = U;

    //--------------------------------------------------------------------------
    // Determine furthest sub-register if requested
    //--------------------------------------------------------------------------
    if (!FurthestSubreg)
      return;

    assert(Uses.size() == Distances.size());
    for (size_t I = 0; I < Uses.size(); ++I) {
      const MachineOperand *MO = Uses[I];

      if (!MO->getSubReg())
        continue;
      LaneBitmask Mask = TRI->getSubRegIndexLaneMask(MO->getSubReg());
      if (Mask.all() || Mask == LaneMask)
        continue;

      // FIXME: Integrate loop over UsesByMask here.
      UsesByMask[Mask].push_back({MO, Distances[I]});
    }

    if (UsesByMask.empty()) {
      if (*FurthestSubreg < U)
        *FurthestSubreg = U;
      continue;
    }

    for (auto &KV : UsesByMask) {
      SmallVector<LiveRegUse> &SubregUses = KV.second;
      LiveRegUse SubregU;
      for (LiveRegUse &LRU : SubregUses) {
        if (!SubregU.Use || LRU < SubregU)
          SubregU = LRU;
      }

      RelevantUses.try_emplace(SubregU.Use, SubregU);
      if (*FurthestSubreg < SubregU) {
        *FurthestSubreg = SubregU;
      }
    }
  }
}

void AMDGPUNextUseAnalysisImpl::printInstructionHeader(
    raw_ostream &OS, const MachineInstr &MI, ModuleSlotTracker &MST) const {
  InstructionInfo Info = parseInstructionString(MI, MST);
  OS << "    {\n";
  OS << "      " << Quote("name") << ": " << Quote(Info.DefName) << ",\n";
  OS << "      " << Quote("type") << ": " << Quote(Info.DefType) << ",\n";
  OS << "      " << Quote("instr") << ": " << Quote(Info.Instr) << ",\n";
  if (DumpNextUseDistanceVerbose) {
    OS << "      " << Quote("id") << ": " << Fmt(getInstrId(&MI)) << ",\n";
    OS << "      " << Quote("head-len") << ": " << Fmt(getHeadLen(&MI))
       << ",\n";
    OS << "      " << Quote("tail-len") << ": " << Fmt(getTailLen(&MI))
       << ",\n";
  }
}

void AMDGPUNextUseAnalysisImpl::printDistances(
    raw_ostream &OS,
    const DenseMap<const MachineOperand *, LiveRegUse> &Uses) const {
  OS << "      " << Quote("distances") << ": {\n";
  unsigned rem = Uses.size();
  for (auto &KV : Uses) {
    const bool FinalUse = --rem == 0;
    const LiveRegUse &U = KV.second;
    std::string RegStr =
        printRegToString(U.getReg(), U.getLaneMask(TRI), MRI, TRI);
    OS << "        ";
    OS << Quote(RegStr) << ": " << Fmt(U.Dist) << Sep(FinalUse) << "\n";
  }
  OS << "      },\n";
}

void AMDGPUNextUseAnalysisImpl::printFurthestUse(raw_ostream &OS,
                                                 const LiveRegUse &Furthest,
                                                 bool Subreg, bool Last) const {
  OS << "      " << Quote(Subreg ? "furthest-subreg" : "furthest") << ": {\n";
  if (Furthest.Use) {
    std::string RegStr = printRegToString(
        Furthest.getReg(),
        Subreg ? Furthest.getLaneMask(TRI) : LaneBitmask::getAll(), MRI, TRI);
    OS << "        " << Quote("register") << ": " << Quote(RegStr) << ",\n";
    if (DumpNextUseDistanceVerbose) {
      std::string UseStr = printToString(Furthest.Use);
      std::string UseMIStr = printToString(Furthest.Use->getParent());
      OS << "        " << Quote("use") << ": " << Quote(UseStr) << ",\n";
      OS << "        " << Quote("use-mi") << ": " << Quote(UseMIStr) << ",\n";
    }
    OS << "        " << Quote("distance") << ": " << Fmt(Furthest.Dist) << "\n";
  }
  OS << "      }" << (Last ? "\n" : ",\n");
}

void AMDGPUNextUseAnalysisImpl::printFurthestDistancesAsJson(
    raw_ostream &OS, const LiveIntervals *LIS) {
  const Function *F = &MF->getFunction();
  const Module *M = F->getParent();

  GCNDownwardRPTracker RPTracker(*LIS);
  ModuleSlotTracker MST(M);
  MST.incorporateFunction(*F);

  SmallSet<unsigned, 4> Defs;
  DenseMap<const MachineOperand *, LiveRegUse> RelevantUses;

  OS << "{\n";
  for (const MachineBasicBlock &MBB : *MF) {
    const bool FinalMBB = &MBB == &MF->back();
    std::string MBBName = nameForMBB(MBB, MST);

    OS << "  " << Quote(MBBName) << ": [\n";
    const MachineInstr *PrevMI = nullptr;
    for (const MachineInstr &MI : MBB) {
      const bool FinalMI = &MI == &MBB.back();

      // Update register pressure tracker
      if (!PrevMI || PrevMI->getOpcode() == AMDGPU::PHI)
        RPTracker.reset(MI);
      RPTracker.advance();

      Defs.clear();
      collectDefinedRegisters(MI, Defs);

      LiveRegUse Furthest;
      LiveRegUse FurthestSubreg;
      RelevantUses.clear();
      computeLiveRegUses(MI, RPTracker.getLiveRegs(), Defs, RelevantUses,
                         Furthest, &FurthestSubreg);

      // Print instruction JSON
      printInstructionHeader(OS, MI, MST);
      printDistances(OS, RelevantUses);
      printFurthestUse(OS, Furthest);
      printFurthestUse(OS, FurthestSubreg, /*Subreg*/ true, /*Last*/ true);

      OS << "    }" << Sep(FinalMI) << "\n";
      PrevMI = &MI;
    }
    OS << "  ]" << Sep(FinalMBB) << "\n";
  }
  OS << "}";
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// AMDGPUNextUseAnalysis
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
void AMDGPUNextUseAnalysis::initialize(const MachineFunction *MF,
                                       const MachineLoopInfo *MLI,
                                       const MachineDominatorTree *DT) {
  Impl = std::make_unique<AMDGPUNextUseAnalysisImpl>();
  Impl->initialize(MF, MLI, DT);
}

AMDGPUNextUseAnalysis::CompatibilityMode
AMDGPUNextUseAnalysis::getCompatibilityMode() {
  return Impl->getCompatibilityMode();
}

void AMDGPUNextUseAnalysis::setCompatibilityMode(CompatibilityMode M) {
  Impl->setCompatibilityMode(M);
}

/// \Returns the next-use distance for \p DefReg.
std::optional<double> AMDGPUNextUseAnalysis::getNextUseDistance(
    Register LiveReg, const MachineInstr &FromMI,
    const SmallVector<const MachineOperand *> &Uses,
    SmallVector<double> *Distances, const MachineOperand **UseOut) {
  return Impl->getNextUseDistance(LiveReg, FromMI, Uses, Distances, UseOut);
}

void AMDGPUNextUseAnalysis::getUses(unsigned Register, LaneBitmask LaneMask,
                                    const MachineInstr &MI,
                                    SmallVector<const MachineOperand *> &Uses) {
  return Impl->getUses(Register, LaneMask, MI, Uses);
}

void AMDGPUNextUseAnalysis::printFurthestDistancesAsJson(
    raw_ostream &OS, const LiveIntervals *LIS) {
  Impl->printFurthestDistancesAsJson(OS, LIS);
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// AMDGPUNextUseAnalysisPass
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
bool AMDGPUNextUseAnalysisPass::runOnMachineFunction(MachineFunction &MF) {
  const MachineLoopInfo *MLI =
      &getAnalysis<MachineLoopInfoWrapperPass>().getLI();
  const MachineDominatorTree *DT =
      &getAnalysis<MachineDominatorTreeWrapperPass>().getDomTree();

  NUA = std::make_unique<AMDGPUNextUseAnalysis>();
  NUA->initialize(&MF, MLI, DT);

  if (DumpNextUseDistanceAsJson.getNumOccurrences()) {
    const LiveIntervals *LIS =
        &getAnalysis<LiveIntervalsWrapperPass>().getLIS();
    std::string FN = DumpNextUseDistanceAsJson;
    if (FN.empty() || FN == "-") {
      NUA->printFurthestDistancesAsJson(outs(), LIS);
    } else {
      std::error_code EC;
      llvm::ToolOutputFile OutF(FN, EC, llvm::sys::fs::OF_None);
      NUA->printFurthestDistancesAsJson(OutF.os(), LIS);
      OutF.keep();
    }
  }

  return true;
}

void AMDGPUNextUseAnalysisPass::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<LiveVariablesWrapperPass>();
  AU.addRequired<MachineLoopInfoWrapperPass>();

  AU.addRequired<LiveIntervalsWrapperPass>();
  AU.addRequired<SlotIndexesWrapperPass>();
  AU.addRequired<MachineDominatorTreeWrapperPass>();

  AU.addPreserved<LiveVariablesWrapperPass>();
  AU.addPreserved<MachineLoopInfoWrapperPass>();

  MachineFunctionPass::getAnalysisUsage(AU);
}

char AMDGPUNextUseAnalysisPass::ID = 0;

INITIALIZE_PASS_BEGIN(AMDGPUNextUseAnalysisPass, DEBUG_TYPE,
                      "Next Use Analysis", false, false)
INITIALIZE_PASS_DEPENDENCY(LiveVariablesWrapperPass)
INITIALIZE_PASS_DEPENDENCY(MachineLoopInfoWrapperPass)

INITIALIZE_PASS_DEPENDENCY(LiveIntervalsWrapperPass)
INITIALIZE_PASS_DEPENDENCY(SlotIndexesWrapperPass)
INITIALIZE_PASS_DEPENDENCY(MachineDominatorTreeWrapperPass)

INITIALIZE_PASS_END(AMDGPUNextUseAnalysisPass, DEBUG_TYPE, "Next Use Analysis",
                    false, false)

char &llvm::AMDGPUNextUseAnalysisID = AMDGPUNextUseAnalysisPass::ID;

FunctionPass *llvm::createAMDGPUNextUseAnalysisPass() {
  return new AMDGPUNextUseAnalysisPass();
}
