//===--- ModulesDriver.cpp - Driver managed module builds -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file defines functionality to support driver managed builds for
/// compilations which use Clang modules or standard C++20 named modules.
///
//===----------------------------------------------------------------------===//

#include "clang/Driver/ModulesDriver.h"
#include "clang/DependencyScanning/DependencyScanningUtils.h"
#include "clang/DependencyScanning/ModuleDepCollector.h"
#include "clang/Driver/Compilation.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/Job.h"
#include "clang/Driver/Tool.h"
#include "clang/Frontend/StandaloneDiagnostic.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/DirectedGraph.h"
#include "llvm/ADT/PointerUnion.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/GraphWriter.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/ThreadPool.h"
#include "llvm/Support/VirtualFileSystem.h"

namespace deps = clang::dependencies;

using namespace llvm::opt;
using namespace clang;
using namespace driver;
using namespace modules;

namespace clang::driver::modules {

static bool fromJSON(const llvm::json::Value &Params,
                     StdModuleManifest::Module::LocalArguments &LocalArgs,
                     llvm::json::Path P) {
  llvm::json::ObjectMapper O(Params, P);
  return O.mapOptional("system-include-directories",
                       LocalArgs.SystemIncludeDirs);
}

static bool fromJSON(const llvm::json::Value &Params,
                     StdModuleManifest::Module &ModuleEntry,
                     llvm::json::Path P) {
  llvm::json::ObjectMapper O(Params, P);
  return O.map("is-std-library", ModuleEntry.IsStdlib) &&
         O.map("logical-name", ModuleEntry.LogicalName) &&
         O.map("source-path", ModuleEntry.SourcePath) &&
         O.mapOptional("local-arguments", ModuleEntry.LocalArgs);
}

static bool fromJSON(const llvm::json::Value &Params,
                     StdModuleManifest &Manifest, llvm::json::Path P) {
  llvm::json::ObjectMapper O(Params, P);
  return O.map("modules", Manifest.Modules);
}

} // namespace clang::driver::modules

/// Parses the Standard library module manifest from \p Buffer.
///
/// The source file paths listed in the manifest are relative to its own
/// path.
static Expected<StdModuleManifest> parseManifest(StringRef Buffer) {
  auto ParsedOrErr = llvm::json::parse(Buffer);
  if (!ParsedOrErr)
    return ParsedOrErr.takeError();

  StdModuleManifest Manifest;
  llvm::json::Path::Root Root;
  if (!fromJSON(*ParsedOrErr, Manifest, Root))
    return Root.getError();

  return Manifest;
}

/// Converts each file path in \p ManifestEntries from paths relative to
/// \p ManifestPath (the manifest's location itself) to absolute.
static void makeManifestPathsAbsolute(
    MutableArrayRef<StdModuleManifest::Module> ManifestEntries,
    StringRef ManifestPath) {
  assert(llvm::sys::path::is_absolute(ManifestPath) &&
         "ManifestPath expected to be absolute!");
  StringRef ManifestDir = llvm::sys::path::parent_path(ManifestPath);

  SmallString<256> TempPath;
  auto PrependManifestDir = [&](StringRef Path) -> std::string {
    assert(llvm::sys::path::is_relative(Path) &&
           "All paths in the manifest are expected to be relative to the "
           "manifest's location.");
    TempPath = ManifestDir;
    llvm::sys::path::append(TempPath, Path);
    return std::string(TempPath);
  };

  for (auto &Entry : ManifestEntries) {
    Entry.SourcePath = PrependManifestDir(Entry.SourcePath);
    if (!Entry.LocalArgs)
      continue;
    for (auto &IncludeDir : Entry.LocalArgs->SystemIncludeDirs)
      IncludeDir = PrependManifestDir(IncludeDir);
  }
}

Expected<StdModuleManifest>
driver::modules::readStdModuleManifest(StringRef ManifestPath,
                                       llvm::vfs::FileSystem &VFS) {
  auto MemBufOrErr = VFS.getBufferForFile(ManifestPath);
  if (!MemBufOrErr)
    return llvm::createFileError(ManifestPath, MemBufOrErr.getError());
  const auto MemBuf = std::move(*MemBufOrErr);

  auto ManifestOrErr = parseManifest(MemBuf->getBuffer());
  if (!ManifestOrErr)
    return ManifestOrErr.takeError();
  auto Manifest = std::move(*ManifestOrErr);

  makeManifestPathsAbsolute(Manifest.Modules, ManifestPath);
  return Manifest;
}

void driver::modules::buildStdModuleManifestInputs(
    ArrayRef<StdModuleManifest::Module> ManifestEntries, Compilation &C,
    InputList &Inputs) {
  auto &Args = C.getArgs();
  const auto &Opts = C.getDriver().getOpts();
  for (const auto &Entry : ManifestEntries) {
    C.getDriver().DiagnoseInputExistence(Args, Entry.SourcePath,
                                         types::TY_CXXModule,
                                         /*TypoCorrect=*/false);
    auto *InputArg = makeInputArg(Args, Opts, Entry.SourcePath);
    Inputs.emplace_back(types::TY_CXXModule, InputArg);
  }
}

/// Computes the -fmodule-cache-path for this compilation.
static std::optional<SmallString<128>>
getModuleCachePath(DerivedArgList &Args) {
  if (const auto &A = Args.getLastArg(options::OPT_fmodules_cache_path))
    return SmallString<128>(A->getValue());
  if (SmallString<128> Path; driver::Driver::getDefaultModuleCachePath(Path))
    return Path;
  return std::nullopt;
}

using ManifestLookupMap =
    llvm::DenseMap<StringRef, const StdModuleManifest::Module *>;

static ManifestLookupMap
createManifestLookupMap(ArrayRef<StdModuleManifest::Module> ManifestEntries) {
  ManifestLookupMap Out;
  for (auto &ManifestEntry : ManifestEntries) {
    const bool Inserted =
        Out.try_emplace(ManifestEntry.SourcePath, &ManifestEntry).second;
    // TODO: Keep a stronger association between manifest inputs and the jobs
    // they generate to handle such cases.
    assert(Inserted && "Unable to handle multiple manifest entries referring "
                       "to the same module path");
  }
  return Out;
}

/// Returns the manifest entry corresponding to \p Job, or \c nullptr if none
/// exists.
static const StdModuleManifest::Module *
getManifestEntryForJob(const Command &Job,
                       const ManifestLookupMap &ManifestLookup) {
  for (const auto &II : Job.getInputInfos()) {
    if (!II.isFilename())
      continue;
    if (const auto It = ManifestLookup.find(II.getFilename());
        It != ManifestLookup.end())
      return It->second;
  }
  return nullptr;
}

/// Adds all \p SystemIncludeDirs to \p Job's arguments.
static void
addSystemIncludeDirsFromManifest(Compilation &C, Command &Job,
                                 ArrayRef<std::string> SystemIncludeDirs) {
  const auto &TC = Job.getCreator().getToolChain();
  const auto &TCArgs = C.getArgsForToolChain(
      &TC, /*BoundArch*/ "", Job.getSource().getOffloadingDeviceKind());

  auto NewArgs = Job.getArguments();
  for (const auto &IncludeDir : SystemIncludeDirs)
    TC.addSystemInclude(TCArgs, NewArgs, IncludeDir);
  Job.replaceArguments(NewArgs);
}

namespace {

/// Context that associates scan inputs with their origin.
struct ScanInputContext {
  using StdlibScanInputMap =
      llvm::DenseMap<std::pair<StringRef, StringRef>, size_t>;

  /// Indices of scan inputs that are -cc1 jobs for user-provided inputs.
  SmallVector<size_t> UserInputIndices;

  /// Maps each (module name, triple) pair to the index of the -cc1 job
  /// providing that Standard library module.
  StdlibScanInputMap StdlibInputLookup;
};

/// The dependencies for a -cc1 job that is a dependency scan input
struct InputDependencies {
  /// The name of the C++20 module exported by this translation unit.
  std::string ModuleName;

  /// A list of modules this translation unit directly depends on, not including
  /// transitive dependencies.
  ///
  /// This may include modules with a different context hash when it can be
  /// determined that the differences are benign for this compilation.
  std::vector<deps::ModuleID> ClangModuleDeps;

  /// A list of the C++20 named modules this translation unit depends on.
  ///
  /// All C++20 named module dependencies are expected to target the same triple
  /// as this translation unit.
  std::vector<std::string> NamedModuleDeps;

  /// The compiler invocation with modifications to properly import all Clang
  /// module dependencies. Does not include argv[0].
  std::vector<std::string> BuildArgs;
};

/// Full dependencies for each dependency scan input and all discovered Clang
/// modules.
struct DependencyScanResults {
  /// Full dependencies for each dependency scan input, in input order.
  ///
  /// Entries corresponding to standard library modules which were not imported
  /// (and thus not scanned) are std::nullopt.
  llvm::SmallVector<std::optional<InputDependencies>, 0> InputDeps;

  /// The full Clang module dependencies for this compilation.
  SmallVector<deps::ModuleDeps, 0> ModuleDeps;
};

class CGNode;
class CGEdge;
using CGNodeBase = llvm::DGNode<CGNode, CGEdge>;
using CGEdgeBase = llvm::DGEdge<CGNode, CGEdge>;
using CGBase = llvm::DirectedGraph<CGNode, CGEdge>;

/// Compilation Graph Node
class CGNode : public CGNodeBase {
public:
  enum class NodeKind {
    CC1Job,
    ImageJob,
    MiscJob,
    Root,
  };

  CGNode() = delete;
  CGNode(const NodeKind K) : Kind(K) {}
  virtual ~CGNode() = 0;

  NodeKind getKind() const { return Kind; }

private:
  NodeKind Kind;
};
CGNode::~CGNode() = default;

/// Subclass of CGNode representing the root node of the graph.
///
/// The root node is a special node that connects to all other nodes with
/// no incoming edges, so that there is always a path from it to any node
/// in the graph.
///
/// There should only be one such node in a given graph.
class RootNode : public CGNode {
public:
  RootNode() : CGNode(NodeKind::Root) {}
  ~RootNode() override = default;

  /// Define classof to be able to use isa<>, cast<>, dyn_cast<>, etc.
  static bool classof(const CGNode *N) {
    return N->getKind() == NodeKind::Root;
  }
};

/// Base class for any CGNode type that represents a single job.
class JobNode : public CGNode {
public:
  JobNode() = delete;
  JobNode(std::unique_ptr<Command> &&Job, NodeKind Kind)
      : CGNode(Kind), Job(std::move(Job)) {}
  virtual ~JobNode() override = 0;

  /// Define classof to be able to use isa<>, cast<>, dyn_cast<>, etc.
  static bool classof(const CGNode *N) {
    return N->getKind() != NodeKind::Root;
  }

  std::unique_ptr<Command> Job;
};
JobNode::~JobNode() = default;

/// Subclass of CGNode representing a -cc1 job.
class CC1JobNode : public JobNode {
public:
  CC1JobNode() = delete;
  CC1JobNode(std::unique_ptr<Command> &&Job)
      : JobNode(std::move(Job), NodeKind::CC1Job) {}
  ~CC1JobNode() override = default;

  llvm::PointerUnion<const InputDependencies *, const deps::ModuleDeps *>
      DependencyInfo;

  /// Returns true if a dependency scan was performed for this -cc1 job, and
  /// false otherwise.
  bool isScanned() const { return !DependencyInfo.isNull(); }

  /// Returns true if -cc1 job does not produce any module, and false otherwise.
  bool isNonModule() const {
    if (const auto *InputDeps =
            DependencyInfo.dyn_cast<const InputDependencies *>())
      return InputDeps->ModuleName.empty();
    return false;
  }

  /// Returns true if this -cc1 job produces a C++20 named module, and false
  /// otherwise.
  bool isCXXNamedModule() const {
    if (const auto *InputDeps =
            DependencyInfo.dyn_cast<const InputDependencies *>())
      return !InputDeps->ModuleName.empty();
    return false;
  }

  /// Return true if this -cc1 job produces a Clang module, and false otherwise.
  bool isClangModule() const {
    return isa<const deps::ModuleDeps *>(DependencyInfo);
  }

  /// Define classof to be able to use isa<>, cast<>, dyn_cast<>, etc.
  static bool classof(const CGNode *N) {
    return N->getKind() == NodeKind::CC1Job;
  }
};

/// Subclass of CGNode representing a job which produces an image file, such as
/// a linker or interface stub merge job.
class ImageJobNode : public JobNode {
public:
  ImageJobNode() = delete;
  ImageJobNode(std::unique_ptr<Command> &&Job)
      : JobNode(std::move(Job), NodeKind::ImageJob) {}
  ~ImageJobNode() override = default;

  /// Define classof to be able to use isa<>, cast<>, dyn_cast<>, etc.
  static bool classof(const CGNode *N) {
    return N->getKind() == NodeKind::ImageJob;
  }
};

/// Subclass of CGNode representing any job not covered by the other node types.
///
/// Jobs represented by this node type are not modified by the modules driver.
class MiscJobNode : public JobNode {
public:
  MiscJobNode() = delete;
  MiscJobNode(std::unique_ptr<Command> &&Job)
      : JobNode(std::move(Job), NodeKind::MiscJob) {}
  ~MiscJobNode() override = default;

  /// Define classof to be able to use isa<>, cast<>, dyn_cast<>, etc.
  static bool classof(const CGNode *N) {
    return N->getKind() == NodeKind::MiscJob;
  }
};

/// Compilation Graph Edge
///
/// Edges connect the producer of an output to its consumer, except for edges
/// stemming from the root node.
class CGEdge : public CGEdgeBase {
public:
  enum class EdgeKind {
    Regular,
    ModuleDependency,
    Rooted,
  };

  CGEdge(CGNode &N, EdgeKind K) : CGEdgeBase(N), Kind(K) {}

  EdgeKind getKind() const { return Kind; };

  bool isRegular() const { return Kind == EdgeKind::Regular; }
  bool isModuleDependency() const { return Kind == EdgeKind::ModuleDependency; }
  bool isRooted() const { return Kind == EdgeKind::Rooted; }

private:
  EdgeKind Kind;
};

/// Compilation Graph
///
/// The graph owns all of its components.
class CompilationGraph : public CGBase {
public:
  CompilationGraph() = default;
  CompilationGraph(const CompilationGraph &) = delete;
  CompilationGraph(CompilationGraph &&G) : CGBase(std::move(G)) {}
  ~CompilationGraph() {
    for (auto *N : *this) {
      for (auto *E : N->getEdges())
        delete E;
      delete N;
    }
  }

  // Equivalent to llvm::DirectedGraph::removeNode, but also deletes the node
  // and all incoming and outgoing edges.
  bool removeAndDeleteNode(CGNode &N);

  void setRoot(RootNode *Root) {
    assert(!this->Root && "The graph already has a root node!");
    this->Root = Root;
  }

  CGNode *getRoot() {
    assert(Root && "Root node has not been set yet!");
    return Root;
  }

  const CGNode *getRoot() const {
    assert(Root && "Root node has not been set yet!");
    return Root;
  }

  /// Transfers ownership of \p DependencyInfo into the graph.
  void storeDependencyInfo(DependencyScanResults &&DependencyInfo) {
    this->DependencyInfo = std::move(DependencyInfo);
  }

private:
  CGNode *Root = nullptr;
  DependencyScanResults DependencyInfo;
};

} // anonymous namespace

bool CompilationGraph::removeAndDeleteNode(CGNode &N) {
  iterator It = findNode(N);
  if (It == Nodes.end())
    return false;
  // Remove incoming edges.
  EdgeListTy EL;
  for (auto *Node : Nodes) {
    if (*Node == N)
      continue;
    Node->findEdgesTo(N, EL);
    for (auto *E : EL) {
      Node->removeEdge(*E);
      delete E;
    }
    EL.clear();
  }
  // Remove the outgoing edges.
  for (auto *E : N) {
    N.removeEdge(*E);
    delete E;
  }
  N.clear();
  // Remove the node itself.
  auto *NodePtr = *It;
  Nodes.erase(It);
  delete NodePtr;
  return true;
}

static const std::string &getTriple(const Command &Job) {
  return Job.getCreator().getToolChain().getTriple().getTriple();
}

namespace llvm {

/// Non-const versions of the GraphTraits specializations for CompilationGraph.
template <> struct GraphTraits<CGNode *> {
  using NodeRef = CGNode *;

  static NodeRef CGGetTargetNode(CGEdgeBase *E) { return &E->getTargetNode(); }

  using ChildIteratorType =
      mapped_iterator<CGNode::iterator, decltype(&CGGetTargetNode)>;
  using ChildEdgeIteratorType = CGNode::iterator;

  static NodeRef getEntryNode(NodeRef N) { return N; }

  static ChildIteratorType child_begin(NodeRef N) {
    return ChildIteratorType(N->begin(), &CGGetTargetNode);
  }

  static ChildIteratorType child_end(NodeRef N) {
    return ChildIteratorType(N->end(), &CGGetTargetNode);
  }

  static ChildEdgeIteratorType child_edge_begin(NodeRef N) {
    return N->begin();
  }
  static ChildEdgeIteratorType child_edge_end(NodeRef N) { return N->end(); }
};

template <> struct GraphTraits<CompilationGraph *> : GraphTraits<CGNode *> {
  using GraphRef = CompilationGraph *;
  using NodeRef = CGNode *;

  using nodes_iterator = CompilationGraph::iterator;

  static NodeRef getEntryNode(GraphRef G) { return G->getRoot(); }

  static nodes_iterator nodes_begin(GraphRef G) { return G->begin(); }

  static nodes_iterator nodes_end(GraphRef G) { return G->end(); }
};

/// Const versions of the GraphTraits specializations for CompilationGraph.
template <> struct GraphTraits<const CGNode *> {
  using NodeRef = const CGNode *;

  static NodeRef CGGetTargetNode(const CGEdgeBase *E) {
    return &E->getTargetNode();
  }

  using ChildIteratorType =
      mapped_iterator<CGNode::const_iterator, decltype(&CGGetTargetNode)>;
  using ChildEdgeIteratorType = CGNode::const_iterator;

  static NodeRef getEntryNode(NodeRef N) { return N; }

  static ChildIteratorType child_begin(NodeRef N) {
    return ChildIteratorType(N->begin(), &CGGetTargetNode);
  }

  static ChildIteratorType child_end(NodeRef N) {
    return ChildIteratorType(N->end(), &CGGetTargetNode);
  }

  static ChildEdgeIteratorType child_edge_begin(NodeRef N) {
    return N->begin();
  }

  static ChildEdgeIteratorType child_edge_end(NodeRef N) { return N->end(); }
};

template <>
struct GraphTraits<const CompilationGraph *> : GraphTraits<const CGNode *> {
  using GraphRef = const CompilationGraph *;
  using NodeRef = const CGNode *;

  using nodes_iterator = CompilationGraph::const_iterator;

  static NodeRef getEntryNode(GraphRef G) { return G->getRoot(); }

  static nodes_iterator nodes_begin(GraphRef G) { return G->begin(); }

  static nodes_iterator nodes_end(GraphRef G) { return G->end(); }
};

template <>
struct DOTGraphTraits<const CompilationGraph *> : DefaultDOTGraphTraits {
  explicit DOTGraphTraits(bool IsSimple = false)
      : DefaultDOTGraphTraits(IsSimple) {}

  static std::string getGraphName(const CompilationGraph *) {
    return "Module Dependency Graph";
  }

  static std::string getGraphProperties(const CompilationGraph *) {
    return "\tnode [shape=Mrecord, colorscheme=set23, "
           "style=filled];\n\trankdir=BT;\n";
  }

  static bool isNodeHidden(const CGNode *N, const CompilationGraph *) {
    return !isa<CC1JobNode>(N) ||
           (isa<CC1JobNode>(N) && !cast<CC1JobNode>(N)->isScanned());
  }

  static std::string getNodeIdentifier(const CGNode *N,
                                       const CompilationGraph *) {
    auto *CC1Node = cast<CC1JobNode>(N);
    if (CC1Node->isClangModule()) {
      const auto &ModuleDeps =
          *cast<const deps::ModuleDeps *>(CC1Node->DependencyInfo);
      return (Twine(ModuleDeps.ID.ModuleName) + "-" + ModuleDeps.ID.ContextHash)
          .str();
    }
    StringRef Triple = getTriple(*CC1Node->Job);
    if (CC1Node->isCXXNamedModule()) {
      const auto &InputDeps =
          *cast<const InputDependencies *>(CC1Node->DependencyInfo);
      return (Twine(InputDeps.ModuleName) + "-" + Triple).str();
    }
    // Scanned non-module jobs always have the source file as their first input.
    assert(CC1Node->isNonModule() && "Expected scanned non-module job!");
    StringRef Filename = CC1Node->Job->getInputInfos().front().getFilename();
    return (Twine(Filename) + "-" + Triple).str();
  }

  static std::string getNodeLabel(const CGNode *N, const CompilationGraph *) {
    auto *CC1Node = cast<CC1JobNode>(N);
    if (CC1Node->isClangModule()) {
      const auto &ModuleDeps =
          *cast<const deps::ModuleDeps *>(CC1Node->DependencyInfo);
      return (Twine("Module type: Clang module \\| Module name: ") +
              ModuleDeps.ID.ModuleName +
              " \\| Hash: " + ModuleDeps.ID.ContextHash)
          .str();
    }
    StringRef Triple = getTriple(*CC1Node->Job);
    if (CC1Node->isCXXNamedModule()) {
      const auto &InputDeps =
          *cast<const InputDependencies *>(CC1Node->DependencyInfo);
      return (Twine("Module type: Named module \\| Module name: ") +
              InputDeps.ModuleName + " \\| Triple: " + Triple)
          .str();
    }
    // Scanned non-module jobs always have the source file as their first input.
    assert(CC1Node->isNonModule() && "Expected scanned non-module job!");
    StringRef Filename = CC1Node->Job->getInputInfos().front().getFilename();
    return (Twine("Filename: ") + Filename + " \\| Triple: " + Triple).str();
  }

  static std::string getNodeAttributes(const CGNode *N,
                                       const CompilationGraph *) {
    auto *CC1Node = cast<CC1JobNode>(N);
    if (CC1Node->isClangModule())
      return "fillcolor=1";
    if (CC1Node->isCXXNamedModule())
      return "fillcolor=2";
    assert(CC1Node->isNonModule() && "Expected scanned non-module job!");
    return "fillcolor=3";
  }
};

/// GraphWriter specialization for CompilationGraph that emits a more
/// human-readable DOT graph.
template <>
class GraphWriter<const CompilationGraph *>
    : public GraphWriterBase<const CompilationGraph *,
                             GraphWriter<const CompilationGraph *>> {
public:
  using GraphType = const CompilationGraph *;
  using Base = GraphWriterBase<GraphType, GraphWriter<GraphType>>;

  GraphWriter(llvm::raw_ostream &O, const GraphType &G, bool IsSimple)
      : Base(O, G, IsSimple) {}

  void writeNodes();

private:
  using Base::DOTTraits;
  using Base::GTraits;
  using Base::NodeRef;

  DenseMap<NodeRef, std::string> NodeIDMap;

  void writeNodeDefinitions(ArrayRef<NodeRef> Nodes);
  void writeNodeRelations(ArrayRef<NodeRef> Nodes);
};

void GraphWriter<const CompilationGraph *>::writeNodes() {
  auto IsNodeVisible = [&](NodeRef N) { return !DTraits.isNodeHidden(N, G); };
  const auto VisibleNodes = llvm::filter_to_vector(nodes(G), IsNodeVisible);

  writeNodeDefinitions(VisibleNodes);
  writeNodeRelations(VisibleNodes);
}

void GraphWriter<const CompilationGraph *>::writeNodeDefinitions(
    ArrayRef<NodeRef> Nodes) {
  for (const auto &Node : Nodes) {
    const auto NodeID = DTraits.getNodeIdentifier(Node, G);
    const auto NodeLabel = DTraits.getNodeLabel(Node, G);
    O << "\t\"" << DOT::EscapeString(NodeID) << "\" [ "
      << DTraits.getNodeAttributes(Node, G) << ", label=\"{ "
      << DOT::EscapeString(NodeLabel) << " }\"];\n";
    NodeIDMap.try_emplace(Node, std::move(NodeID));
  }
  O << "\n";
}

void llvm::GraphWriter<const CompilationGraph *>::writeNodeRelations(
    ArrayRef<NodeRef> Nodes) {
  for (const auto *Node : Nodes) {
    const auto &SourceNodeID = NodeIDMap.at(Node);
    for (const auto *Edge : Node->getEdges()) {
      if (!Edge->isModuleDependency())
        continue;
      const auto *TargetNode = GTraits::CGGetTargetNode(Edge);
      const auto &TargetNodeID = NodeIDMap.at(TargetNode);
      O << "\t\"" << DOT::EscapeString(SourceNodeID) << "\" -> \""
        << DOT::EscapeString(TargetNodeID) << "\";\n";
    }
  }
}

} // namespace llvm

static bool isCC1Job(const Command &Job) {
  return StringRef(Job.getCreator().getName()) == "clang";
}

static JobNode *createJobNode(std::unique_ptr<Command> &&Job) {
  if (isCC1Job(*Job))
    return new CC1JobNode(std::move(Job));
  if (Job->getCreator().isLinkJob())
    return new ImageJobNode(std::move(Job));
  return new MiscJobNode(std::move(Job));
}

/// Builds the compilation graph from the list of \p Jobs produced by the
/// driver.
static CompilationGraph
createGraphFromJobs(SmallVectorImpl<std::unique_ptr<Command>> &&Jobs) {
  CompilationGraph Graph;

  llvm::DenseMap<StringRef, CGNode *> OutputToProducerNode;
  for (auto &OwnedJob : Jobs) {
    auto *NewNode = createJobNode(std::move(OwnedJob));
    Graph.addNode(*NewNode);
    const auto &Job = *NewNode->Job;

    for (const auto &II : Job.getInputInfos()) {
      if (!II.isFilename())
        continue;

      const auto It = OutputToProducerNode.find(II.getFilename());
      if (It == OutputToProducerNode.end())
        continue;

      CGNode *FromNode = It->getSecond();
      CGEdge *E = new CGEdge(*NewNode, CGEdge::EdgeKind::Regular);
      Graph.connect(*FromNode, *NewNode, *E);
    }

    for (const auto &Output : Job.getOutputFilenames()) {
      const bool Inserted =
          OutputToProducerNode.try_emplace(Output, NewNode).second;
      assert(Inserted &&
             "Driver should not produce multiple jobs with identical outputs!");
    }
  }

  return Graph;
}

/// Creates and adds nodes for each Clang module in by \p ClangModuleDeps.
///
/// TODO: Generate and pass in the corresponding jobs instead of \p
/// ClangModuleDepsCount.
///
/// \returns the list of newly created nodes.
static SmallVector<CC1JobNode *>
createClangModuleNodes(CompilationGraph &Graph, size_t ClangModuleDepsCount) {
  SmallVector<CC1JobNode *> NewNodes(ClangModuleDepsCount);
  for (size_t I = 0; I != ClangModuleDepsCount; ++I) {
    auto *NewNode = new CC1JobNode(nullptr);
    Graph.addNode(*NewNode);
    NewNodes[I] = NewNode;
  }
  return NewNodes;
}

/// Returns true if the -cc1 job \p Job is eligible as a dependency scan input.
///
/// A job is eligible if it is a clang -cc1 job and its first input is a source
/// file.
static bool isEligibleScanInput(const Command &CC1Job) {
  assert(isCC1Job(CC1Job) && "Input job must be -cc1!");
  const auto &InputInfos = CC1Job.getInputInfos();
  return !InputInfos.empty() && types::isSrcFile(InputInfos.front().getType());
}

// Prunes jobs generated for standard library modules that weren't imported by
// any user inputs (i.e., each job corresponding to a std::nullopt in \p
// InputDeps).
static void pruneUnimportedStdlibModuleJobs(
    CompilationGraph &Graph, MutableArrayRef<CC1JobNode *> ScanInputNodes,
    ArrayRef<std::optional<InputDependencies>> InputDeps) {
  SmallVector<CGNode *> UnusedStdlibModuleJobs;
  for (auto &&[Node, Deps] : llvm::zip_equal(ScanInputNodes, InputDeps))
    if (!Deps)
      UnusedStdlibModuleJobs.push_back(Node);

  // Collect any dependent jobs we also have to delete
  llvm::DenseMap<ImageJobNode *, SmallVector<CGNode *>> ImageToDeadJobs;
  for (auto *DeadRoot : UnusedStdlibModuleJobs) {
    llvm::df_iterator_default_set<CGNode *, 4> Visited;
    for (auto *Node : llvm::depth_first_ext(DeadRoot, Visited))
      if (auto *ImageNode = dyn_cast<ImageJobNode>(Node))
        llvm::append_range(ImageToDeadJobs[ImageNode], Visited);
  }

  // Remove outputs of jobs we are deleting from any downstream image-producing
  // jobs before deletion. Image-producing jobs are never deleted, since at
  // least one job originating from a user-provided input must feed into them.
  for (auto &[ImageNode, DeadJobNodes] : ImageToDeadJobs) {
    SmallVector<StringRef, 2> OutputsToRemove;
    for (auto *DeadNode : DeadJobNodes)
      llvm::append_range(OutputsToRemove,
                         cast<JobNode>(DeadNode)->Job->getOutputFilenames());

    auto NewArgs = ImageNode->Job->getArguments();
    llvm::erase_if(NewArgs, [&](StringRef Arg) {
      return llvm::is_contained(OutputsToRemove, Arg);
    });
    ImageNode->Job->replaceArguments(NewArgs);

    for (auto *Job : DeadJobNodes) {
      Graph.removeAndDeleteNode(*Job);
    }
  }

  // Remove the dangling pointers.
  for (auto &&[Node, Deps] : llvm::zip_equal(ScanInputNodes, InputDeps))
    if (!Deps)
      Node = nullptr;
}

/// Adds a module-dependency edge from the dependency node in \p Lookup keyed by
/// \p Key to \p DependentNode, if any.
template <typename MapT, typename KeyT>
static void connectModuleDependencyEdges(CompilationGraph &Graph,
                                         CGNode &DependentNode,
                                         const MapT &Lookup, const KeyT &Key) {
  const auto It = Lookup.find(Key);
  // Missing dependencies are diagnosed during compile-job execution.
  if (It == Lookup.end())
    return;

  auto *E = new CGEdge(DependentNode, CGEdge::EdgeKind::ModuleDependency);
  Graph.connect(*It->second, DependentNode, *E);
}

/// Applies the dependency scan results to the compilation graph.
///
/// \returns true on success if no duplicate modules are detected, and false
/// otherwise (with diagnostics reported to \p Diags).
static bool addModuleDependencyInfo(
    CompilationGraph &Graph, MutableArrayRef<CC1JobNode *> ScanInputNodes,
    MutableArrayRef<CC1JobNode *> ClangModuleNodes,
    DependencyScanResults &&ScanResults, DiagnosticsEngine &Diags) {
  assert(all_of(Graph, llvm::IsaPred<JobNode>) &&
         "Expected only job nodes at this stage of building the graph!");
  llvm::DenseMap<deps::ModuleID, CC1JobNode *> ClangModuleLookup;
  llvm::DenseMap<std::pair<StringRef, StringRef>, CC1JobNode *>
      NamedModuleLookup;

  for (auto &&[ClangModuleNode, ModuleDeps] :
       llvm::zip_equal(ClangModuleNodes, ScanResults.ModuleDeps)) {
    ClangModuleNode->DependencyInfo = &ModuleDeps;
    const bool Inserted =
        ClangModuleLookup.try_emplace(ModuleDeps.ID, ClangModuleNode).second;
    assert(Inserted &&
           "Expected ScanResults to only contain unique ModuleDeps!");
  }

  bool HasDuplicateModuleError = false;
  for (auto &&[ScanInputNode, InputDeps] :
       llvm::zip_equal(ScanInputNodes, ScanResults.InputDeps)) {
    // Skip previously deleted nodes.
    if (!InputDeps)
      continue;

    ScanInputNode->DependencyInfo = &*InputDeps;
    if (ScanInputNode->isNonModule())
      continue;

    StringRef Triple = getTriple(*ScanInputNode->Job);
    const auto [It, Inserted] = NamedModuleLookup.try_emplace(
        {InputDeps->ModuleName, Triple}, ScanInputNode);
    if (!Inserted) {
      Diags.Report(diag::err_modules_driver_named_module_redefinition)
          << InputDeps->ModuleName
          // For scan input jobs, their first input is always a filename and
          // the scanned source.
          << It->second->Job->getInputInfos().front().getFilename()
          << ScanInputNode->Job->getInputInfos().front().getFilename();
      HasDuplicateModuleError = true;
    }
  }
  if (HasDuplicateModuleError)
    return false;

  // Connect the dependencies of all Clang module nodes.
  for (auto &&[DependentNode, ModuleDeps] :
       llvm::zip_equal(ClangModuleNodes, ScanResults.ModuleDeps)) {
    for (const deps::ModuleID &DepID : ModuleDeps.ClangModuleDeps)
      connectModuleDependencyEdges(Graph, *DependentNode, ClangModuleLookup,
                                   DepID);
  }

  // Connect the dependencies of all non-module / C++20 named module nodes.
  for (auto &&[DependentNode, InputDeps] :
       llvm::zip_equal(ScanInputNodes, ScanResults.InputDeps)) {
    // Skip previously deleted nodes.
    if (!InputDeps)
      continue;

    for (const deps::ModuleID &DepID : InputDeps->ClangModuleDeps)
      connectModuleDependencyEdges(Graph, *DependentNode, ClangModuleLookup,
                                   DepID);

    const auto &Triple = getTriple(*DependentNode->Job);
    for (const auto &DepName : InputDeps->NamedModuleDeps) {
      connectModuleDependencyEdges(Graph, *DependentNode, NamedModuleLookup,
                                   std::make_pair(DepName, Triple));
    }
  }

  // Pointers stay stable because the vector is moved into storage and never
  // resized thereafter.
  Graph.storeDependencyInfo(std::move(ScanResults));
  return true;
}

/// Creates the root node and connects it to all nodes with no incoming edges
/// so that every node in the graph is reachable from the root.
static void createAndConnectRootNode(CompilationGraph &Graph) {
  assert(all_of(Graph, llvm::IsaPred<JobNode>) &&
         "Expected only job nodes at this stage of building the graph!");
  llvm::SmallPtrSet<CGNode *, 16> HasIncomingEdge;
  for (auto *N : Graph)
    for (auto *E : N->getEdges())
      HasIncomingEdge.insert(&E->getTargetNode());

  auto NodesWithoutRoot = llvm::iterator_range(Graph);

  auto *Root = new RootNode();
  Graph.addNode(*Root);
  Graph.setRoot(Root);

  for (auto *N : NodesWithoutRoot) {
    if (HasIncomingEdge.contains(N))
      continue;
    auto *E = new CGEdge(*N, CGEdge::EdgeKind::Rooted);
    Graph.connect(*Root, *N, *E);
  }
}

namespace {

/// Pool of reusable dependency scanning workers and their contexts with
/// RAII-based acquire/release.
class ScanningWorkerPool {
public:
  ScanningWorkerPool(size_t NumWorkers, deps::DependencyScanningService &S,
                     llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> &BaseFS);

  /// Acquires a unique pointer to a dependency scanning worker and its
  /// context.
  ///
  /// The worker bundle automatically released back to the pool when the
  /// pointer is destroyed. The pool has to outlive the leased worker bundle.
  [[nodiscard]] auto scopedAcquire();

private:
  /// Releases the worker bundle at \c Index back into the pool.
  void release(size_t Index);

  /// A scanning worker with its associated context.
  struct WorkerBundle {
    WorkerBundle(deps::DependencyScanningService &S,
                 llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> &BaseFS)
        : Worker(std::make_unique<deps::DependencyScanningWorker>(S, BaseFS)) {}

    std::unique_ptr<deps::DependencyScanningWorker> Worker;
    llvm::DenseSet<deps::ModuleID> SeenModules;
  };

  std::mutex Lock;
  std::condition_variable CV;
  SmallVector<size_t> AvailableSlots;
  SmallVector<WorkerBundle, 0> Slots;
};

} // anonymous namespace

ScanningWorkerPool::ScanningWorkerPool(
    size_t NumWorkers, deps::DependencyScanningService &S,
    llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> &BaseFS) {
  for (size_t I = 0; I < NumWorkers; ++I)
    Slots.emplace_back(S, BaseFS);

  AvailableSlots.resize(NumWorkers);
  std::iota(AvailableSlots.begin(), AvailableSlots.end(), 0);
}

[[nodiscard]] auto ScanningWorkerPool::scopedAcquire() {
  std::unique_lock<std::mutex> UL(Lock);
  CV.wait(UL, [&] { return !AvailableSlots.empty(); });
  const size_t Index = AvailableSlots.pop_back_val();
  auto ReleaseHandle = [this, Index](WorkerBundle *) { release(Index); };
  return std::unique_ptr<WorkerBundle, decltype(ReleaseHandle)>(&Slots[Index],
                                                                ReleaseHandle);
}

void ScanningWorkerPool::release(size_t Index) {
  {
    std::scoped_lock<std::mutex> SL(Lock);
    AvailableSlots.push_back(Index);
  }
  CV.notify_one();
}

// Creates a ThreadPool and a corresponding ScanningWorkerPool optimized for
// the configuration of dependency scan inputs.
static std::pair<std::unique_ptr<llvm::ThreadPoolInterface>,
                 std::unique_ptr<ScanningWorkerPool>>
createOptimalThreadAndWorkerPool(
    size_t NumInputs, bool HasStdlibInputs,
    deps::DependencyScanningService &ScanningService,
    IntrusiveRefCntPtr<llvm::vfs::FileSystem> BaseFS) {
  // TODO: Benchmark: Determine the optimal number of worker threads for a
  // given number of inputs. How many inputs are required for multi-threading
  // to be beneficial? How many inputs should each thread scan at least?
#if LLVM_ENABLE_THREADS
  std::unique_ptr<llvm::ThreadPoolInterface> ThreadPool;
  size_t WorkerCount;

  if (NumInputs == 1 || (HasStdlibInputs && NumInputs <= 2)) {
    auto S = llvm::optimal_concurrency(1);
    ThreadPool = std::make_unique<llvm::SingleThreadExecutor>(std::move(S));
    WorkerCount = 1;
  } else {
    auto ThreadPoolStrategy = llvm::optimal_concurrency(
        NumInputs - static_cast<size_t>(HasStdlibInputs));
    ThreadPool = std::make_unique<llvm::DefaultThreadPool>(
        std::move(ThreadPoolStrategy));
    const size_t MaxConcurrency = ThreadPool->getMaxConcurrency();
    WorkerCount = std::min(
        MaxConcurrency,
        NumInputs - (HasStdlibInputs && NumInputs < MaxConcurrency ? 1 : 0));
  }
#else
  auto ThreadPool = std::make_unique<llvm::SingleThreadExecutor>();
  size_t WorkerCount = 1;
#endif

  return {std::move(ThreadPool), std::make_unique<ScanningWorkerPool>(
                                     WorkerCount, ScanningService, BaseFS)};
}

namespace {

using StandaloneDiagList = SmallVector<StandaloneDiagnostic, 0>;

/// Collects diagnostics in a form that can be retained until after their
/// associated SourceManager is destroyed.
class StandaloneDiagCollector : public DiagnosticConsumer {
public:
  void BeginSourceFile(const LangOptions &LangOpts,
                       const Preprocessor *PP = nullptr) override {
    this->LangOpts = &LangOpts;
  }

  void HandleDiagnostic(DiagnosticsEngine::Level Level,
                        const Diagnostic &Info) override {
    StoredDiagnostic StoredDiag(Level, Info);
    StandaloneDiags.emplace_back(*LangOpts, StoredDiag);
    DiagnosticConsumer::HandleDiagnostic(Level, Info);
  }

  /// Takes the collected diagnostics.
  StandaloneDiagList takeDiagnostics() { return std::move(StandaloneDiags); }

private:
  const LangOptions *LangOpts = nullptr;
  StandaloneDiagList StandaloneDiags;
};

/// RAII utility to report StandaloneDiagnostics through a DiagnosticsEngine.
///
/// The driver's DiagnosticsEngine usually does not have a SourceManager at
/// this point of building the compilation, in which case the
/// StandaloneDiagReporter supplies its own.
class StandaloneDiagReporter {
public:
  explicit StandaloneDiagReporter(DiagnosticsEngine &Diags);

  /// Emits all diagnostics in \c StandaloneDiags using the associated
  /// DiagnosticsEngine.
  void Report(ArrayRef<StandaloneDiagnostic> StandaloneDiags) const;

private:
  DiagnosticsEngine &Diags;
  IntrusiveRefCntPtr<FileManager> OwnedFileMgr;
  IntrusiveRefCntPtr<SourceManager> OwnedSrcMgr;

  FileManager &getFileManager() const;
  SourceManager &getSourceManager() const;
};

} // anonymous namespace

StandaloneDiagReporter::StandaloneDiagReporter(DiagnosticsEngine &Diags)
    : Diags(Diags) {
  if (!Diags.hasSourceManager()) {
    FileSystemOptions Opts;
    Opts.WorkingDir = ".";
    OwnedFileMgr = llvm::makeIntrusiveRefCnt<FileManager>(std::move(Opts));
    OwnedSrcMgr =
        llvm::makeIntrusiveRefCnt<SourceManager>(Diags, *OwnedFileMgr);
  }
}

void StandaloneDiagReporter::Report(
    ArrayRef<StandaloneDiagnostic> StandaloneDiags) const {
  llvm::StringMap<SourceLocation> SrcLocCache;
  Diags.getClient()->BeginSourceFile(LangOptions(), nullptr);
  for (auto &StandaloneDiag : StandaloneDiags) {
    const auto StoredDiag = translateStandaloneDiag(
        getFileManager(), getSourceManager(), StandaloneDiag, SrcLocCache);
    Diags.Report(StoredDiag);
  }
  Diags.getClient()->EndSourceFile();
}

FileManager &StandaloneDiagReporter::getFileManager() const {
  if (OwnedFileMgr)
    return *OwnedFileMgr;
  return Diags.getSourceManager().getFileManager();
}

SourceManager &StandaloneDiagReporter::getSourceManager() const {
  if (OwnedSrcMgr)
    return *OwnedSrcMgr;
  return Diags.getSourceManager();
}

namespace {

/// Collects the results of DependencyScanningWorker calls from multiple threads
/// into deterministically ordered scan results and diagnostics.
class ScanResultCollector {
public:
  explicit ScanResultCollector(size_t NumInputs)
      : InputDeps(NumInputs), ModuleDeps(NumInputs), DiagLists(NumInputs) {}

  /// Records the dependency scan results \p TUDeps for the scan input at \p
  /// InputIndex.
  ///
  /// Thread safe, given that each index is written to at most once.
  void handleTUDeps(deps::TranslationUnitDeps &&TUDeps, StringRef Triple,
                    size_t InputIndex);

  /// Records the diagnostics produced by the scan for the input at
  /// \p InputIndex.
  ///
  /// Thread safe, given that each index is written to at most once.
  void
  handleDiagnostics(SmallVectorImpl<StandaloneDiagnostic> &&StandaloneDiags,
                    size_t InputIndex);

  /// Finalizes and takes all aggregated scan results.
  ///
  /// Not thread-safe.
  DependencyScanResults takeScanResults();

  /// Takes all collected diagnostics.
  ///
  /// Not thread-safe.
  SmallVector<StandaloneDiagList, 0> takeDiagnostics();

private:
  SmallVector<std::optional<InputDependencies>, 0> InputDeps;
  SmallVector<std::vector<deps::ModuleDeps>, 0> ModuleDeps;
  SmallVector<StandaloneDiagList, 0> DiagLists;
};

} // anonymous namespace

void ScanResultCollector::handleTUDeps(deps::TranslationUnitDeps &&TUDeps,
                                       StringRef Triple, size_t InputIndex) {
  assert(!InputDeps[InputIndex].has_value() &&
         "Each slot should be written to at most once.");
  InputDeps[InputIndex].emplace();
  auto &NewInputDep = InputDeps[InputIndex].value();
  NewInputDep.ModuleName = std::move(TUDeps.ID.ModuleName);
  NewInputDep.NamedModuleDeps = std::move(TUDeps.NamedModuleDeps);
  NewInputDep.ClangModuleDeps = std::move(TUDeps.ClangModuleDeps);
  assert(TUDeps.Commands.size() == 1 && "Expected exactly one command");
  NewInputDep.BuildArgs = TUDeps.Commands.front().Arguments;

  assert(ModuleDeps[InputIndex].empty() &&
         "Each slot should be written to at most once.");
  ModuleDeps[InputIndex] = std::move(TUDeps.ModuleGraph);
}

void ScanResultCollector::handleDiagnostics(
    SmallVectorImpl<StandaloneDiagnostic> &&DiagsList, size_t InputIndex) {
  assert(DiagLists[InputIndex].empty() &&
         "Each slot should be written to at most once.");
  DiagLists[InputIndex] = std::move(DiagsList);
}

DependencyScanResults ScanResultCollector::takeScanResults() {
  DependencyScanResults Results;

  // Record each module only once, from its first importing input.
  // This keeps the output deterministic.
  llvm::DenseSet<deps::ModuleID> AlreadySeen;
  for (auto &ModuleGraph : ModuleDeps) {
    for (auto &MD : ModuleGraph) {
      auto [It, Inserted] = AlreadySeen.insert(MD.ID);
      if (!Inserted)
        continue;
      Results.ModuleDeps.push_back(std::move(MD));
    }
  }

  Results.InputDeps = std::move(InputDeps);
  return Results;
}

SmallVector<StandaloneDiagList, 0> ScanResultCollector::takeDiagnostics() {
  return std::move(DiagLists);
}

namespace {

/// Thread-safe registry of standard library scan inputs.
struct StdlibScanInputRegistry {
  StdlibScanInputRegistry(
      const ScanInputContext::StdlibScanInputMap &StdlibInputLookup);

  /// Returns the indices of all standard library scan inputs corresponding
  /// to modules newly discovered in \p NamedDeps.
  ///
  /// Thread-safe.
  SmallVector<size_t, 2> getNewScanInputs(ArrayRef<std::string> NamedDeps,
                                          StringRef Triple);

private:
  const ScanInputContext::StdlibScanInputMap &StdlibInputLookup;
  llvm::SmallDenseSet<size_t> IsInputScheduled;
  std::mutex Lock;
};

} // anonymous namespace

StdlibScanInputRegistry::StdlibScanInputRegistry(
    const ScanInputContext::StdlibScanInputMap &StdlibInputLookup)
    : StdlibInputLookup(StdlibInputLookup) {
  IsInputScheduled.reserve(StdlibInputLookup.size());
}

SmallVector<size_t, 2>
StdlibScanInputRegistry::getNewScanInputs(ArrayRef<std::string> NamedDeps,
                                          StringRef Triple) {
  SmallVector<size_t, 2> NewScanInputs;

  std::scoped_lock<std::mutex> Guard(Lock);
  for (StringRef DepName : NamedDeps) {
    const auto It = StdlibInputLookup.find(std::make_pair(DepName, Triple));
    if (It == StdlibInputLookup.end())
      continue;
    const size_t InputIndex = It->second;
    const bool AlreadyScanned = !IsInputScheduled.insert(InputIndex).second;
    if (AlreadyScanned)
      continue;
    NewScanInputs.push_back(InputIndex);
  }

  return NewScanInputs;
}

/// Construct a path for the explicitly built PCM.
static std::string constructPCMPath(const deps::ModuleID &ID,
                                    StringRef OutputDir) {
  assert(!ID.ModuleName.empty() && !ID.ContextHash.empty() &&
         "Invalid ModuleID!");
  SmallString<256> ExplicitPCMPath(OutputDir);
  llvm::sys::path::append(ExplicitPCMPath, ID.ContextHash,
                          ID.ModuleName + "-" + ID.ContextHash + ".pcm");
  return std::string(ExplicitPCMPath);
}

namespace {

/// A simple dependency action controller that only provides module lookup for
/// Clang modules.
class ModuleLookupController : public deps::DependencyActionController {
public:
  ModuleLookupController(StringRef OutputDir) : OutputDir(OutputDir) {}

  std::string lookupModuleOutput(const deps::ModuleDeps &MD,
                                 deps::ModuleOutputKind Kind) override {
    if (Kind == deps::ModuleOutputKind::ModuleFile)
      return constructPCMPath(MD.ID, OutputDir);

    // Driver command lines that trigger lookups for unsupported
    // ModuleOutputKinds are not supported by the modules driver. Those
    // command lines should probably be adjusted or rejected in
    // Driver::handleArguments or Driver::HandleImmediateArgs.
    llvm::reportFatalInternalError(
        "call to lookupModuleOutput with unexpected ModuleOutputKind");
  }

private:
  StringRef OutputDir;
};

} // anonymous namespace

/// Constructs the full command line, including the executable, for \c Job.
static SmallVector<std::string, 0> buildCommandLine(const Command &Job) {
  const auto &JobArgs = Job.getArguments();
  SmallVector<std::string, 0> CommandLine;
  CommandLine.reserve(JobArgs.size() + 1);
  CommandLine.emplace_back(Job.getExecutable());
  for (const auto &Arg : JobArgs)
    CommandLine.emplace_back(Arg);
  return CommandLine;
}

/// Runs a dependency scan for a single compilation job.
///
/// \returns The pair of discovered dependencies (or std::nullopt on failure)
/// and the diagnostics collected during the scan.
static std::pair<std::optional<deps::TranslationUnitDeps>, StandaloneDiagList>
scanDependenciesForJob(const Command &Job, ScanningWorkerPool &WorkerPool,
                       ModuleLookupController &LookupController) {
  StandaloneDiagCollector DiagConsumer;
  std::optional<deps::TranslationUnitDeps> TUDeps;

  {
    const auto CC1CommandLine = buildCommandLine(Job);
    auto WorkerHandle = WorkerPool.scopedAcquire();
    deps::FullDependencyConsumer FullDepsConsumer(WorkerHandle->SeenModules);

    if (WorkerHandle->Worker->computeDependencies(
            /*WorkingDirectory*/ ".", CC1CommandLine, FullDepsConsumer,
            LookupController, DiagConsumer))
      TUDeps = FullDepsConsumer.takeTranslationUnitDeps();
  }

  return {TUDeps, DiagConsumer.takeDiagnostics()};
}

/// Scans the given range of -cc1 jobs for module dependencies.
///
/// Inputs for standard library modules are scanned on demand if imported by any
/// user-provided input. The association between user and standard library
/// inputs is provided by \p InputContext.
///
/// \returns the dependency scan result, or std::nullopt on failure, with all
/// diagnostics reported to \p Diags in both cases.
template <typename JobRange>
static std::optional<DependencyScanResults> scanDependencies(
    const JobRange &ScanInputs, const ScanInputContext &InputContext,
    StringRef ModuleCachePath, IntrusiveRefCntPtr<llvm::vfs::FileSystem> BaseFS,
    DiagnosticsEngine &Diags) {
  llvm::PrettyStackTraceString CrashInfo("Performing module dependency scan.");

  deps::DependencyScanningService ScanningService(
      deps::ScanningMode::DependencyDirectivesScan,
      deps::ScanningOutputFormat::Full);

  const size_t NumInputs = llvm::size(ScanInputs);
  const bool HasStdlibInputs = !InputContext.StdlibInputLookup.empty();

  auto ThreadAndWorkerPool = createOptimalThreadAndWorkerPool(
      NumInputs, HasStdlibInputs, ScanningService, BaseFS);
  // Note: Lambda capture of structured bindings is C++20 and later.
  auto &ThreadPool = *ThreadAndWorkerPool.first;
  auto &WorkerPool = *ThreadAndWorkerPool.second;

  ScanResultCollector ResultCollector(NumInputs);
  ModuleLookupController LookupController(ModuleCachePath);
  StdlibScanInputRegistry StdlibInputRegistry(InputContext.StdlibInputLookup);
  std::atomic<bool> HasError = false;

  // Scans one input and schedules scans for newly discovered standard library
  // module imports.
  std::function<void(size_t)> ScanOneAndScheduleNew;
  ScanOneAndScheduleNew = [&](size_t InputIndex) {
    const Command &InputJob = *(ScanInputs.begin() + InputIndex);
    auto [MaybeTUDeps, Diags] =
        scanDependenciesForJob(InputJob, WorkerPool, LookupController);

    // Always capture diagnostics, as even successful scans may produce
    // warnings or notes.
    ResultCollector.handleDiagnostics(std::move(Diags), InputIndex);

    if (!MaybeTUDeps) {
      HasError.store(true, std::memory_order_relaxed);
      return;
    }

    /// Schedule scans for any newly discovered imports of Standard library
    /// modules.
    StringRef Triple = getTriple(InputJob);
    for (auto NewInputIndex : StdlibInputRegistry.getNewScanInputs(
             MaybeTUDeps->NamedModuleDeps, Triple)) {
      ThreadPool.async(
          [&, NewInputIndex]() { ScanOneAndScheduleNew(NewInputIndex); });
    }

    ResultCollector.handleTUDeps(std::move(*MaybeTUDeps), Triple, InputIndex);
  };

  // Initiate the dependency scan with all user inputs.
  for (const size_t UserInputIndex : InputContext.UserInputIndices) {
    ThreadPool.async([&ScanOneAndScheduleNew, UserInputIndex]() {
      ScanOneAndScheduleNew(UserInputIndex);
    });
  }
  ThreadPool.wait();

  // Report the diagnostics for each dependency scan.
  StandaloneDiagReporter DiagReporter(Diags);
  for (auto &DiagsList : ResultCollector.takeDiagnostics())
    DiagReporter.Report(DiagsList);

  if (HasError.load(std::memory_order_relaxed))
    return std::nullopt;
  return ResultCollector.takeScanResults();
}

void driver::modules::runModulesDriver(
    Compilation &C, ArrayRef<StdModuleManifest::Module> ManifestEntries) {
  llvm::PrettyStackTraceString CrashInfo("Running modules driver.");
  auto &Diags = C.getDriver().getDiags();

  const auto MaybeModuleCachePath = getModuleCachePath(C.getArgs());
  if (!MaybeModuleCachePath) {
    Diags.Report(diag::err_default_modules_cache_not_available);
    return;
  }

  auto Graph = createGraphFromJobs(C.getJobs().takeJobs());

  // Build the list of scan inputs and the associated context and for any jobs
  // corresponding to manifest entries, apply the specified manifest-specified
  // local-arguments.
  const auto CC1Nodes =
      llvm::map_range(llvm::make_filter_range(Graph, llvm::IsaPred<CC1JobNode>),
                      llvm::CastTo<CC1JobNode>);
  const auto ManifestLookup = createManifestLookupMap(ManifestEntries);
  SmallVector<CC1JobNode *> ScanInputNodes;
  ScanInputContext InputContext;
  for (auto *CC1Node : CC1Nodes) {
    auto &CC1Job = *CC1Node->Job;
    if (auto *ManifestEntry = getManifestEntryForJob(CC1Job, ManifestLookup)) {
      if (const auto LocalArgs = ManifestEntry->LocalArgs)
        addSystemIncludeDirsFromManifest(C, CC1Job,
                                         LocalArgs->SystemIncludeDirs);
      if (isEligibleScanInput(CC1Job)) {
        const size_t InputIndex = ScanInputNodes.size();
        ScanInputNodes.push_back(CC1Node);

        StringRef Triple =
            CC1Job.getCreator().getToolChain().getTriple().getTriple();
        InputContext.StdlibInputLookup.try_emplace(
            {ManifestEntry->LogicalName, Triple}, InputIndex);
      }
    } else if (isEligibleScanInput(CC1Job)) {
      const size_t InputIndex = ScanInputNodes.size();
      ScanInputNodes.push_back(CC1Node);
      InputContext.UserInputIndices.push_back(InputIndex);
    }
  }

  const auto ScanInputJobs = llvm::map_range(
      ScanInputNodes, [](const auto *CC1Node) { return *CC1Node->Job; });
  auto MaybeScanResults =
      scanDependencies(ScanInputJobs, InputContext, *MaybeModuleCachePath,
                       &C.getDriver().getVFS(), Diags);
  if (!MaybeScanResults) {
    Diags.Report(diag::err_dependency_scan_failed);
    return;
  }

  pruneUnimportedStdlibModuleJobs(Graph, ScanInputNodes,
                                  MaybeScanResults->InputDeps);

  // TODO: Generate -cc1 jobs for each Clang module and pass in the jobs here
  // instead.
  auto ClangModuleNodes =
      createClangModuleNodes(Graph, MaybeScanResults->ModuleDeps.size());

  const bool Success =
      addModuleDependencyInfo(Graph, ScanInputNodes, ClangModuleNodes,
                              std::move(*MaybeScanResults), Diags);
  if (!Success)
    return;

  createAndConnectRootNode(Graph);

  // TODO: Detect cyclic dependencies in the module dependency graph.

  Diags.Report(diag::remark_printing_module_graph);
  if (!Diags.isLastDiagnosticIgnored())
    llvm::WriteGraph<const CompilationGraph *>(llvm::errs(), &Graph);

  // TODO: Update each driver job's command line to emit or pass-in the correct
  // module files.

  // TODO: Topologically sort the graph and merge the jobs back into the
  // compilation's job list.
}
