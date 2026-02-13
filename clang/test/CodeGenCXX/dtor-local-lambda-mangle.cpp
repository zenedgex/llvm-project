// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++20 -O2 -emit-llvm -o /dev/null %s

struct E {
  E();
  ~E();
};

E::E() {
  struct {
    int anotherValue = [] { return 2; }();
  } obj;
}

E::~E() {
  struct {
    int anotherValue = [] { return 2; }();
  } obj;
}
