#ifndef RANDOM_H
#define RANDOM_H

#include <random>

#include "util.h"

class Random {
 public:
  Random() { u = std::uniform_real_distribution<Float>(0, 1); }

  Random(unsigned int seed) {
      e.seed(seed);
      u = std::uniform_real_distribution<Float>(0, 1);
  }

  Float uniformRandom(){
      return u(e);
  }

 private:
  std::default_random_engine e;
  std::uniform_real_distribution<Float> u;
};
#endif  // RANDOM_H