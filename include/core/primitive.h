#ifndef CORE_PRIMITIVE_H
#define CORE_PRIMITIVE_H

#include <memory>

#include "util.h"
#include "shape.h"
#include "material.h"

class Primitive {
 public:
  virtual Float intersectP(const Ray &ray) const = 0;

  virtual Bounds3f worldBound() const = 0;

 private:
 std::shared_ptr<Shape> shape_;
 std::shared_ptr<Material> material_;
};

#endif  // CORE_PRIMITIVE_H
