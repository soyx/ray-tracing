#ifndef CORE_SHAPE_H
#define CORE_SHAPE_H

#include <memory>

#include "transform.h"
#include "util.h"

class Shape {
 public:
  virtual bool intersect(const Ray& ray, SurfaceData& surface_data) const = 0;

  virtual Bounds3f worldBound() const = 0;

};

#endif  // CORE_SHAPE_H
