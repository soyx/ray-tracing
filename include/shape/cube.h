#ifndef SHAPE_CUBE_H
#define SHAPE_CUBE_H

#include "core/shape.h"
#include "core/util.h"

class Cube : public Shape {
 public:
  Cube(const Point3f& position, const Float length)
      : position(position), length(length) {}

  bool intersect(const Ray& ray, SurfaceData& surface_data) const final;
  Bounds3f worldBound() const final;

 private:
  // for simple, position is in world space
  Point3f position;
  Float length;
};

#endif  // SHAPE_CUBE_H