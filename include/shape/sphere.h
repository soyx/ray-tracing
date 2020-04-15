#ifndef SHAPE_SPHERE_H
#define SHAPE_SPHERE_H

#include "core/shape.h"
#include "core/util.h"

class Sphere : public Shape {
 public:
  Sphere(const Point3f& position, const Float radius)
      : position(position), radius(radius) {}

  bool intersect(const Ray& ray, SurfaceData& surface_data) const final;
  Bounds3f worldBound() const final;

 private:
  // for simple, position is in world space
  Point3f position;
  Float radius;
};

#endif  // SHAPE_SPHERE_H