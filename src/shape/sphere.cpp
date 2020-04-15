#include "shape/sphere.h"

bool Sphere::intersect(const Ray& ray, SurfaceData& surface_data) const {
  Vector3f op = position - ray.o;
  Float a = Dot(ray.d, ray.d);
  Float b = -2 * Dot(ray.d, op);
  Float c = Dot(op, op) - radius * radius;

  Float delta = b * b - 4 * a * c;
  if (delta < 0) return false;
  Float sdelta = std::sqrt(delta);
  Float t = 0;
  Float t1, t2;
  t1 = (-b - sdelta) / (2 * a);
  t2 = (-b + sdelta) / (2 * a);
  if (t1 > 0) {
    t = t1;
    if (t2 > 0 && t2 < t) t = t2;
  } else {
    if (t2 > 0) t = t2;
  }

  if(t > 1e-6){
    if(t > surface_data.ray_t) return false;
    surface_data.position = (ray.o + ray.d * t);
    surface_data.normal = Normal3f((surface_data.position - position)).normalize();
    surface_data.ray_t = t; 
    return true;
  }
  return false;

}

Bounds3f Sphere::worldBound() const {
  return Bounds3f(position - Vector3f(radius, radius, radius),
                  position + Vector3f(radius, radius, radius));
}