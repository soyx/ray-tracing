#include "shape/cube.h"

#include <array>

bool Cube::intersect(const Ray& ray, SurfaceData& surface_data) const {
  Bounds3f cube_bbox(position - Vector3f(1.0f, 1.0f, 1.0f) * 0.5 * length,
                     position + Vector3f(1.0f, 1.0f, 1.0f) * 0.5 * length);

  auto&& tp_min = (cube_bbox.pMin - ray.o) / ray.d;
  auto&& tp_max = (cube_bbox.pMax - ray.o) / ray.d;

  std::array<Float, 2> interval_x{tp_min.x, tp_max.x};
  std::array<Float, 2> interval_y{tp_min.y, tp_max.y};
  std::array<Float, 2> interval_z{tp_min.z, tp_max.z};

  if (interval_x[0] > interval_x[1]) std::swap(interval_x[0], interval_x[1]);
  if (interval_y[0] > interval_y[1]) std::swap(interval_y[0], interval_y[1]);
  if (interval_z[0] > interval_z[1]) std::swap(interval_z[0], interval_z[1]);

  Float t_min = std::max(std::max(interval_x[0], interval_y[0]), interval_z[0]);
  Float t_max = std::min(std::min(interval_x[1], interval_y[1]), interval_z[1]);

  Float t = 0;
  if (t_min < t_max - 1e-6) {
    if(t_min > 0) t = t_min;
    else t = t_max;
    if(t <= 0) return false;
    if(t > surface_data.ray_t) return false;
    surface_data.position = ray.o + ray.d * t;
    surface_data.ray_t = t;
    Vector3f&& v = (surface_data.position - position).normalize();
    if (std::abs(v.x) > std::abs(v.y)) {
      if (std::abs(v.x) > std::abs(v.z)) {
        surface_data.normal = v.x > 0 ? Normal3f(1, 0, 0) : Normal3f(-1, 0, 0);
      } else {
        surface_data.normal = v.z > 0 ? Normal3f(0, 0, 1) : Normal3f(0, 0, -1);
      }
    } else {
      if (std::abs(v.y) > std::abs(v.z)) {
        surface_data.normal = v.y > 0 ? Normal3f(0, 1, 0) : Normal3f(0, -1, 0);
      } else {
        surface_data.normal = v.z > 0 ? Normal3f(0, 0, 1) : Normal3f(0, 0, -1);
      }
    }
    return true;
  }
  return false;
};

Bounds3f Cube::worldBound() const {
  return Bounds3f(position - Vector3f(1.0f, 1.0f, 1.0f) * 0.5 * length,
                  position + Vector3f(1.0f, 1.0f, 1.0f) * 0.5 * length);
}