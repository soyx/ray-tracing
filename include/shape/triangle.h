#ifndef SHAPE_TRIANGLE_H
#define SHAPE_TRIANGLE_H
#include <array>
#include <memory>
#include <vector>

#include "core/shape.h"
#include "core/util.h"

struct TriangleMesh {
  TriangleMesh(const Transform& object_to_world, int n_triangles,
               const std::vector<int>& vertex_indices,
               const std::vector<Point3f>& points,
               const std::vector<Normal3f>& normals);

  TriangleMesh(const Transform& object_to_world, int n_triangles,
               std::vector<int>&& vertex_indices, std::vector<Point3f>&& points,
               std::vector<Normal3f>&& normals);
  int num_triangles;
  std::vector<int> vertex_indices;

  std::vector<Point3f> p;
  std::vector<Normal3f> n;
};

class Triangle : public Shape {
 public:
  Triangle(const std::shared_ptr<TriangleMesh>& mesh, int tri_index);

  bool intersect(const Ray& ray, SurfaceData& surface_data) const final;

  Bounds3f worldBound() const final;

 private:
  std::shared_ptr<TriangleMesh> mesh_;
  const int* v;
};

#endif // SHAPE_TRIANGLE_H