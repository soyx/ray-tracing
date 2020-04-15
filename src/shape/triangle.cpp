#include "shape/triangle.h"

TriangleMesh::TriangleMesh(const Transform& object_to_world, int n_triangles,
                           const std::vector<int>& vertex_indices,
                           const std::vector<Point3f>& points,
                           const std::vector<Normal3f>& normals) {
  this->num_triangles = n_triangles;
  this->vertex_indices = vertex_indices;
  if (object_to_world != Transform::Identity()) {
    p.resize(points.size());
    n.resize(normals.size());
    for (int i = 0; i < points.size(); ++i) {
      p[i] = object_to_world(points[i]);
    }
    for (int i = 0; i < normals.size(); ++i) {
      n[i] = object_to_world(normals[i]);
    }
  } else {
    p = points;
    n = normals;
  }
}

TriangleMesh::TriangleMesh(const Transform& object_to_world, int n_triangles,
                           std::vector<int>&& vertex_indices,
                           std::vector<Point3f>&& points,
                           std::vector<Normal3f>&& normals) {
  this->num_triangles = n_triangles;
  this->vertex_indices = std::move(vertex_indices);
  if (object_to_world != Transform::Identity()) {
    p.resize(points.size());
    n.resize(normals.size());
    for (int i = 0; i < points.size(); ++i) {
      p[i] = object_to_world(points[i]);
    }
    for (int i = 0; i < normals.size(); ++i) {
      n[i] = object_to_world(normals[i]);
    }
  } else {
    this->p = std::move(points);
    this->n = std::move(normals);
  }
}

Triangle::Triangle(const std::shared_ptr<TriangleMesh>& mesh, int tri_index)
    : mesh_(mesh) {
  v = &this->mesh_->vertex_indices[tri_index * 3];
}

bool Triangle::intersect(const Ray& ray, SurfaceData& surface_data) const {
  const Point3f& p1 = mesh_->p[v[0]];
  const Point3f& p2 = mesh_->p[v[1]];
  const Point3f& p3 = mesh_->p[v[2]];

  const Vector3f& v12 = p2 - p1;
  const Vector3f& v13 = p3 - p1;

  const Normal3f& geo_n = Cross(v12, v13).normalize();

  if (Dot(geo_n, ray.d) >= 0) return false;

  const Vector3f& op1 = p1 - ray.o;
  Float t = Dot(op1, geo_n) / Dot(ray.d, geo_n);


  if(t < 0 || t > surface_data.ray_t) return false;

  const Point3f& ip = ray(t);

  // compute barycentric coordinates
  int maxn_coor;
  if (std::abs(geo_n.x) > std::abs(geo_n.y)) {
    maxn_coor = std::abs(geo_n.x) > std::abs(geo_n.z) ? 0 : 2;
  } else {
    maxn_coor = std::abs(geo_n.y) > std::abs(geo_n.z) ? 1 : 2;
  }

  Point2f p1p, p2p, p3p, ipp;
  if (maxn_coor == 0) {
    // project to yoz
    p1p.x = p1.y;
    p1p.y = p1.z;
    p2p.x = p2.y;
    p2p.y = p2.z;
    p3p.x = p3.y;
    p3p.y = p3.z;
    ipp.x = ip.y;
    ipp.y = ip.z;
  } else if (maxn_coor == 1) {
    // project to zox
    p1p.x = p1.z;
    p1p.y = p1.x;
    p2p.x = p2.z;
    p2p.y = p2.x;
    p3p.x = p3.z;
    p3p.y = p3.x;
    ipp.x = ip.z;
    ipp.y = ip.x;
  } else {
    // project to xoy
    p1p.x = p1.x;
    p1p.y = p1.y;
    p2p.x = p2.x;
    p2p.y = p2.y;
    p3p.x = p3.x;
    p3p.y = p3.y;
    ipp.x = ip.x;
    ipp.y = ip.y;
  }

  const Vector2f& vp_p1 = p1p - ipp;
  const Vector2f& vp_p2 = p2p - ipp;
  const Vector2f& vp_p3 = p3p - ipp;

  Float e1 = vp_p2.x * vp_p3.y - vp_p2.y * vp_p3.x;
  Float e2 = vp_p3.x * vp_p1.y - vp_p3.y * vp_p1.x;
  Float e3 = vp_p1.x * vp_p2.y - vp_p1.y * vp_p2.x;

  if(e1 < 0 && e2 < 0 && e3 < 0){
    e1 = -e1;
    e2 = -e2;
    e3 = -e3;
  }
  if (e1 < 0 || e2 < 0 || e3 < 0) return false;

  Float det_inv = 1 / (e1 + e2 + e3);
  Float b1 = e1 * det_inv;
  Float b2 = e2 * det_inv;
  Float b3 = e3 * det_inv;
  surface_data.ray_t = t;
  surface_data.position = ray(t);

  surface_data.normal = BarycentricInerpolation(mesh_->n[v[0]], mesh_->n[v[1]],
                                                mesh_->n[v[2]], b1, b2, b3);

  return true;
}

Bounds3f Triangle::worldBound() const {
  const Point3f& p1 = mesh_->p[v[0]];
  const Point3f& p2 = mesh_->p[v[1]];
  const Point3f& p3 = mesh_->p[v[2]];
  return Union(Bounds3f(p1, p2), p3);
}