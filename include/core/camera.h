#ifndef CORE_CAMERA_H
#define CORE_CAMERA_H

#include <vector>

#include "model.h"
#include "transform.h"
#include "util.h"

class Camera {
 public:
  Camera(){};

  Camera(const Point3f& position, const Point3f& target,
         const Vector3f& world_up, const Float focalLength = 5,
         Float fovy = 60. / 180 * M_PI, int film_width = 800,
         int film_height = 600);

  void setPerspective(Float fovy, Float near, Float far);

  void writeColor(int r, int c, const Vec3f& color);

  Ray generateRay(const Point2f& pos);

  Ray generateRay(const Float x, const Float y);

  // world_to_camera and camera_to_ndc
  Transform view_transform, project_transform;

  Transform film_to_camera;

  std::vector<Vec3f> film;
  Vec2<int> film_size;

  Point3f position;

  Float near, far;
  Float fovy;

 private:
  Vector3f up;
  Point3f target;
  Float focalLength;
};

#endif  // CORE_CAMERA_H