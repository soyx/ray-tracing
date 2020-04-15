#ifndef CORE_TRANSFORM_H
#define CORE_TRANSFORM_H

#include "matrix.h"
#include "util.h"
class Transform {
 public:
  Transform();
  Transform(Mat4 m);
  Transform(Mat4 m, Mat4 inv);

  static Transform Identity();

  Transform getMatrix() const;
  Transform getInverseMatrix() const;

  Transform inverse() const;

  Transform transpose() const;

  Point3f operator()(const Point3f &p) const;
  Vector3f operator()(const Vector3f &v) const;
  Normal3f operator()(const Normal3f &n) const;
  Ray operator()(const Ray &r) const;

  Transform operator*(const Transform &t2) const;

  bool operator==(const Transform& t2) const;
  bool operator!=(const Transform& t2) const;

 private:
  Mat4 mat_;
  Mat4 inv_;
};

Transform translate(const Vector3f &delta);
Transform translate(Float x, Float y, Float z);
Transform scale(Float x, Float y, Float z);
Transform rotateX(Float theta);
Transform rotateX(Float theta);
Transform rotateX(Float theta);
Transform rotate(Float theta, const Vector3f &axis);
Transform lookAt(const Point3f &pos, const Point3f &look, const Vector3f &up);
Transform perspective(Float fov, Float znear, Float zfar);
#endif
