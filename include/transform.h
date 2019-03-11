#ifndef TRANSFORM_H
#define TRANSFORM_H

#include "util.h"
#include "matrix.h"

class Transform
{
public:
  Transform();
  Transform(Mat4 m);
  Transform(Mat4 m, Mat4 inv);

  Transform getMatrix() const;
  Transform getInverseMatrix() const;

  Transform inverse() const;

  Transform transpose() const;

  Point3f operator()(const Point3f &p) const;
  Vector3f operator()(const Vector3f &v) const;
  Normal3f operator()(const Normal3f &n) const;
  Ray operator()(const Ray &r) const;

  Transform operator*(const Transform &t2)const;

private:
  Mat4 mat;
  Mat4 inv;
};

Transform translate(const Vector3f &delta);
Transform scale(float x, float y, float z);
Transform rotateX(float theta);
Transform rotateX(float theta);
Transform rotateX(float theta);
Transform rotate(float theta, const Vector3f &axis);
Transform lookAt(const Point3f &pos, const Point3f &look, const Vector3f &up);
Transform perspective(float fov, float znear, float zfar);
#endif
