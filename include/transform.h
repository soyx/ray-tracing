#ifndef TRANSFORM_H
#define TRANSFORM_H

#include "util.h"
#include "matrix.h"

class Transform
{
public:
  __host__ __device__ Transform();
  __host__ __device__ Transform(Mat4 m);
  __host__ __device__ Transform(Mat4 m, Mat4 inv);

  __host__ __device__ Transform getMatrix() const;
  __host__ __device__ Transform getInverseMatrix() const;

  __host__ __device__ Transform inverse() const;

  __host__ __device__ Transform transpose() const;

  __host__ __device__ Point3f operator()(const Point3f &p) const;
  __host__ __device__ Vector3f operator()(const Vector3f &v) const;

  __host__ __device__ Transform operator*(const Transform &t2) const;

private:
  Mat4 mat;
  Mat4 inv;
};

 __host__ __device__ Transform translate(const Vector3f &delta);
 __host__ __device__ Transform scale(double x, double y, double z);
 __host__ __device__ Transform rotateX(double theta);
 __host__ __device__ Transform rotateX(double theta);
 __host__ __device__ Transform rotateX(double theta);
 __host__ __device__ Transform rotate(double theta, const Vector3f &axis);
 __host__ __device__ Transform lookAt(const Point3f &pos, const Point3f &look, const Vector3f &up);
 __host__ __device__ Transform perspective(double fov, double znear, double zfar);
#endif
