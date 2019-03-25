#ifndef MATRIX_H
#define MATRIX_H

#include "util.h"

#include <iostream>
#include <cmath>
#include <cstring>

class Mat4
{
public:
  double m[4][4];

  __host__ __device__ Mat4();
  __host__ __device__ Mat4(const double mat[4][4]);
  __host__ __device__ Mat4(double m00, double m01, double m02, double m03,
       double m10, double m11, double m12, double m13,
       double m20, double m21, double m22, double m23,
       double m30, double m31, double m32, double m33);

  __host__ __device__ Mat4 operator=(const Mat4 &m2);
  __host__ __device__ Mat4 operator+(const Mat4 &m2) const;
  __host__ __device__ Mat4 operator-(const Mat4 &m2) const;
  __host__ __device__ Mat4 operator*(const Mat4 &m2) const;

  __host__ __device__ bool operator==(const Mat4 &m2) const;
  __host__ __device__ bool operator!=(const Mat4 &m2) const;

  __host__ __device__ Mat4 inverse() const;
  __host__ __device__ Mat4 transpose() const;
};
#endif // MATRIX_H
