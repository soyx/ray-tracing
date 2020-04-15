#ifndef CORE_MATRIX_H
#define CORE_MATRIX_H

#include <cmath>
#include <cstring>
#include <iostream>

#include "util.h"

class Mat4 {
 public:
  Float m[4][4];

  Mat4();
  Mat4(const Float mat[4][4]);
  Mat4(Float m00, Float m01, Float m02, Float m03, Float m10, Float m11,
       Float m12, Float m13, Float m20, Float m21, Float m22, Float m23,
       Float m30, Float m31, Float m32, Float m33);
  static Mat4 Identity();

  Mat4 operator=(const Mat4 &m2);
  Mat4 operator+(const Mat4 &m2) const;
  Mat4 operator-(const Mat4 &m2) const;
  Mat4 operator*(const Mat4 &m2) const;

  bool operator==(const Mat4 &m2) const;
  bool operator!=(const Mat4 &m2) const;

  Mat4 inverse() const;
  Mat4 transpose() const;
};
#endif  // CORE_MATRIX_H
