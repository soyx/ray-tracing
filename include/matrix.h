#ifndef MATRIX_H
#define MATRIX_H

#include "util.h"

#include <iostream>
#include <cmath>

class Mat4
{
public:
  float m[4][4];

  Mat4();
  Mat4(const float mat[4][4]);
  Mat4(float m00, float m01, float m02, float m03,
       float m10, float m11, float m12, float m13,
       float m20, float m21, float m22, float m23,
       float m30, float m31, float m32, float m33);

  Mat4 operator=(const Mat4 &m2);
  Mat4 operator+(const Mat4 &m2) const;
  Mat4 operator-(const Mat4 &m2) const;
  Mat4 operator*(const Mat4 &m2) const;

  bool operator==(const Mat4 &m2) const;
  bool operator!=(const Mat4 &m2) const;

  Mat4 inverse() const;
  Mat4 transpose() const;
};
#endif // MATRIX_H