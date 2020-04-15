#include "core/transform.h"

Transform::Transform() {}

Transform::Transform(Mat4 m) {
  mat_ = m;
  inv_ = m.inverse();
}

Transform::Transform(Mat4 m, Mat4 inv) {
  this->mat_ = m;
  this->inv_ = inv;
}

Transform Transform::Identity() {
  return Transform(Mat4::Identity(), Mat4::Identity());
}

Transform Transform::getMatrix() const { return this->mat_; }

Transform Transform::getInverseMatrix() const { return this->inv_; }

Transform Transform::inverse() const {
  return Transform(this->inv_, this->mat_);
}

Transform Transform::transpose() const {
  return Transform(this->mat_.transpose(), this->inv_.transpose());
}

Point3f Transform::operator()(const Point3f &p) const {
  Float xx = mat_.m[0][0] * p.x + mat_.m[0][1] * p.y + mat_.m[0][2] * p.z +
             mat_.m[0][3];
  Float yy = mat_.m[1][0] * p.x + mat_.m[1][1] * p.y + mat_.m[1][2] * p.z +
             mat_.m[1][3];
  Float zz = mat_.m[2][0] * p.x + mat_.m[2][1] * p.y + mat_.m[2][2] * p.z +
             mat_.m[2][3];
  Float ww = mat_.m[3][0] * p.x + mat_.m[3][1] * p.y + mat_.m[3][2] * p.z +
             mat_.m[3][3];
  if (ww == 1)
    return Point3f(xx, yy, zz);
  else
    return Point3f(xx / ww, yy / ww, zz / ww);
}

Vector3f Transform::operator()(const Vector3f &v) const {
  return Vector3f(mat_.m[0][0] * v.x + mat_.m[0][1] * v.y + mat_.m[0][2] * v.z,
                  mat_.m[1][0] * v.x + mat_.m[1][1] * v.y + mat_.m[1][2] * v.z,
                  mat_.m[2][0] * v.x + mat_.m[2][1] * v.y + mat_.m[2][2] * v.z);
}

Normal3f Transform::operator()(const Normal3f &n) const {
  return Normal3f(inv_.m[0][0] * n.x + inv_.m[1][0] * n.y + inv_.m[2][0] * n.z,
                  inv_.m[0][1] * n.x + inv_.m[1][1] * n.y + inv_.m[2][1] * n.z,
                  inv_.m[0][2] * n.x + inv_.m[1][2] * n.y + inv_.m[2][2] * n.z);
}

Ray Transform::operator()(const Ray &r) const {
  return Ray((*this)(r.o), (*this)(r.d));
}

Transform Transform::operator*(const Transform &t2) const {
  return Transform(mat_ * t2.mat_, t2.inv_ * inv_);
}
bool Transform::operator==(const Transform &t2) const {
  return t2.mat_ == mat_;
}
bool Transform::operator!=(const Transform &t2) const {
  return t2.mat_ != mat_;
}

Transform translate(const Vector3f &delta) {
  Mat4 mat(1, 0, 0, delta.x, 0, 1, 0, delta.y, 0, 0, 1, delta.z, 0, 0, 0, 1);
  Mat4 inv(1, 0, 0, -delta.x, 0, 1, 0, -delta.y, 0, 0, 1, -delta.z, 0, 0, 0, 1);
  return Transform(mat, inv);
}

Transform translate(Float x, Float y, Float z) {
  Mat4 mat(1, 0, 0, x, 0, 1, 0, y, 0, 0, 1, z, 0, 0, 0, 1);
  Mat4 inv(1, 0, 0, -x, 0, 1, 0, -y, 0, 0, 1, -z, 0, 0, 0, 1);
  return Transform(mat, inv);
}

Transform scale(Float x, Float y, Float z) {
  Mat4 mat(x, 0, 0, 0, 0, y, 0, 0, 0, 0, z, 0, 0, 0, 0, 1);

  Mat4 inv(1 / x, 0, 0, 0, 0, 1 / y, 0, 0, 0, 0, 1 / z, 0, 0, 0, 0, 1);

  return Transform(mat, inv);
}

Transform rotateX(Float theta) {
  Float sinTheta = std::sin(theta);
  Float cosTheta = std::cos(theta);
  Mat4 mat(1, 0, 0, 0, 0, cosTheta, -sinTheta, 0, 0, sinTheta, cosTheta, 0, 0,
           0, 0, 1);

  Mat4 inv(1, 0, 0, 0, 0, cosTheta, sinTheta, 0, 0, -sinTheta, cosTheta, 0, 0,
           0, 0, 1);
  return Transform(mat, inv);
}

Transform rotateY(Float theta) {
  Float sinTheta = std::sin(theta);
  Float cosTheta = std::cos(theta);
  Mat4 mat(cosTheta, 0, sinTheta, 0, 0, 1, 0, 0, -sinTheta, 0, cosTheta, 0, 0,
           0, 0, 1);

  Mat4 inv(cosTheta, 0, -sinTheta, 0, 0, 1, 0, 0, sinTheta, 0, cosTheta, 0, 0,
           0, 0, 1);
  return Transform(mat, inv);
}

Transform rotateZ(Float theta) {
  Float sinTheta = std::sin(theta);
  Float cosTheta = std::cos(theta);
  Mat4 mat(cosTheta, -sinTheta, 0, 0, sinTheta, cosTheta, 0, 0, 0, 0, 1, 0, 0,
           0, 0, 1);

  Mat4 inv(cosTheta, sinTheta, 0, 0, -sinTheta, cosTheta, 0, 0, 0, 0, 1, 0, 0,
           0, 0, 1);
  return Transform(mat, inv);
}

Transform rotate(Float theta, const Vector3f &axis) {
  Vector3f a = axis.normalize();
  Float sinTheta = std::sin(theta);
  Float cosTheta = std::cos(theta);
  Mat4 m;
  // Compute rotation of first basis vector
  m.m[0][0] = a.x * a.x + (1 - a.x * a.x) * cosTheta;
  m.m[0][1] = a.x * a.y * (1 - cosTheta) - a.z * sinTheta;
  m.m[0][2] = a.x * a.z * (1 - cosTheta) + a.y * sinTheta;
  m.m[0][3] = 0;

  // Compute rotations of second and third basis vectors
  m.m[1][0] = a.x * a.y * (1 - cosTheta) + a.z * sinTheta;
  m.m[1][1] = a.y * a.y + (1 - a.y * a.y) * cosTheta;
  m.m[1][2] = a.y * a.z * (1 - cosTheta) - a.x * sinTheta;
  m.m[1][3] = 0;

  m.m[2][0] = a.x * a.z * (1 - cosTheta) - a.y * sinTheta;
  m.m[2][1] = a.y * a.z * (1 - cosTheta) + a.x * sinTheta;
  m.m[2][2] = a.z * a.z + (1 - a.z * a.z) * cosTheta;
  m.m[2][3] = 0;
  return Transform(m, m.transpose());
}

Transform lookAt(const Point3f &pos, const Point3f &look, const Vector3f &up) {
  Mat4 camera2World;
  camera2World.m[0][3] = pos.x;
  camera2World.m[1][3] = pos.y;
  camera2World.m[2][3] = pos.z;
  camera2World.m[3][3] = 1;

  Vector3f dir = (look - pos).normalize();
  Vector3f right = Cross(up.normalize(), dir).normalize();
  Vector3f newUp = Cross(dir, right);

  camera2World.m[0][0] = right.x;
  camera2World.m[1][0] = right.y;
  camera2World.m[2][0] = right.z;
  camera2World.m[3][0] = 0.f;
  camera2World.m[0][1] = newUp.x;
  camera2World.m[1][1] = newUp.y;
  camera2World.m[2][1] = newUp.z;
  camera2World.m[3][1] = 0.f;
  camera2World.m[0][2] = dir.x;
  camera2World.m[1][2] = dir.y;
  camera2World.m[2][2] = dir.z;
  camera2World.m[3][2] = 0.f;
  return Transform(camera2World.inverse(), camera2World);
}

Transform perspective(Float fov, Float n, Float f) {
  // Perform projective divide for perspective projection
  Mat4 persp(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, f / (f - n), -f * n / (f - n), 0, 0,
             1, 0);
  // Scale canonical perspective view to specified field of view
  Float inv_tan_ang = 1 / std::tan((Float)(fov / 180 * PI) / 2);
  return scale(inv_tan_ang, inv_tan_ang, 1) * Transform(persp);
}