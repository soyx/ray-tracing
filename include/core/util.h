#ifndef CORE_UTIL_H
#define CORE_UTIL_H

#include <cmath>
#include <ctime>
#include <limits>

using Float = float;
// #define INF std::numeric_limits<Float>::infinity()
constexpr Float PI = M_PI;

template <typename T>
class Vec2 {
 public:
  T x, y;

  T maxCor;

  Vec2() {
    x = y = 0;
    maxCor = 0;
  }

  Vec2(T x, T y) : x(x), y(y) { maxCor = x > y ? x : y; }

  Vec2<T> &operator=(const Vec2<T> &v) {
    this->x = v.x;
    this->y = v.y;
    this->maxCor = v.maxCor;
    return *this;
  }

  Vec2<T> operator+(const Vec2<T> v) const { return Vec2(x + v.x, y + v.y); }

  Vec2<T> operator-(const Vec2<T> v) const { return Vec2(x - v.x, y - v.y); }

  template <typename U>
  Vec2<T> operator*(const U s) const {
    return Vec2(x * s, y * s);
  }

  template <typename U>
  Vec2<T> operator/(const U s) const {
    return Vec2(x / s, y / s);
  }

  T operator[](int i) const {
    if (i == 0)
      return x;
    else
      return y;
  }

  T &operator[](int i) {
    if (i == 0)
      return x;
    else
      return y;
  }
};

template <typename T>
class Vec3 {
 public:
  T x, y, z;

  T maxCor;

  Vec3() {
    x = y = z = 0;
    maxCor = 0;
  }

  Vec3(T x, T y, T z) : x(x), y(y), z(z) {
    maxCor = x > y && x > z ? x : y > z ? y : z;
  }

  Vec3<T> &operator=(const Vec3<T> &v) {
    this->x = v.x;
    this->y = v.y;
    this->z = v.z;
    this->maxCor = v.maxCor;
    return *this;
  }

  Vec3<T> operator+(const Vec3<T> v) const {
    return Vec3<T>(x + v.x, y + v.y, z + v.z);
  }

  Vec3<T> operator-(const Vec3<T> v) const {
    return Vec3<T>(x - v.x, y - v.y, z - v.z);
  }

  template <typename U>
  Vec3<T> operator*(const U s) const {
    return Vec3<T>(x * s, y * s, z * s);
  }

  template <typename U>
  Vec3<T> operator/(const U s) const {
    T temp = (T)(1. / s);
    return Vec3(x * temp, y * temp, z * temp);
  }

  T operator[](int i) const {
    if (i == 0)
      return x;
    else if (i == 1)
      return y;
    else
      return z;
  }

  T &operator[](int i) {
    if (i == 0)
      return x;
    else if (i == 1)
      return y;
    else
      return z;
  }
};

template <typename T>
Vec2<T> Mul(const Vec2<T> &v1, const Vec2<T> &v2) {
  return Vec2<T>(v1.x * v2.x, v1.y * v2.y);
}

template <typename T>
Vec3<T> Mul(const Vec3<T> &v1, const Vec3<T> &v2) {
  return Vec3<T>(v1.x * v2.x, v1.y * v2.y, v1.z * v2.z);
}

using Vec2f = Vec2<Float>;
using Vec3f = Vec3<Float>;

using Color3f = Vec3f;
using Radiance3f = Vec3f;

// Vector
template <typename T>
class Vector2 {
 public:
  T x, y;

  Vector2() { x = y = 0; }

  Vector2(T x, T y) : x(x), y(y) {}

  Vector2(const Vector2<T> &v) {
    x = v.x;
    y = v.y;
  }

  Vector2<T> &operator=(const Vector2<T> &v2) {
    this->x = v2.x;
    this->y = v2.y;
    return *this;
  }

  // vec2 + vec2
  Vector2<T> operator+(const Vector2<T> &v2) const {
    return Vector2<T>(x + v2.x, y + v2.y);
  }

  // vec2 += vec2
  Vector2<T> &operator+=(const Vector2<T> &v2) {
    x += v2.x;
    y += v2.y;
    return *this;
  }

  // vec2 - vec2
  Vector2<T> operator-(const Vector2<T> &v2) const {
    return Vector2<T>(x - v2.x, y - v2.y);
  }

  // vec2 -= vec2
  Vector2<T> &operator-=(const Vector2<T> &v2) {
    x -= v2.x;
    y -= v2.y;
    return *this;
  }

  // vec2 * scalar
  template <typename U>
  Vector2<T> operator*(U s) const {
    return Vector2<T>(s * x, s * y);
  }

  // vec2 *= scalar
  template <typename U>
  Vector2<T> &operator*=(U s) {
    x *= s;
    y *= s;
    return *this;
  }

  // vec2 / scalar
  template <typename U>
  Vector2<T> operator/(U s) const {
    return Vector2<T>(s / x, s / y);
  }

  // vec2 /= scalar
  template <typename U>
  Vector2<T> &operator/=(U s) {
    x /= s;
    y /= s;
    return *this;
  }

  Float getMagnitudeSquare() const { return x * x + y * y; }

  Vector2<T> normalize() const {
    return (*this) / std::sqrt(getMagnitudeSquare());
  }
};

template <typename T>
class Vector3 {
 public:
  T x, y, z;

  Vector3() { x = y = z = 0; }

  Vector3(T x, T y, T z) : x(x), y(y), z(z) {}

  Vector3(const Vector3<T> &v) {
    x = v.x;
    y = v.y;
    z = v.z;
  }

  Vector3<T> &operator=(const Vector3<T> &v3) {
    this->x = v3.x;
    this->y = v3.y;
    this->z = v3.z;
    return *this;
  }

  // vec3 + vec3
  Vector3<T> operator+(const Vector3<T> &v3) const {
    return Vector3<T>(x + v3.x, y + v3.y, z + v3.z);
  }

  // vec3 += vec3
  Vector3<T> &operator+=(const Vector3<T> &v3) {
    x += v3.x;
    y += v3.y;
    z += v3.z;
    return *this;
  }

  // vec3 - vec3
  Vector3<T> operator-(const Vector3<T> &v3) const {
    return Vector3<T>(x - v3.x, y - v3.y, z - v3.z);
  }

  // vec3 -= vec3
  Vector3<T> &operator-=(const Vector3<T> &v3) {
    x -= v3.x;
    y -= v3.y;
    z -= v3.z;
    return *this;
  }

  // vec3 * scalar
  template <typename U>
  Vector3<T> operator*(U s) const {
    return Vector3<T>(s * x, s * y, s * z);
  }

  // vec3 *= scalar
  template <typename U>
  Vector3<T> &operator*=(U s) {
    x *= s;
    y *= s;
    z *= s;
    return *this;
  }

  // vec3 / vec3
  template <typename U>
  const Vector3<T> operator/(Vector3<U> v) const {
    if (v.x == 0 || v.y == 0 || v.z == 0) return Vector3<T>(0, 0, 0);
    return Vector3<T>(x / v.x, y / v.y, z / v.z);
  }

  // vec3 / scalar
  template <typename U>
  Vector3<T> operator/(U s) const {
    if (s == 0) return Vector3<T>(0, 0, 0);
    return Vector3<T>(x / s, y / s, z / s);
  }

  // vec3 /= scalar
  template <typename U>
  Vector3<T> &operator/=(U s) {
    x /= s;
    y /= s;
    z /= s;
    return *this;
  }

  Float getMagnitudeSquare() const { return x * x + y * y + z * z; }

  Vector3<T> normalize() const {
    return (*this) / std::sqrt(getMagnitudeSquare());
  }
};

template <typename T>
T Dot(const Vector2<T> &v1, const Vector2<T> &v2) {
  return v1.x * v2.x + v1.y * v2.y;
}

template <typename T>
T Dot(const Vector3<T> &v1, const Vector3<T> &v2) {
  return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}

template <typename T>
Vector3<T> Cross(const Vector3<T> &v1, const Vector3<T> &v2) {
  return Vector3<T>(v1.y * v2.z - v1.z * v2.y, v1.z * v2.x - v1.x * v2.z,
                    v1.x * v2.y - v1.y * v2.x);
}

using Vector2f = Vector2<Float>;
using Vector3f = Vector3<Float>;

// Point
template <typename T>
class Point2 {
 public:
  T x, y;

  Point2() { x = y = 0; }

  Point2(T x, T y) : x(x), y(y) {}

  Point2(const Point2<T> &p) {
    x = p.x;
    y = p.y;
  }

  Point2<T> &operator=(const Point2<T> &p) {
    this->x = p.x;
    this->y = p.y;
    return *this;
  }

  Point2<T> operator+(const Vector2<T> &v) const {
    return Point2<T>(x + v.x, y + v.y);
  }

  Point2<T> &operator+=(const Vector2<T> &v) {
    x += v.x;
    y += v.y;
    return *this;
  }

  Point2<T> operator-(const Vector2<T> &v) const {
    return Point2<T>(x - v.x, y - v.y);
  }

  Vector2<T> operator-(const Point2<T> &p) const {
    return Vector2<T>(x - p.x, y - p.y);
  }

  Point2<T> &operator-=(const Vector2<T> &v) const {
    x -= v.x;
    y -= v.y;
    return *this;
  }
};

template <typename T>
class Point3 {
 public:
  T x, y, z;

  Point3() { x = y = z = 0; }

  Point3(T x, T y, T z) : x(x), y(y), z(z) {}

  Point3(const Point3<T> &p) {
    x = p.x;
    y = p.y;
    z = p.z;
  }

  Point3<T> &operator=(const Point3<T> &p) {
    this->x = p.x;
    this->y = p.y;
    this->z = p.z;
    return *this;
  }

  Point3<T> operator+(const Vector3<T> &v) const {
    return Point3<T>(x + v.x, y + v.y, z + v.z);
  }

  Point3<T> &operator+=(const Vector3<T> &v) {
    x += v.x;
    y += v.y;
    z += v.z;
    return *this;
  }

  Point3<T> operator-(const Vector3<T> &v) const {
    return Point3<T>(x - v.x, y - v.y, z - v.z);
  }

  Vector3<T> operator-(const Point3<T> &p) const {
    return Vector3<T>(x - p.x, y - p.y, z - p.z);
  }

  Point3<T> &operator-=(const Vector3<T> &v) const {
    x -= v.x;
    y -= v.y;
    z -= v.z;
    return *this;
  }
};

template <typename T>
Float DistanceSquare(const Point2<T> &p1, const Point2<T> &p2) {
  return (p1 - p2).getMagnitudeSquare();
}

using Point2f = Point2<Float>;
using Point3f = Point3<Float>;

// Normal
template <typename T>
class Normal3 {
 public:
  T x, y, z;

  Normal3() { x = y = z = 0; }

  Normal3(T x, T y, T z) : x(x), y(y), z(z) {}

  Normal3(Vector3<T> v) : x(v.x), y(v.y), z(v.z) {}

  Normal3<T> &operator=(const Normal3<T> n3) {
    this->x = n3.x;
    this->y = n3.y;
    this->z = n3.z;
    return *this;
  }

  Normal3<T> normalize() {
    Float m = std::sqrt(getMagnitudeSquare());
    return Normal3<T>(x / m, y / m, z / m);
  }

  Float getMagnitudeSquare() const { return x * x + y * y + z * z; }
};

template <typename T>
T Dot(const Vector3<T> &v, const Normal3<T> &n) {
  return v.x * n.x + v.y * n.y + v.z * n.z;
}
template <typename T>
T Dot(const Normal3<T> &n, const Vector3<T> &v) {
  return v.x * n.x + v.y * n.y + v.z * n.z;
}

using Normal3f = Normal3<Float>;

inline Normal3f BarycentricInerpolation(const Normal3f &n1, const Normal3f &n2,
                                 const Normal3f &n3, Float w1, Float w2,
                                 Float w3) {
  const Normal3f &n = Normal3f(n1.x * w1 + n2.x * w2 + n3.x * w3,
                               n1.y * w1 + n2.y * w2 + n3.y * w3,
                               n1.z * w1 + n2.z * w2 + n3.z * w3)
                          .normalize();
  return n;
}

class Ray {
 public:
  // origin point
  Point3f o;
  // direction of light
  Vector3f d;

  Ray(const Point3f &o, const Vector3f &d) : o(o), d(d) {}

  Point3f operator()(Float t) const { return o + d * t; }
};

// bounding box
template <typename T>
class Bounds2 {
 public:
  Bounds2() {
    T minNum = std::numeric_limits<T>::min();
    T maxNum = std::numeric_limits<T>::max();

    pMin = Point2<T>(maxNum, maxNum);
    pMin = Point2<T>(minNum, minNum);
  }

  Bounds2(const Point2<T> &p1, const Point2<T> &p2) {
    pMin = Point2<T>(std::fmin(p1.x, p2.x), std::fmin(p1.y, p2.y));
    pMax = Point2<T>(std::fmax(p1.x, p2.x), std::fmax(p1.y, p2.y));
  }

  Point2<T> pMin, pMax;
};

template <typename T>
class Bounds3 {
 public:
  Bounds3() {
    T minNum = std::numeric_limits<T>::min();
    T maxNum = std::numeric_limits<T>::max();

    pMin = Bounds3<T>(maxNum, maxNum);
    pMin = Bounds3<T>(minNum, minNum);
  }

  Bounds3(const Point3<T> &p1, const Point3<T> &p2) {
    pMin = Point3<T>(std::fmin(p1.x, p2.x), std::fmin(p1.y, p2.y),
                     std::fmin(p1.z, p2.z));
    pMax = Point3<T>(std::fmax(p1.x, p2.x), std::fmax(p1.y, p2.y),
                     std::fmax(p1.z, p2.z));
  }

  Point3<T> pMin, pMax;
};

using Bounds2f = Bounds2<Float>;
using Bounds3f = Bounds3<Float>;

// construct a new box from a box and a point
template <typename T>
Bounds3<T> Union(const Bounds3<T> &b, const Point3<T> &p) {
  return Bounds3<T>(
      Point3<T>(std::min<T>(b.pMin.x, p.x), std::min<T>(b.pMin.y, p.y),
                std::min<T>(b.pMin.z, p.z)),
      Point3<T>(std::min<T>(b.pMax.x, p.x), std::min<T>(b.pMax.y, p.y),
                std::min<T>(b.pMax.z, p.z)));
}
// construct a new box from two boxes
template <typename T>
Bounds3<T> Union(const Bounds3<T> &b1, const Bounds3<T> &b2) {
  return Bounds3<T>(Point3<T>(std::min<T>(b1.pMin.x, b2.pMin.x),
                              std::min<T>(b1.pMin.y, b2.pMin.y),
                              std::min<T>(b1.pMin.z, b2.pMin.z)),
                    Point3<T>(std::max<T>(b1.pMax.x, b2.pMax.x),
                              std::max<T>(b1.pMax.y, b2.pMax.y),
                              std::max<T>(b1.pMax.z, b2.pMax.z)));
}
// the intersection of two box
template <typename T>
Bounds2<T> Intersect(const Bounds2<T> &b1, const Bounds2<T> &b2) {
  return Bounds2<T>(Point2<T>(std::max<T>(b1.pMin.x, b2.pMin.x),
                              std::max<T>(b1.pMin.y, b2.pMin.y)),
                    Point2<T>(std::min<T>(b1.pMax.x, b2.pMax.x),
                              std::min<T>(b1.pMax.y, b2.pMax.y)));
}
template <typename T>
Bounds3<T> Intersect(const Bounds3<T> &b1, const Bounds3<T> &b2) {
  return Bounds3<T>(Point3<T>(std::max<T>(b1.pMin.x, b2.pMin.x),
                              std::max<T>(b1.pMin.y, b2.pMin.y),
                              std::max<T>(b1.pMin.z, b2.pMin.z)),
                    Point3<T>(std::min<T>(b1.pMax.x, b2.pMax.x),
                              std::min<T>(b1.pMax.y, b2.pMax.y),
                              std::min<T>(b1.pMax.z, b2.pMax.z)));
}

struct SurfaceData {
  Float ray_t = std::numeric_limits<Float>::infinity();
  Point3f position;
  Normal3f normal;
  Point2f texture_coord;
};
#endif  // CORE_UTIL_H
