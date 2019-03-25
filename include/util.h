#ifndef UTIL_H
#define UTIL_H

#include <limits>
#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>
#include "cuda/helper_cuda.h"
#include "cuda/helper_functions.h"

#define INF std::numeric_limits<double>::infinity()
#define PI 3.1415926

template<typename T>
class Vec2 {
public:
    T x, y;

    T maxCor;

    __host__ __device__ Vec2() {
        x = y = 0;
        maxCor = 0;
    }

    __host__ __device__ Vec2(T x, T y) : x(x), y(y) {
        maxCor = x > y ? x : y;
    }

    __host__ __device__ Vec2<T> &operator=(const Vec2<T> &v) {
        this->x = v.x;
        this->y = v.y;
        this->maxCor = v.maxCor;
        return *this;
    }

    __host__ __device__ Vec2<T> operator+(const Vec2<T> v) const {
        return Vec2(x + v.x, y + v.y);
    }

    __host__ __device__ Vec2<T> operator-(const Vec2<T> v) const {
        return Vec2(x - v.x, y - v.y);
    }

    template<typename U>
    __host__ __device__ Vec2<T> operator*(const U s) const {
        return Vec2(x * s, y * s);
    }

    template<typename U>
    __host__ __device__ Vec2<T> operator/(const U s) const {
        return Vec2(x / s, y / s);
    }

    __host__ __device__ T operator[](int i) const {
        if (i == 0) return x;
        else return y;
    }

    __host__ __device__ T &operator[](int i) {
        if (i == 0) return x;
        else return y;
    }


};

template<typename T>
class Vec3 {
public:
    T x, y, z;

    T maxCor;

    __host__ __device__ Vec3() {
        x = y = z = 0;
        maxCor = 0;
    }

    __host__ __device__ Vec3(T x, T y, T z) : x(x), y(y), z(z) {
        maxCor = x > y && x > z ? x : y > z ? y : z;
    }

    __host__ __device__ Vec3<T> &operator=(const Vec3<T> &v) {
        this->x = v.x;
        this->y = v.y;
        this->z = v.z;
        this->maxCor = v.maxCor;
        return *this;
    }

    __host__ __device__ Vec3<T> operator+(const Vec3<T> v) const {
        return Vec3<T>(x + v.x, y + v.y, z + v.z);
    }

    __host__ __device__ Vec3<T> operator-(const Vec3<T> v) const {
        return Vec3<T>(x - v.x, y - v.y, z + v.z);
    }

    template<typename U>
    __host__ __device__ Vec3<T> operator*(const U s) const {
        return Vec3<T>(x * s, y * s, z * s);
    }

    template<typename U>
    __host__ __device__ Vec3<T> operator/(const U s) const {
        T temp = (T) (1. / s);
        return Vec3(x * temp, y * temp, z * temp);
    }

    __host__ __device__ T operator[](int i) const {
        if (i == 0) return x;
        else if (i == 1) return y;
        else
            return z;
    }

    __host__ __device__ T &operator[](int i) {
        if (i == 0) return x;
        else if (i == 1) return y;
        else
            return z;
    }
};

template<typename T>
 __host__ __device__ Vec2<T> mul(const Vec2<T>& v1, const Vec2<T>& v2) {
    return Vec2<T>(v1.x * v2.x, v1.y * v2.y);
}

template<typename T>
 __host__ __device__ Vec3<T> mul(const Vec3<T>& v1, const Vec3<T>& v2) {
    return Vec3<T>(v1.x * v2.x, v1.y * v2.y, v1.z * v2.z);
}


using Vec2f = Vec2<double>;
using Vec3f = Vec3<double>;

// Vector
template<typename T>
class Vector2 {
public:
    T x, y;

__host__ __device__ Vector2() { x = y = 0; }

    __host__ __device__ Vector2(T x, T y) : x(x), y(y) {}

    __host__ __device__ Vector2(const Vector2<T> &v) {
        x = v.x;
        y = v.y;
    }

    __host__ __device__ Vector2<T> &operator=(const Vector2<T> &v2) {
        this->x = v2.x;
        this->y = v2.y;
        return *this;
    }

    // vec2 + vec2
    __host__ __device__ Vector2<T> operator+(const Vector2<T> &v2) const {
        return Vector2<T>(x + v2.x, y + v2.y);
    }

    // vec2 += vec2
    __host__ __device__ Vector2<T> &operator+=(const Vector2<T> &v2) {
        x += v2.x;
        y += v2.y;
        return *this;
    }

    // vec2 - vec2
    __host__ __device__ Vector2<T> operator-(const Vector2<T> &v2) const {
        return Vector2<T>(x - v2.x, y - v2.y);
    }

    // vec2 -= vec2
    __host__ __device__ Vector2<T> &operator-=(const Vector2<T> &v2) {
        x -= v2.x;
        y -= v2.y;
        return *this;
    }

    // vec2 * scalar
    template<typename U>
    __host__ __device__ Vector2<T> operator*(U s) const {
        return Vector2<T>(s * x, s * y);
    }

    // vec2 *= scalar
    template<typename U>
    __host__ __device__ Vector2<T> &operator*=(U s) {
        x *= s;
        y *= s;
        return *this;
    }

    // vec2 / scalar
    template<typename U>
    __host__ __device__ Vector2<T> operator/(U s) const {
        return Vector2<T>(s / x, s / y);
    }

    // vec2 /= scalar
    template<typename U>
    __host__ __device__ Vector2<T> &operator/=(U s) {
        x /= s;
        y /= s;
        return *this;
    }

    __host__ __device__ double getMagnitudeSquare() const {
        return x * x + y * y;
    }

    __host__ __device__ Vector2<T> normalize() const {
        return (*this) / std::sqrt(getMagnitudeSquare());
    }
};

template<typename T>
class Vector3 {
public:
    T x, y, z;

    __host__ __device__ Vector3() { x = y = z = 0; }

    __host__ __device__ Vector3(T x, T y, T z) : x(x), y(y), z(z) {}

    __host__ __device__ Vector3(const Vector3<T> &v) {
        x = v.x;
        y = v.y;
        z = v.z;
    }

    __host__ __device__ Vector3<T> &operator=(const Vector3<T> &v3) {
        this->x = v3.x;
        this->y = v3.y;
        this->z = v3.z;
        return *this;
    }

    // vec3 + vec3
    __host__ __device__ Vector3<T> operator+(const Vector3<T> &v3) const {
        return Vector3<T>(x + v3.x, y + v3.y, z + v3.z);
    }

    // vec3 += vec3
    __host__ __device__ Vector3<T> &operator+=(const Vector3<T> &v3) {
        x += v3.x;
        y += v3.y;
        z += v3.z;
        return *this;
    }

    // vec3 - vec3
    __host__ __device__ Vector3<T> operator-(const Vector3<T> &v3) const {
        return Vector3<T>(x - v3.x, y - v3.y, z - v3.z);
    }

    // vec3 -= vec3
    __host__ __device__ Vector3<T> &operator-=(const Vector3<T> &v3) {
        x -= v3.x;
        y -= v3.y;
        z -= v3.z;
        return *this;
    }

    // vec3 * scalar
    template<typename U>
    __host__ __device__ Vector3<T> operator*(U s) const {
        return Vector3<T>(s * x, s * y, s * z);
    }

    // vec3 *= scalar
    template<typename U>
    __host__ __device__ Vector3<T> &operator*=(U s) {
        x *= s;
        y *= s;
        z *= s;
        return *this;
    }

    // vec3 / vec3
    template<typename U>
    __host__ __device__ const Vector3<T> operator/(Vector3<U> v) const {
        if (v.x == 0 || v.y == 0 || v.z == 0)
            return Vector3<T>(0, 0, 0);
        return Vector3<T>(x / v.x, y / v.y, z / v.z);
    }

    // vec3 / scalar
    template<typename U>
    __host__ __device__ Vector3<T> operator/(U s) const {
        if (s == 0)
            return Vector3<T>(0, 0, 0);
        return Vector3<T>(x / s, y / s, z / s);
    }

    // vec3 /= scalar
    template<typename U>
    __host__ __device__ Vector3<T> &operator/=(U s) {
        x /= s;
        y /= s;
        z /= s;
        return *this;
    }

    __host__ __device__ double getMagnitudeSquare() const {
        return x * x + y * y + z * z;
    }

    __host__ __device__ Vector3<T> normalize() const {
        return (*this) / std::sqrt(getMagnitudeSquare());
    }
};

template<typename T>
 __host__ __device__ T dot(const Vector2<T> &v1, const Vector2<T> &v2) {
    return v1.x * v2.x + v1.y * v2.y;
}

template<typename T>
 __host__ __device__ T dot(const Vector3<T> &v1, const Vector3<T> &v2) {
    return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}

template<typename T>
 __host__ __device__ Vector3<T> cross(const Vector3<T> &v1, const Vector3<T> &v2) {
    return Vector3<T>(v1.y * v2.z - v1.z * v2.y,
                      v1.z * v2.x - v1.x * v2.z,
                      v1.x * v2.y - v1.y * v2.x);
}

using Vector2f = Vector2<double>;
using Vector3f = Vector3<double>;

// Point
template<typename T>
class Point2 {
public:
    T x, y;

    __host__ __device__ Point2() { x = y = 0; }

    __host__ __device__ Point2(T x, T y) : x(x), y(y) {}

    __host__ __device__ Point2(const Point2<T> &p) {
        x = p.x;
        y = p.y;
    }

    __host__ __device__ Point2<T> &operator=(const Point2<T> &p) {
        this->x = p.x;
        this->y = p.y;
        return *this;
    }

    __host__ __device__ Point2<T> operator+(const Vector2<T> &v) const {
        return Point2<T>(x + v.x, y + v.y);
    }

    __host__ __device__ Point2<T> &operator+=(const Vector2<T> &v) {
        x += v.x;
        y += v.y;
        return *this;
    }

    __host__ __device__ Point2<T> operator-(const Vector2<T> &v) const {
        return Point2<T>(x - v.x, y - v.y);
    }

    __host__ __device__ Vector2<T> operator-(const Point2<T> &p) const {
        return Vector2<T>(x - p.x, y - p.y);
    }

    __host__ __device__ Point2<T> &operator-=(const Vector2<T> &v) const {
        x -= v.x;
        y -= v.y;
        return *this;
    }
};

template<typename T>
class Point3 {
public:
    T x, y, z;

    __host__ __device__ Point3() { x = y = z = 0; }

    __host__ __device__ Point3(T x, T y, T z) : x(x), y(y), z(z) {}

    __host__ __device__ Point3(const Point3<T> &p) {
        x = p.x;
        y = p.y;
        z = p.z;
    }

    __host__ __device__ Point3<T> &operator=(const Point3<T> &p) {
        this->x = p.x;
        this->y = p.y;
        this->z = p.z;
        return *this;
    }

    __host__ __device__ Point3<T> operator+(const Vector3<T> &v) const {
        return Point3<T>(x + v.x, y + v.y, z + v.z);
    }

    __host__ __device__ Point3<T> &operator+=(const Vector3<T> &v) {
        x += v.x;
        y += v.y;
        z += v.z;
        return *this;
    }

    __host__ __device__ Point3<T> operator-(const Vector3<T> &v) const {
        return Point3<T>(x - v.x, y - v.y, z - v.z);
    }

    __host__ __device__ Vector3<T> operator-(const Point3<T> &p) const {
        return Vector3<T>(x - p.x, y - p.y, z - p.z);
    }

    __host__ __device__ Point3<T> &operator-=(const Vector3<T> &v) const {
        x -= v.x;
        y -= v.y;
        z -= v.z;
        return *this;
    }
};


using Point2f = Point2<double>;
using Point3f = Point3<double>;

class Ray {
public:
    // origin point
    Point3f o;
    // direction of light
    Vector3f d;

    Ray(const Point3f &o, const Vector3f &d) : o(o), d(d) {}

    Point3f operator()(double t) const {
        return o + d * t;
    }
};

#endif // UTIL_H
