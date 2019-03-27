#ifndef UTIL_H
#define UTIL_H

#include <limits>
#include <cmath>
#include <ctime>

#define INF std::numeric_limits<double>::infinity()
#define PI 3.1415926

template<typename T>
class Vec2 {
public:
    T x, y;

    T maxCor;

    Vec2() {
        x = y = 0;
        maxCor = 0;
    }

    Vec2(T x, T y) : x(x), y(y) {
        maxCor = x > y ? x : y;
    }

    Vec2<T> &operator=(const Vec2<T> &v) {
        this->x = v.x;
        this->y = v.y;
        this->maxCor = v.maxCor;
        return *this;
    }

    Vec2<T> operator+(const Vec2<T> v) const {
        return Vec2(x + v.x, y + v.y);
    }

    Vec2<T> operator-(const Vec2<T> v) const {
        return Vec2(x - v.x, y - v.y);
    }

    template<typename U>
    Vec2<T> operator*(const U s) const {
        return Vec2(x * s, y * s);
    }

    template<typename U>
    Vec2<T> operator/(const U s) const {
        return Vec2(x / s, y / s);
    }

    T operator[](int i) const {
        if (i == 0) return x;
        else return y;
    }

    T &operator[](int i) {
        if (i == 0) return x;
        else return y;
    }


};

template<typename T>
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

    template<typename U>
    Vec3<T> operator*(const U s) const {
        return Vec3<T>(x * s, y * s, z * s);
    }

    template<typename U>
    Vec3<T> operator/(const U s) const {
        T temp = (T) (1. / s);
        return Vec3(x * temp, y * temp, z * temp);
    }

    T operator[](int i) const {
        if (i == 0) return x;
        else if (i == 1) return y;
        else
            return z;
    }

    T &operator[](int i) {
        if (i == 0) return x;
        else if (i == 1) return y;
        else
            return z;
    }
};

template<typename T>
Vec2<T> mul(const Vec2<T>& v1, const Vec2<T>& v2) {
    return Vec2<T>(v1.x * v2.x, v1.y * v2.y);
}

template<typename T>
Vec3<T> mul(const Vec3<T>& v1, const Vec3<T>& v2) {
    return Vec3<T>(v1.x * v2.x, v1.y * v2.y, v1.z * v2.z);
}


using Vec2f = Vec2<double>;
using Vec3f = Vec3<double>;

// Vector
template<typename T>
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
    template<typename U>
    Vector2<T> operator*(U s) const {
        return Vector2<T>(s * x, s * y);
    }

    // vec2 *= scalar
    template<typename U>
    Vector2<T> &operator*=(U s) {
        x *= s;
        y *= s;
        return *this;
    }

    // vec2 / scalar
    template<typename U>
    Vector2<T> operator/(U s) const {
        return Vector2<T>(s / x, s / y);
    }

    // vec2 /= scalar
    template<typename U>
    Vector2<T> &operator/=(U s) {
        x /= s;
        y /= s;
        return *this;
    }

    double getMagnitudeSquare() const {
        return x * x + y * y;
    }

    Vector2<T> normalize() const {
        return (*this) / std::sqrt(getMagnitudeSquare());
    }
};

template<typename T>
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
    template<typename U>
    Vector3<T> operator*(U s) const {
        return Vector3<T>(s * x, s * y, s * z);
    }

    // vec3 *= scalar
    template<typename U>
    Vector3<T> &operator*=(U s) {
        x *= s;
        y *= s;
        z *= s;
        return *this;
    }

    // vec3 / vec3
    template<typename U>
    const Vector3<T> operator/(Vector3<U> v) const {
        if (v.x == 0 || v.y == 0 || v.z == 0)
            return Vector3<T>(0, 0, 0);
        return Vector3<T>(x / v.x, y / v.y, z / v.z);
    }

    // vec3 / scalar
    template<typename U>
    Vector3<T> operator/(U s) const {
        if (s == 0)
            return Vector3<T>(0, 0, 0);
        return Vector3<T>(x / s, y / s, z / s);
    }

    // vec3 /= scalar
    template<typename U>
    Vector3<T> &operator/=(U s) {
        x /= s;
        y /= s;
        z /= s;
        return *this;
    }

    double getMagnitudeSquare() const {
        return x * x + y * y + z * z;
    }

    Vector3<T> normalize() const {
        return (*this) / std::sqrt(getMagnitudeSquare());
    }
};

template<typename T>
T dot(const Vector2<T> &v1, const Vector2<T> &v2) {
    return v1.x * v2.x + v1.y * v2.y;
}

template<typename T>
T dot(const Vector3<T> &v1, const Vector3<T> &v2) {
    return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}

template<typename T>
Vector3<T> cross(const Vector3<T> &v1, const Vector3<T> &v2) {
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

template<typename T>
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

template<typename T>
double distanceSquare(const Point2<T> &p1, const Point2<T> &p2) {
    return (p1 - p2).getMagnitudeSquare();
}

using Point2f = Point2<double>;
using Point3f = Point3<double>;

// Normal
template<typename T>
class Normal3 {
public:
    T x, y, z;

    Normal3() { x = y = z = 0; }

    Normal3(T x, T y, T z) : x(x), y(y), z(z) {}

    Normal3<T> &operator=(const Normal3<T> n3) {
        this->x = n3.x;
        this->y = n3.y;
        this->z = n3.z;
        return *this;
    }

    void normalize() {
        double m = std::sqrt(getMagnitudeSquare());
        if (m > 0) {
            x /= m;
            y /= m;
            z /= m;
        }
    }

    double getMagnitudeSquare() const {
        return x * x + y * y + z * z;
    }
};

template<typename T>
T dot(Vector3<T> v, Normal3<T> n) {
    return v.x * n.x + v.y * n.y + v.z * n.z;
}

using Normal3f = Normal3<double>;

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

// bounding box
template<typename T>
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

using Bounds2f = Bounds2<double>;
#endif // UTIL_H
