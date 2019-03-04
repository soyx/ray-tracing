#ifndef UTIL_H
#define UTIL_H

#include <limits>
#include <cmath>

#define INF std::numeric_limits<float>::infinity()

// Vector
template <typename T>
class Vector2
{
  public:
    T x, y;

    Vector2() { x = y = 0; }
    Vector2(T x, T y) : x(x), y(y) {}
    Vector2(const Vector2<T> &v)
    {
        x = v.x;
        y = v.y;
    }

    Vector2<T> &operator=(const Vector2<T> &v2)
    {
        this->x = v2.x;
        this->y = v2.y;
        return *this;
    }

    // vec2 + vec2
    Vector2<T> operator+(const Vector2<T> &v2) const
    {
        return Vector2<T>(x + v2.x, y + v2.y);
    }
    // vec2 += vec2
    Vector2<T> &operator+=(const Vector2<T> &v2)
    {
        x += v2.x;
        y += v2.y;
        return *this;
    }

    // vec2 - vec2
    Vector2<T> operator-(const Vector2<T> &v2) const
    {
        return Vector2<T>(x - v2.x, y - v2.y);
    }
    // vec2 -= vec2
    Vector2<T> &operator-=(const Vector2<T> &v2)
    {
        x -= v2.x;
        y -= v2.y;
        return *this;
    }

    // vec2 * scalar
    template <typename U>
    Vector2<T> operator*(U s) const
    {
        return Vector2<T>(s * x, s * y);
    }
    // vec2 *= scalar
    template <typename U>
    Vector2<T> &operator*=(U s)
    {
        x *= s;
        y *= s;
        return *this;
    }

    // vec2 / scalar
    template <typename U>
    Vector2<T> operator/(U s) const
    {
        return Vector2<T>(s / x, s / y);
    }
    // vec2 /= scalar
    template <typename U>
    Vector2<T> &operator/=(U s)
    {
        x /= s;
        y /= s;
        return *this;
    }

    float getMagnitudeSquare() const
    {
        return x * x + y * y;
    }

    Vector2<T> normalize() const
    {
        return (*this) / std::sqrtf(getMagnitudeSquare());
    }
};

template <typename T>
class Vector3
{
  public:
    T x, y, z;

    Vector3() { x = y = z = 0; }
    Vector3(T x, T y, T z) : x(x), y(y), z(z) {}
    Vector3(const Vector3<T> &v)
    {
        x = v.x;
        y = v.y;
        z = v.z;
    }

    Vector3<T> &operator=(const Vector3<T> &v3)
    {
        this->x = v3.x;
        this->y = v3.y;
        this->z = v3.z;
        return *this;
    }

    // vec3 + vec3
    Vector3<T> operator+(const Vector3<T> &v3) const
    {
        return Vector3<T>(x + v3.x, y + v3.y, z + v3.z);
    }
    // vec3 += vec3
    Vector3<T> &operator+=(const Vector3<T> &v3)
    {
        x += v3.x;
        y += v3.y;
        z += v3.z;
        return *this;
    }

    // vec3 - vec3
    Vector3<T> operator-(const Vector3<T> &v3) const
    {
        return Vector3<T>(x - v3.x, y - v3.y, z - v3.z);
    }
    // vec3 -= vec3
    Vector3<T> &operator-=(const Vector3<T> &v3)
    {
        x -= v3.x;
        y -= v3.y;
        z -= v3.z;
        return *this;
    }

    // vec3 * scalar
    template <typename U>
    Vector3<T> operator*(U s) const
    {
        return Vector3<T>(s * x, s * y, s * z);
    }
    // vec3 *= scalar
    template <typename U>
    Vector3<T> &operator*=(U s)
    {
        x *= s;
        y *= s;
        z *= s;
        return *this;
    }

    // vec3 / scalar
    template <typename U>
    Vector3<T> operator/(U s) const
    {
        return Vector3<T>(s / x, s / y, s / z);
    }
    // vec3 /= scalar
    template <typename U>
    Vector3<T> &operator/=(U s)
    {
        x /= s;
        y /= s;
        z /= s;
        return *this;
    }

    float getMagnitudeSquare() const
    {
        return x * x + y * y + z * z;
    }

    Vector3<T> normalize() const
    {
        return (*this) / std::sqrtf(getMagnitudeSquare());
    }
};

template <typename T>
T dot(const Vector2<T> &v1, const Vector2<T> &v2)
{
    return v1.x * v2.x + v1.y * v2.y;
}

template <typename T>
T dot(const Vector3<T> &v1, const Vector3<T> &v2)
{
    return v1.x * v2.x + v1.y + v2.y + v1.z * v2.z;
}

template <typename T>
Vector3<T> cross(const Vector3<T> &v1, const Vector3<T> &v2)
{
    return Vector3<T>(v1.y * v2.z - v1.z * v2.y,
                      v1.z * v2.x - v1.x * v2.z,
                      v1.x * v2.y - v1.y * v2.x);
}

using Vector2f = Vector2<float>;
using Vector3f = Vector3<float>;

// Point
template <typename T>
class Point2
{
  public:
    T x, y;

    Point2() { x = y = 0; }
    Point2(T x, T y) : x(x), y(y) {}
    Point2(const Point2<T> &p)
    {
        x = p.x;
        y = p.y;
    }

    Point2<T> &operator=(const Point2<T> &p)
    {
        this->x = p.x;
        this->y = p.y;
        return *this;
    }

    Point2<T> operator+(const Vector2<T> &v) const
    {
        return Point2<T>(x + v.x, y + v.y);
    }

    Point2<T> &operator+=(const Vector2<T> &v)
    {
        x += v.x;
        y += v.y;
        return *this;
    }

    Point2<T> operator-(const Vector2<T> &v) const
    {
        return Point2<T>(x - v.x, y - v.y);
    }
    Vector2<T> operator-(const Point2<T> &p) const
    {
        return Vector2<T>(x - p.x, y - p.y);
    }
    Point2<T> &operator-=(const Vector2<T> &v) const
    {
        x -= v.x;
        y -= v.y;
        return *this;
    }
};

template <typename T>
class Point3
{
  public:
    T x, y, z;

    Point3() { x = y = z = 0; }
    Point3(T x, T y, T z) : x(x), y(y), z(z) {}
    Point3(const Point3<T> &p)
    {
        x = p.x;
        y = p.y;
        z = p.z;
    }

    Point3<T> &operator=(const Point3<T> &p) const
    {
        this->x = p.x;
        this->y = p.y;
        this->z = p.z;
        return *this;
    }
    Point3<T> operator+(const Vector3<T> &v) const
    {
        return Point3<T>(x + v.x, y + v.y, z + v.z);
    }

    Point3<T> &operator+=(const Vector3<T> &v)
    {
        x += v.x;
        y += v.y;
        z += v.z;
        return *this;
    }

    Point3<T> operator-(const Vector3<T> &v) const
    {
        return Point3<T>(x - v.x, y - v.y, z - v.z);
    }
    Vector3<T> operator-(const Point3<T> &p) const
    {
        return Vector3<T>(x - p.x, y - p.y, z - p.z);
    }
    Point3<T> &operator-=(const Vector3<T> &v) const
    {
        x -= v.x;
        y -= v.y;
        z -= v.z;
        return *this;
    }
};

template <typename T>
float distanceSquare(const Point2<T> &p1, const Point2<T> &p2)
{
    return (p1 - p2).getMagnitudeSquare();
}

using Point2f = Point2<float>;
using Point3f = Point3<float>;

// Normal
template <typename T>
class Normal3
{
  public:
    T x, y, z;

    Normal3() { x = y = z = 0; }
    Normal3(T x, T y, T z) : x(x), y(y), z(z) {}

    Normal3<T> &operator=(const Normal3<T> n3) const
    {
        this->x = n3.x;
        this->y = n3.y;
        this->z = n3.z;
        return *this;
    }

    void normalize()
    {
        float m = std::sqrt(getMagnitudeSquare());
        if (m > 0)
        {
            x /= m;
            y /= m;
            z /= m;
        }
    }

    float getMagnitudeSquare() const
    {
        return x * x + y * y + z * z;
    }
};

template <typename T>
T dot(Vector3<T> v, Normal3<T> n)
{
    return v.x * n.x + v.y * n.y + v.z * n.z;
}

using Normal3f = Normal3<float>;

class Ray
{
  public:
    // origin point
    Point3f o;
    // direction of light
    Vector3f d;
    // the max length of the light when the unit is the mag of the direction
    mutable float tMax;

    float time;

    Ray() : tMax(INF), time(0.f) {}
    Ray(const Point3f &o, const Vector3f &d,
        float tMax = INF, float time = 0.f)
        : o(o), d(d), tMax(tMax), time(time) {}

    Point3f operator()(float t) const
    {
        return o + d * t;
    }
};

#endif // UTIL_H