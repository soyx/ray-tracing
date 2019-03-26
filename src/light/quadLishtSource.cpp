#include "light/quadLightSource.h"

double QuadLightSource::intersect(const Ray &ray) const {
    if (dot(normal, ray.d) == 0) return 0;

    double a = normal.x;
    double b = normal.y;
    double c = normal.z;
    double d = -(a * center.x + b * center.y +  c * center.z);

    double t =
        (-d - (a * ray.o.x + b * ray.o.y + c * ray.o.z)) / dot(normal, ray.d);

    Point3f p = ray.o + ray.d * t;

    // 假设光源面与坐标面平行

    Vector3f v = p - center;

    if (normal.x != 0) {
        if (std::abs(v.y) <= size.x && std::abs(v.z) <= size.y) return t;
    } else if (normal.y != 0) {
        if (std::abs(v.z) <= size.x && std::abs(v.x) <= size.y)
            return t;
    } else if (normal.z != 0) {
        if (std::abs(v.x) <= size.x && std::abs(v.y) <= size.y)
            return t;
    }
    return 0;
}

void QuadLightSource::clear(){
    this->center = Point3f();
    this->emission = Vec3f();
    this->KDiffuse = Vec3f();
    this->KSpecular = Vec3f();
    this->normal = Vector3f();
    this->size = Vec2f();
}