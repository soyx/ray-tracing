#include "light/sphereLightSource.h"

Float SphereLightSource::intersect(const Ray &ray) const {
    Vector3f op = position - ray.o;
    Float a = dot(ray.d, ray.d);
    Float b = -2 * dot(ray.d, op);
    Float c = dot(op, op) - radius * radius;

    Float delta = b * b - 4 * a * c;
    if (delta < 0) return 0;
    Float sdelta = std::sqrt(delta);
    Float t = 0;
    Float t1, t2;
    t1 = (-b - sdelta) / (2 * a);
    t2 = (-b + sdelta) / (2 * a);
    if (t1 > 0) {
        t = t1;
        if (t2 > 0 && t2 < t) t = t2;
    } else {
        if (t2 > 0) t = t2;
    }

    return t;
}

void SphereLightSource::clear(){
    this->emission = Vec3f();
    this->KDiffuse = Vec3f();
    this->KSpecular = Vec3f();
    this->position = Point3f();
    this->radius = 0;
}