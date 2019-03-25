#include "light/sphereLightSource.h"

 __host__ __device__ double SphereLightSource::intersect(const Ray &ray) const {
    Vector3f op = position - ray.o;
    double a = dot(ray.d, ray.d);
    double b = -2 * dot(ray.d, op);
    double c = dot(op, op) - radius * radius;

    double delta = b * b - 4 * a * c;
    if (delta < 0) return 0;
    double sdelta = std::sqrt(delta);
    double t = 0;
    double t1, t2;
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
