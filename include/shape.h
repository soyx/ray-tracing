#ifndef SHAPE_H
#define SHAPE_H

#include "util.h"

class Shape {
public:

    Shape() {}

    Shape(Vec3f emission, Vec3f kd, Vec3f ks) : emission(emission), KDiffuse(kd), KSpecular(ks) {}

    virtual double intersect(const Ray &ray) const = 0;

    Vec3f emission;
    Vec3f KDiffuse;
    Vec3f KSpecular;
};

#endif //SHAPE_H
