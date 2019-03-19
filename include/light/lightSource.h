#ifndef LIGHT_LIGHTSOURCE_H
#define LIGHT_LIGHTSOURCE_H

#include "util.h"

class LightSource {
public:
    LightSource()= default;

    LightSource(Vec3f emission, Vec3f kd, Vec3f ks) : emission(emission), KDiffuse(kd), KSpecular(ks) {}

    virtual ~LightSource() = default;

    virtual double intersect(const Ray &ray) const = 0;

public:
    Vec3f emission;
    Vec3f KDiffuse;
    Vec3f KSpecular;
};

#endif //LIGHT_LIGHTSOURCE_H
