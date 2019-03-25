#ifndef LIGHT_LIGHTSOURCE_H
#define LIGHT_LIGHTSOURCE_H

#include "util.h"

class LightSource {
public:
    __host__ __device__ LightSource()= default;

    __host__ __device__ LightSource(Vec3f emission, Vec3f kd, Vec3f ks) : emission(emission), KDiffuse(kd), KSpecular(ks) {}

    __host__ __device__ virtual ~LightSource() = default;

    __host__ __device__ virtual double intersect(const Ray &ray) const = 0;

public:
    Vec3f emission;
    Vec3f KDiffuse;
    Vec3f KSpecular;
};

#endif //LIGHT_LIGHTSOURCE_H
