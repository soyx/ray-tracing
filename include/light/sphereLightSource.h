#ifndef LIGHT_SPHERELIGHTSOURCE_H
#define LIGHT_SPHERELIGHTSOURCE_H

#include "lightSource.h"
#include "util.h"

class SphereLightSource : public LightSource {
public:
    SphereLightSource()= default;

    SphereLightSource(double r, Point3f p, Vec3f emission, Vec3f kd = Vec3f(), Vec3f ks = Vec3f())
            : radius(r), position(p), LightSource(emission, kd, ks) {}

    ~SphereLightSource() override = default;

    double intersect(const Ray &ray) const override;

    void clear() override;

    Point3f position;
    double radius;
};

#endif //LIGHT_SPHERELIGHTSOURCE_H
