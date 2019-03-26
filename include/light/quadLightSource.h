#ifndef LIGHT_QUADLIGHTSOURCE_H
#define LIGHT_QUADLIGHTSOURCE_H

#include "light/lightSource.h"
#include "util.h"

class QuadLightSource : public LightSource {
   public:
    QuadLightSource() = default;

    QuadLightSource(Point3f center, Vector3f normal, Vec2f size, Vec3f emission,
                    Vec3f kd = Vec3f(), Vec3f ks = Vec3f())
        : center(center),
          normal(normal),
          size(size),
          LightSource(emission, kd, ks) {}

    ~QuadLightSource() override = default;

    double intersect(const Ray &ray) const override;

    void clear() override;

    Point3f center;
    Vector3f normal;
    Vec2f size;
};

#endif  // LIGHT_QUADLIGHTSOURCE_H