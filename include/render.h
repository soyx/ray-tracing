#ifndef RENDER_H
#define RENDER_H

#include <ctime>
#include <cstdlib>
#include <cmath>

#include "util.h"
#include "model.h"
#include "camera.h"
// #include <omp.h>

#define RANDNUM std::rand() / (double)(RAND_MAX)

struct Color {
    Color(double r = 0, double g = 0, double b = 0) : r(r), g(g), b(b) {}

    double r, g, b;
};

class Render {
public:

    time_t interTime = 0, renderTime = 0;

    Render(Model &model, Camera &camera, int sampleNum = 50);

    void run();
    void run(int x1, int x2, int y1, int y2);

private:

    double intersect(const Face &face, const Ray &ray);

    Vec3f radiance(const Ray &ray, int depth);

    Model &renderModel;

    Camera &camera;

    int sampleNum;
};


#endif // RENDER_H
