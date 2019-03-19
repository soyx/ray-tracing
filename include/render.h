#ifndef RENDER_H
#define RENDER_H

#include <ctime>
#include <cstdlib>
#include <cmath>

#include "util.h"
#include "model.h"
#include "camera.h"

#define RANDNUM std::rand() / (double)(RAND_MAX)

struct Color {
    Color(double r = 0, double g = 0, double b = 0) : r(r), g(g), b(b) {}

    double r, g, b;
};

class Render {
public:
//    Render();

    Render(Model &model, Camera &camera, const int sampleNum = 50,
           const Vec2<int> imageSize = Vec2<int>(800, 600));

    void run();

private:

    double intersect(const Face &face, const Ray &ray);

    Vec3f radiance(const Ray &ray, int depth);

//  bool rayTrace(Point3f origin, Vector3f normal, Point3f &iPointLog, Color &color, int &debugDepth);

//  Ray generateRay(const Point3f &origin, const Vector3f &normal);

    // setcolor
//  bool getIntersection(const Ray &ray, Point3f &iPoint, Face &iFace, bool &isLightSource);

    // receive a pdf to .
    // todo
//  Vector3f sample(const Vector3f &normal);

    // generate random numbers according to the pdf;
//  double generateRandom();

    Model &renderModel;

    Camera &camera;

    int sampleNum;
};


#endif // RENDER_H
