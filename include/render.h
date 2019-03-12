#ifndef RENDER_H
#define RENDER_H
#include <ctime>
#include <cstdlib>
#include <cmath>

#include "util.h"
#include "model.h"
#include "camera.h"

#define RANDNUM std::rand() / (float)(RAND_MAX)

struct Color{
    Color(float r = 0, float g = 0, float b = 0):r(r), g(g), b(b){}
    float r, g, b;
};

class Render
{
public:
  Render();
  Render(const Model &model, const Camera &camera);

  void run();

private:
  bool rayTrace(Point3f origin, Vector3f normal, Point3f &iPointLog, Color &color, int &debugDepth);

  Ray generateRay(const Point3f &origin, const Vector3f &normal);

  // setcolor
  bool getIntersection(const Ray &ray, Point3f &iPoint, Face &iFace, bool &isLightSource);

  // receive a pdf to .
  // todo
  Vector3f sample(const Vector3f &normal);

  // generate random numbers according to the pdf;
  float generateRandom();

  Model renderModel;

  Camera camera;
};


#endif // RENDER_H
