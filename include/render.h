#ifndef RENDER_H
#define RENDER_H
#include <ctime>
#include <cstdlib>
#include <cmath>

#include "util.h"
#include "model.h"
#include "camera.h"

#define RANDNUM  std::rand() / (float)(RAND_MAX)
class Render
{
  public:
    Render();
    Render(Model &model);

  private:
    Ray generateRay(const Point3f &origin, const Vector3f &normal);

    // setcolor
    void getIntersection(const Ray &ray, Point3f &iPoint, Face &iFace);
    
    // receive a pdf to .
    // todo
    Vector3f sample(const Vector3f &normal);

    // generate random numbers according to the pdf;
    float generateRandom();

    Model renderModel;

    
};

#endif // RENDER_H
