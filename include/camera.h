#ifndef CAMERA_H
#define CAMERA_H

#include "util.h"
#include "transform.h"
#include "model.h"

//#define RANDNUM std::rand() / (double)(RAND_MAX)

class Camera {
public:
    Camera() {};

//    Camera(const Transform &camera2World, const Transform &camera2Screen,
//           const Bounds2f &screenWindow, cv::Mat &film);

    Camera(const Point3f position, const Point3f look, const Vector3f up, const double focalLength = 5,
           double fovy = 60. / 180 * M_PI, int width = 800, int height = 600);

//    Transform camera2World;
//    Transform camera2Screen, raster2Camera;
//    Transform screen2Raster, raster2Screen;

//    Transform raster2World;
    Transform camera2World, raster2Camera;

    // cv::Mat film;

    std::vector<Vec3f> film;
    Vec2<int> filmSize;

    Point3f position;
private:
    Transform raster2Screen, screen2Camera;

    Vector3f up;
    Point3f look;
    double fovy;
    double focalLength;

};

#endif // CAMERA_H