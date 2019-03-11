#ifndef CAMERA_H
#define CAMERA_H

#include "util.h"
#include "transform.h"
#include "model.h"

#include <opencv2/opencv.hpp>

#define RANDNUM std::rand() / (float)(RAND_MAX)

class Camera {
public:
    Camera() {};

    Camera(const Transform &camera2World, const Transform &camera2Screen,
           const Bounds2f &screenWindow, cv::Mat &film);

    Camera(const Point3f position, const Vector3f up, const Point3f look, const Bounds2f &screenWindow,
           float fovy, int width = 800, int height = 600);

    Transform camera2World;
    Transform camera2Screen, raster2Camera;
    Transform screen2Raster, raster2Screen;

    cv::Mat film;

    Point3f position;
    Vector3f up;
    Point3f look;

    float fovy;

};

#endif // CAMERA_H