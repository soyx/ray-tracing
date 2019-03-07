#ifndef CAMERA_H
#define CAMERA_H
#include "util.h"
#include "transform.h"
#include <opencv2/opencv.hpp>

class Camera
{
  public:
    Camera(const Transform &camera2World, const Transform &camera2Screen,
           const Bounds2f &screenWindow, cv::Mat &film);

    // Transform camera2World, world2Camera;
    // Transform screen2Raster, raster2Screen;
    // Transform camera2Screen, screen2Camera;
    // Transform raster2Camera, camera2Raster;

    Transform camera2World;
    Transform camera2Screen, raster2Camera;
    Transform screen2Raster, raster2Screen;

    cv::Mat *film;
};

#endif // CAMERA_H