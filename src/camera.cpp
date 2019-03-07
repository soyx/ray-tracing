#include "camera.h"

Camera::Camera(const Transform &camera2World, const Transform &camera2Screen,
               const Bounds2f &screenWindow, cv::Mat &film)
{
    // compute the transform
    this->camera2World = camera2World;

    this->raster2Screen = scale(film.size().width, film.size().height, 1) *
                          scale(1 / (screenWindow.pMax.x - screenWindow.pMin.x),
                                1 / (screenWindow.pMin.y - screenWindow.pMax.y), 1) *
                          translate(Vector3f(-screenWindow.pMin.x, -screenWindow.pMax.y, 0));
    this->raster2Screen = screen2Raster.inverse();
    this->raster2Camera = camera2Screen.inverse() * raster2Screen;
}