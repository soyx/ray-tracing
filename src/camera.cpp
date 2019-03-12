#include "camera.h"

Camera::Camera(const Transform &camera2World, const Transform &camera2Screen,
               const Bounds2f &screenWindow, cv::Mat &film) {
    // compute the transform
    this->camera2World = camera2World;

    this->screen2Raster = scale(film.size().width, film.size().height, 1) *
                          scale(1 / (screenWindow.pMax.x - screenWindow.pMin.x),
                                1 / (screenWindow.pMin.y - screenWindow.pMax.y), 1) *
                          translate(Vector3f(-screenWindow.pMin.x, -screenWindow.pMax.y, 0));
    this->raster2Screen = screen2Raster.inverse();
    this->raster2Camera = camera2Screen.inverse() * raster2Screen;
}

Camera::Camera(const Point3f position, const Vector3f up, const Point3f look, const Bounds2f &screenWindow,
               float fovy, int rasterWidth, int rasterHeight) {
    this->position = position;
    this->up = up;
    this->look = look;
    this->fovy = fovy;

    this->camera2World = lookAt(position, look, up);

    this->screen2Raster = scale(rasterWidth, rasterHeight, 1) *
                          scale(1 / (screenWindow.pMax.x - screenWindow.pMin.x),
                                1 / (screenWindow.pMin.y - screenWindow.pMax.y), 1) *
                          translate(Vector3f(-screenWindow.pMin.x, -screenWindow.pMax.y, 0));
    this->raster2Screen = screen2Raster.inverse();
    camera2Screen = perspective(fovy, 1e-2f, 10.f);
    this->raster2Camera = camera2Screen.inverse() * raster2Screen;
    film.create(rasterHeight, rasterWidth, CV_8UC3);
    for(int c = 0; c < rasterWidth; c++){
        for(int r = 0; r < rasterHeight; r++){
            film.at<cv::Vec3b>(r,c) = cv::Vec3b(0,0,0);
        }
    }

    srand((unsigned) time(NULL));
}


