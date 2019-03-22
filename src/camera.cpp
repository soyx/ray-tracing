#include "camera.h"

Camera::Camera(const Point3f position, const Point3f look, const Vector3f up, const double focalLength,
               double fovy, int rasterWidth, int rasterHeight) {
    this->position = position;
    this->up = up;
    this->look = look;
    this->fovy = fovy;

    filmSize.x = rasterWidth;
    filmSize.y = rasterHeight;
    film.resize((unsigned long) rasterHeight * rasterWidth);

    double m = std::min(rasterHeight, rasterWidth);

    raster2Screen = scale(2 / m, -2 / m, -1) * translate(Vector3f(-0.5 * rasterWidth, -0.5 * rasterHeight, 0));

    double alpha = (focalLength * std::tan(fovy * 0.5));
    screen2Camera = scale(alpha, alpha, focalLength);

    Vector3f cx, cy, cz;
    cy = up.normalize();
    cz = (look - position).normalize()*-1;
    cx = cross(cy, cz);

    camera2World = Transform(Mat4(cx.x, cy.x, cz.x, 0,
                                  cx.y, cy.y, cz.y, 0,
                                  cx.z, cy.z, cz.z, 0,
                                  0, 0, 0, 1));

    raster2Camera = screen2Camera * raster2Screen;
}
