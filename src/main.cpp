#include "util.h"
#include <iostream>
#include "model.h"
#include "camera.h"
#include "render.h"

//inline double clamp(double x) { return x < 0 ? 0 : x > 1 ? 1 : x; }
//
//inline int toInt(double x) {
//    return int(pow(clamp(x), 1 / 2.2) * 255 + .5);
//}

inline int toInt(double x){
    return int(x <0 ? 0 : x > 255 ? 255 : x);
}

int main(int argc, char *argv[]) {
    std::string filename = "image.ppm";
    if (argc >= 2) {
        filename = std::string(argv[1]);
    }
    Model model("../resources/Scene02/room.obj", "../resources/Scene02/room.mtl");
//    Camera camera(Point3f(50, 60,160), Vector3f(0,1,0), Point3f(50 ,30, 0), 100);
//    Camera camera(Point3f(50, 45, 170), Point3f(50, 30, 0), Vector3f(0, 1, 0), 10);

    int width = 800;
    int height = 600;
    double fovy = 60.0 / 180 * M_PI;
    double focal = 5;

    Camera camera(model.config.cameraparams.position, model.config.cameraparams.lookat, model.config.cameraparams.up,
                  focal, fovy, width, height);
    Render render(model, camera, 5);

    render.run();

    FILE *file = fopen(filename.c_str(), "w");
    fprintf(file, "P3\n%d %d\n%d\n", width, height, 255);
    for (auto c : camera.film) {
        fprintf(file, "%d %d %d ", toInt(c.x), toInt(c.y), toInt(c.z));
    }
    return 0;
}