#include "util.h"
#include <iostream>
#include "model.h"
#include "camera.h"
#include "render.h"

inline double clamp(double x) { return x < 0 ? 0 : x > 1 ? 1 : x; }

inline int toInt(double x) {
    return int(pow(clamp(x), 1 / 2.2) * 255 + .5);
}

//inline int toInt(double x){
//    return int(x <0 ? 0 : x > 255 ? 255 : x);
//}

int main(int argc, char *argv[]) {
    std::string filename = "image.ppm";
    int sampleNum = 10;
    int index = 1;
    std::string path;
    for(int i = 0; i < argc; i++){
        if(std::string(argv[i]) == "-o" && i + 1 <= argc){
            i++;
            filename = std::string(argv[i]);
            continue;
        }
        if(std::string(argv[i]) == "-n" && i + 1 <= argc){
            i++;
            sampleNum = atoi(argv[i]);
            continue;
        }

        path = std::string(argv[i]);
    }
    // if (argc >= 2) {
    //     if(std::string(argv[index]) == "-o" && argc >= index + 2){
    //         index++;
    //         filename = std::string(argv[index++]);
    //     }
    //     else if(std::string(argv[index]) == "-n" && argc >= index + 2){
    //         index++;
    //         sampleNum = atoi(argv[index++]);
    //     }
    //     else{
    //         path = std::string(argv[index++]);
    //         std::cout << "objpath:" << path << std::endl;
    //     }
    // }
    // Model model("../resources/Scene02/room.obj", "../resources/Scene02/room.mtl");
    Model model(path);
//    Camera camera(Point3f(50, 60,160), Vector3f(0,1,0), Point3f(50 ,30, 0), 100);
//    Camera camera(Point3f(50, 45, 170), Point3f(50, 30, 0), Vector3f(0, 1, 0), 10);

    // int width = 800;
    // int height = 600;
    double focal = 1000.;

    Camera camera(model.config.cameraparams.position, model.config.cameraparams.lookat, model.config.cameraparams.up,
                  focal, model.config.cameraparams.fovy * 1.0 / 180 * M_PI, model.config.resolution.width, model.config.resolution.height);
    Render render(model, camera, sampleNum);

    render.run();

    FILE *file = fopen(filename.c_str(), "w");
    fprintf(file, "P3\n%d %d\n%d\n", model.config.resolution.width, model.config.resolution.height, 255);
    for (auto c : camera.film) {
        fprintf(file, "%d %d %d ", toInt(c.x), toInt(c.y), toInt(c.z));
    }
    return 0;
}