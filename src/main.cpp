#include "util.h"
#include <iostream>
#include "model.h"
#include "camera.h"
#include "render.h"
#include <ctime>

inline double clamp(double x) { return x < 0 ? 0 : x > 1 ? 1 : x; }

inline int toInt(double x) {
    return int(pow(clamp(x), 1 / 2.2) * 255 + .5);
}

//inline int toInt(double x){
//    return int(x <0 ? 0 : x > 255 ? 255 : x);
//}

int main(int argc, char *argv[]) {
    time_t start = time(NULL);
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
    Model model(path);
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
    time_t stop = time(NULL);
    time_t duration = stop-start;
    printf("samplenum:%d,time:%ldd%ldh%ldm%lds\n", sampleNum, duration/(24*3600), duration%(24*3600)/3600, duration%3600/60, duration%60);
    return 0;
}