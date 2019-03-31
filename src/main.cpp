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

struct Time{
    int d, h, m, s;
};

Time getTime(time_t duration){
    Time t;
    t.d = duration/(3600*24);
    t.h = duration%(3600*24)/3600;
    t.m = duration%(3600)/60;
    t.s = duration%(60);
    return t;
}
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
    printf("*****report*****\n");
    printf("**samplenum:%d\n", sampleNum);
    Time tTime = getTime(duration);
    printf("**total time:%dd%dh%dm%ds\n", tTime.d,tTime.h,tTime.m,tTime.s);
//    tTime = getTime(model.loadTime);
//    printf("****load time:%dd%dh%dm%ds\n", tTime.d,tTime.h,tTime.m,tTime.s);
//    tTime = getTime(render.renderTime);
//    printf("****render time:%dd%dh%dm%ds\n", tTime.d,tTime.h,tTime.m,tTime.s);
//    tTime = getTime(render.interTime);
//    printf("*******intersect time:%dd%dh%dm%ds\n", tTime.d,tTime.h,tTime.m,tTime.s);
//    tTime = getTime(render.renderTime - render.interTime);
//    printf("*******tracing time:%dd%dh%dm%ds\n", tTime.d,tTime.h,tTime.m,tTime.s);
    
    return 0;
}