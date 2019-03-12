#include "util.h"
#include <iostream>
#include "model.h"
#include "camera.h"
#include "render.h"

int main(){
    int a = 0;
    Model model("../resources/Scene02/room.obj", "../resources/Scene02/room.mtl");
    std::cout << model.scene.mtlName2ID["lamp"] << std::endl;
    Bounds2f screenWindow = Bounds2f(Point2f(0,0), Point2f(800,600));

    Camera camera(model.config.cameraparams.position, model.config.cameraparams.up, model.config.cameraparams.lookat,
            screenWindow ,model.config.cameraparams.fovy);

    Render render(model, camera);

    render.run();

    cv::namedWindow("render", CV_WINDOW_AUTOSIZE);
    cv::imshow("render",camera.film);
    for(int i = 0; i < 800; i++){
        for(int j = 0; j < 600; j++){
            cv::Vec3b vi = camera.film.at<cv::Vec3b>(j, i);
            if(vi(0) != 0 || vi(1) != 0 || vi(2) != 0)
                printf("c = %d, r = %d, color=(%d, %d, %d)\n", j, i, vi(0), vi(1), vi(2));
        }
    }

    while(true){
        cv::waitKey(100);
        printf("finish\n");
    }
    return 0;
}