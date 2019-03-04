#include "util.h"
#include <iostream>
#include "model.h"

int main(){
    int a = 0;
    Model model("../resources/Scene02/room.obj", "../resources/Scene02/room.mtl");
    std::cout << model.scene.mtlName2ID["lamp"] << std::endl;
    return 0;
}