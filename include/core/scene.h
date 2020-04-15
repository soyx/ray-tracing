#ifndef CORE_SCENE_H
#define CORE_SCENE_H

#include "camera.h"
#include "shape.h"
#include "util.h"

class Scene {
 public:
  Scene() = default;

  void addObject(const std::shared_ptr<Shape>& shape);
  void setCamera(const std::shared_ptr<Camera>& camera);

  void render();
  void render(int x_start, int x_length, int y_start, int y_length,
                   int sample_nums); 

 private:
  std::vector<std::shared_ptr<Shape>> shape_list;
  // std::vector<std::shared_ptr<Light>> light_list;
  std::shared_ptr<Camera> camera;

};

#endif  // CORE_SCENE_H