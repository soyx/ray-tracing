#include <memory>
#include <string>

#include "core/camera.h"
#include "core/model.h"
#include "core/scene.h"
#include "shape/cube.h"
#include "shape/sphere.h"
#include "shape/triangle.h"

int main() {
  Scene scene;
  MeshModel model(std::string("../resources/Scene02/room.obj"));

  for (const Mesh& mesh : model.mesh_list) {
    for (const auto& shape : mesh.tri_list) {
      scene.addObject(shape);
    }
  }

    // auto c = std::make_shared<Camera>(Point3f(50, 30, 170), Point3f(50, 30,
    // 0),
    //                                   Vector3f(0, 1, 0));
  auto c = std::make_shared<Camera>(Point3f(0, 0, 4),
                                    Point3f(0, 0, 0), Vector3f(0, 1, 0));
  c->setPerspective(70.f / 180 * PI, 0.1, 200);
  scene.setCamera(c);

  scene.render(0, 800, 0, 600, 1);

  FILE* file = fopen("image.ppm", "w");
  fprintf(file, "P3\n%d %d\n%d\n", c->film_size.x, c->film_size.y, 255);
  for (auto col : c->film) {
    fprintf(file, "%d %d %d ", (int)(255 * col.x), (int)(col.y * 255),
            (int)(col.z * 255));
  }
  fclose(file);
  return 0;
}