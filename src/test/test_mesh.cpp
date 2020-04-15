#include <memory>

#include "core/camera.h"
#include "core/scene.h"
#include "shape/cube.h"
#include "shape/sphere.h"
#include "shape/triangle.h"

int main() {
  Scene scene;

  std::vector<int> vertex_indices{0, 1, 2};
  std::vector<Point3f> points{Point3f(0, 0, -1), Point3f(1, 0, -1),
                              Point3f(0, 1, -1)};
  std::vector<Normal3f> normals{Normal3f(1, -1, -1).normalize(),
                                Normal3f(-1, 1, -1).normalize(),
                                Normal3f(-1, -1, 1).normalize()};
  std::shared_ptr<TriangleMesh> mesh_ptr = std::make_shared<TriangleMesh>(
      Transform::Identity(), 1, std::move(vertex_indices), std::move(points),
      std::move(normals));

  std::shared_ptr<Shape> triangle = std::make_shared<Triangle>(mesh_ptr, 0);

  scene.addObject(triangle);

  std::shared_ptr<Shape> cube_ptr;
  cube_ptr = std::make_shared<Cube>(Point3f(-1.8, -1.5, -4), 0.8f);
  scene.addObject(cube_ptr);

  std::shared_ptr<Shape> sphere_ptr;
  sphere_ptr = std::make_shared<Sphere>(Point3f(0, 0, -2), 0.5f);
  scene.addObject(sphere_ptr);

  auto c = std::make_shared<Camera>(Point3f(0, 0, 4), Point3f(0, 0, -1),
                                    Vector3f(0, 1, 0));
  c->setPerspective(70.f / 180 * PI, 0.1, 100);
  scene.setCamera(c);

  scene.render(0, 800, 0, 600, 10);

  FILE *file = fopen("image.ppm", "w");
  fprintf(file, "P3\n%d %d\n%d\n", c->film_size.x, c->film_size.y, 255);
  for (auto col : c->film) {
    fprintf(file, "%d %d %d ", (int)(255 * col.x), (int)(col.y * 255),
            (int)(col.z * 255));
  }
  fclose(file);
  return 0;
}