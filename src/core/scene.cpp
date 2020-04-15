#include "core/scene.h"

#include "core/random.h"

void Scene::addObject(const std::shared_ptr<Shape> &shape) {
  shape_list.push_back(shape);
}

void Scene::setCamera(const std::shared_ptr<Camera> &camera) {
  this->camera = std::move(camera);
}

void Scene::render() {
  Random random;
  int sample_nums = 20;
  for (int col = 0; col < camera->film_size.x; ++col) {
    for (int row = 0; row < camera->film_size.y; ++row) {
      int x = col, y = camera->film_size.y - row - 1;

      Vec3f color(0.f, 0.f, 0.f);
      for (int i = 0; i < sample_nums; ++i) {
        // Ray ray = computeEyeRay(x + random.uniformRandom(),
        //                         y + random.uniformRandom(), *camera);
        Ray ray = camera->generateRay(x + random.uniformRandom(),
                                      y + random.uniformRandom());
        for (auto s : shape_list) {
          SurfaceData s_data;
          if (s->intersect(ray, s_data)) {
            color = color + Vec3f((s_data.normal.x + 1) * 0.5,
                                  (s_data.normal.y + 1) * 0.5,
                                  (s_data.normal.z + 1) * 0.5) /
                                sample_nums;
            // color = color + Vec3f(std::abs(s_data.normal.x),
            //                       std::abs(s_data.normal.y),
            //                       std::abs(s_data.normal.z)) /
            //                     sample_nums;
          }
        }
      }

      camera->writeColor(row, col, color);

      //         d = (camera->film _to_camera(Point3f(x, y, 0.f)) - Point3f(0,
      //         0, 0)).normalize();
      //      d = (camera->project_transform.inverse()(Point3f(-1, -1, 1)) -
      //           Point3f(0, 0, 0))
      //              .normalize();
      //   camera->writeColor(
      //       row, col, Vec3f(0.5 * (d.x + 1), 0.5 * (d.y + 1), 0.5 * (d.z +
      //       1)));
    }
  }
}

void Scene::render(int x_start, int x_length, int y_start, int y_length,
                   int sample_nums) {
  Random random;
  x_start = x_start < 0 ? 0 : x_start;
  y_start = y_start < 0 ? 0 : y_start;
  for (int col = x_start; col < x_start + x_length && col < camera->film_size.x;
       ++col) {
    for (int row = camera->film_size.y - y_start - y_length;
         row < camera->film_size.y - y_start && row < camera->film_size.y;
         ++row) {
      int x = col, y = camera->film_size.y - row - 1;

      Vec3f color(0.f, 0.f, 0.f);
      for (int i = 0; i < sample_nums; ++i) {
        Ray ray = camera->generateRay(x + random.uniformRandom(),
                                      y + random.uniformRandom());
        bool intersect = false;
        SurfaceData s_data;
        for (auto s : shape_list) {
          // SurfaceData s_data;
          if (s->intersect(ray, s_data)) intersect = true;

          // if (s->intersect(ray, s_data)) {
          //   color = color + Vec3f((s_data.normal.x + 1) * 0.5,
          //                         (s_data.normal.y + 1) * 0.5,
          //                         (s_data.normal.z + 1) * 0.5) /
          //                       sample_nums;
          // }
        }
        if (intersect) {
          color = color + Vec3f((s_data.normal.x + 1) * 0.5,
                                (s_data.normal.y + 1) * 0.5,
                                (s_data.normal.z + 1) * 0.5) /
                              sample_nums;
        }
      }

      camera->writeColor(row, col, color);
    }
  }
}