#ifndef CORE_MODEL_H
#define CORE_MODEL_H

#include <assimp/postprocess.h>
#include <assimp/scene.h>

#include <assimp/Importer.hpp>
#include <vector>

#include "material.h"
#include "util.h"
#include "shape/triangle.h"
class Mesh {
 public:
  Mesh() = default;

  Mesh(aiMesh *ai_mesh);

  void loadFromAssimp(aiMesh *ai_mesh);

  std::shared_ptr<TriangleMesh> triangle_mesh;
  std::vector<std::shared_ptr<Shape>> tri_list;
  unsigned int material_index;
};

class MeshModel {
 public:
  MeshModel() = default;

  MeshModel(const std::string &file_path);

  void loadFile(const std::string &file_path);

  std::vector<Mesh> mesh_list;
  std::vector<Material> material_list;

 private:
  void fromAssimp(const aiScene *scene);
};

#endif  // CORE_MODEL_H