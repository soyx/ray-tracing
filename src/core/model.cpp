#include "core/model.h"

#include "core/transform.h"
#include "shape/triangle.h"

MeshModel::MeshModel(const std::string &file_path) { loadFile(file_path); }

void MeshModel::loadFile(const std::string &file_path) {
  Assimp::Importer importer;

  const aiScene *scene = importer.ReadFile(
      file_path, aiProcess_Triangulate | aiProcess_GenNormals);

  fromAssimp(scene);
}
void MeshModel::fromAssimp(const aiScene *scene) {
  // Material
  for (int i = 0; i < scene->mNumMaterials; ++i) {
    Material material(scene->mMaterials[i]);
    material_list.push_back(std::move(material));
  }

  // Mesh
  for (int i = 0; i < scene->mNumMeshes; ++i) {
    Mesh mesh;
    mesh.loadFromAssimp(scene->mMeshes[i]);
    mesh_list.push_back(std::move(mesh));
  }
}

Mesh::Mesh(aiMesh *ai_mesh) { loadFromAssimp(ai_mesh); }

void Mesh::loadFromAssimp(aiMesh *ai_mesh) {
  if (ai_mesh->mPrimitiveTypes != aiPrimitiveType_TRIANGLE) return;

  std::vector<Point3f> vertex_list;
  std::vector<Normal3f> normal_list;

  // vertices & normal
  for (int i = 0; i < ai_mesh->mNumVertices; ++i) {
    vertex_list.emplace_back(ai_mesh->mVertices[i].x, ai_mesh->mVertices[i].y,
                             ai_mesh->mVertices[i].z);
    normal_list.emplace_back(ai_mesh->mNormals[i].x, ai_mesh->mNormals[i].y,
                             ai_mesh->mNormals[i].z);
  }

  std::vector<int> vertex_indices;
  for (int i = 0; i < ai_mesh->mNumFaces; ++i) {
    vertex_indices.push_back(ai_mesh->mFaces[i].mIndices[0]);
    vertex_indices.push_back(ai_mesh->mFaces[i].mIndices[1]);
    vertex_indices.push_back(ai_mesh->mFaces[i].mIndices[2]);
  }
  triangle_mesh = std::make_shared<TriangleMesh>(
      Transform::Identity(), ai_mesh->mNumFaces, std::move(vertex_indices),
      std::move(vertex_list), std::move(normal_list));

  for (int i = 0; i < ai_mesh->mNumFaces; ++i) {
    tri_list.push_back(std::make_shared<Triangle>(triangle_mesh, i));
  }

  material_index = ai_mesh->mMaterialIndex;
}