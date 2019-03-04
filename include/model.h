#ifndef MODEL_H
#define MODEL_H

#include "util.h"
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <map>

// Model->Scene->Mesh->Face

struct Face{

    int mVerticesIndices[3] = {-1, -1, -1};
    int mNormalsIndices[3] = {-1, -1, -1};
    int mTextureCoordsIndices[3] = {-1, -1, -1};

    std::string  materialName;
};

struct Mesh
{
    std::string name;

    int mNumVertices;
    int mNumNormals;
    int mNumTextureCoords;
    int numFaces;

    std::vector<Vector3f> mVertices;
    std::vector<Vector3f> mNormals;
    std::vector<Vector2f> mTextureCoords;

    std::vector<Face> faces;

};

struct Material
{
    std::string name;

    int illum;

    Vector3f Kd, Ka, Tf, Ks;

    float Ni;
    float Ns;
};

struct Light
{
};

struct Scene
{
    int mNumMeshes;
    int mNumMaterials;
    int mNumLights;

    std::map<std::string, int> mtlName2ID;

    std::vector<Mesh> mMeshes;
    std::vector<Material> mMaterials;
    std::vector<Light> mLights;
};

class Model
{
  public:
    Model();
    Model(std::string objPath, std::string mtlPath = NULL);

    bool load(std::string objpath, std::string mtlPath = NULL);

    Scene scene;

    private:
    bool loadObj(std::string objPath);

    bool loadMtl(std::string mtlPath);
};

#endif // MODEL_H