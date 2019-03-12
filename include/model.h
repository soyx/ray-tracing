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




struct Face {

    float a, b, c, d;

    Vector3f faceNormal;

    unsigned int meshId;

    int mVerticesIndices[3] = {-1, -1, -1};
    int mNormalsIndices[3] = {-1, -1, -1};
    int mTextureCoordsIndices[3] = {-1, -1, -1};

    // {xMax, yMax, zMax}
    int maxVerticesIndices[3] = {-1, -1, -1};
    int minVerticesIndices[3] = {-1, -1, -1};

    std::string materialName;
};

struct Mesh {
    std::string name;

    int numFaces;

    bool isLightSource = false;

    std::vector<Face> faces;
};

struct Material {
    std::string name;

    int illum;

    float Kd[3], Ka[3], Tf[3], Ks[3];

    float Ni;
    float Ns;
};

struct Light {
    std::string groupname;
    Point3f center;
    float radius;
    float Le[3];
};

struct Scene {
    int mNumMeshes;
    int mNumMaterials;
    int mNumLights;

    int mNumVertices;
    int mNumNormals;
    int mNumTextureCoords;

    std::map<std::string, int> mtlName2ID;

    std::vector<Mesh> mMeshes;
    std::vector<Material> mMaterials;
    std::vector<Light> mLights;

    std::vector<Point3f> mVertices;
    std::vector<Vector3f> mNormals;
    std::vector<Point2f> mTextureCoords;
};

struct Config{
    struct resolution{
        int width;
        int height;
    }resolution;

    struct cameraparams{
        Point3f position;
        Point3f lookat;
        Vector3f up;
        float fovy;
    }cameraparams;

    Light light;
};

class Model {
public:
    Model();

    Model(std::string objPath, std::string mtlPath = "");

    bool load(std::string objpath, std::string mtlPath = "", std::string cfgPath = "");

    Scene scene;

    Config config;

private:
    bool loadObj(std::string objPath);

    bool loadMtl(std::string mtlPath);

    bool loadCfg(std::string cfgPath);

    void getMaxIndices(Face &face, const Mesh &mesh);

    void getMinIndices(Face &face, const Mesh &mesh);

    void computeFaceNormal(Face &face, const Mesh &mesh);
};




#endif // MODEL_H