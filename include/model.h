#ifndef MODEL_H
#define MODEL_H

#include "util.h"
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include "light/sphereLightSource.h"

// Model->Scene->Mesh->Face

struct Face {

    double a, b, c, d;

    Vector3f faceNormal;

    unsigned int meshId;
    Vec3f emission;

    Vec3<int> verticesIndices;
    Vec3<int> normalsIndices;
    Vec2<int> textureCoordsIndices;

//    int mVerticesIndices[3] = {-1, -1, -1};
//    int mNormalsIndices[3] = {-1, -1, -1};
//    int mTextureCoordsIndices[3] = {-1, -1, -1};

    // {xMax, yMax, zMax}
    Vec3<int> maxVerticesIndices;
    Vec3<int> minVerticesIndices;
//    int maxVerticesIndices[3] = {-1, -1, -1};
//    int minVerticesIndices[3] = {-1, -1, -1};

    std::string materialName;
};

struct Mesh {
    std::string name;

    int numFaces;

    std::vector<Face> faces;
};

struct Material {
    std::string name;

    int illum;

    double Kd[3], Ka[3], Tf[3], Ks[3];
    Vec3f KDiffuse;
    Vec3f KSpecular;

    double Ni;
    double Ns;
};

//struct Light {
//    std::string groupname;
//    Point3f center;
//    double radius;
//    double Le[3];
//};

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
//    std::vector<LightSource> mLights;

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
        double fovy;
    }cameraparams;

    std::vector<SphereLightSource> sphereLights;
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