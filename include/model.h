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
#include "light/quadLightSource.h"

// Model->Scene->Mesh->Face

struct Face {

    double a, b, c, d;

    Vector3f faceNormal;

    unsigned int meshId;
    Vec3f emission;

    Vec3<int> verticesIndices;
    Vec3<int> normalsIndices;
    Vec2<int> textureCoordsIndices;

    // {xMax, yMax, zMax}
    Vec3<int> maxVerticesIndices;
    Vec3<int> minVerticesIndices;

    std::string materialName;
};

struct Mesh {

    std::string name;

    int numFaces;
    Vec3f maxVertices;
    Vec3f minVertices;

    std::vector<Face> faces;

    bool isIntersect(const Ray &ray);
};

struct Material {
    std::string name;

    int illum;

    Vec3f KDiffuse;
    Vec3f KSpecular;
    Vec3f Tf = Vec3f(1,1,1);

    double Ni;
    double Ns;
};


struct Scene {
    int mNumMeshes;
    int mNumMaterials;
    int mNumLights;

    int mNumVertices;
    int mNumNormals;
    int mNumTextureCoords;

    std::vector<SphereLightSource> sphereLights;
    std::vector<QuadLightSource> quadLights;

    std::map<std::string, int> mtlName2ID;

    std::vector<Mesh> mMeshes;
    std::vector<Material> mMaterials;

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