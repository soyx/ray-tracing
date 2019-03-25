#ifndef DMODEL_H
#define DMODEL_H

#include "util.h"
struct DFace{
    double d_a, d_b, d_c, d_d;
    Vector3f d_faceNormal;

    unsigned int d_meshId;
    Vec3f d_emission;

    Vec3<int> d_verticesIndices;
    Vec3<int> d_normalsIndices;
    Vec2<int> d_textureCoordsIndices;

    Vec3<int> d_maxVerticesIndices;
    Vec3<int> d_minVerticesIndices;

    int d_materialId;
}

struct DMesh{
    char d_name[20];
    
    int d_numFaces;
    Vec3f d_maxVertices;
    Vec3f d_minVertices;
    
    DFace *d_faces;

    bool isIntersect(const Ray &ray);
};

struct DMaterial{
    char d_name[20];

    int d_illum;

    Vec3f d_KDiffuse;
    Vec3f d_Kspecular;
    Vec3f d_Tf;

    double d_Ni;
    double d_Ns;
};

struct DScene{
    int d_mNumMeshes;
    int d_mNumMaterials;
    int d_mNumLights;

    int d_mNumVertices;
    int d_mNumNormals;
    int d_mNumTextureCoords;

    DMesh* d_mMeshes;
    DMaterial* d_mMaterials;

    Point3f* d_mVertices;
    Vector3f* d_mNormals;
    Point2f* d_mTextureCoords;
};

struct DModel{
    DScene* d_scene;
};
#endif // DMODEL_H
