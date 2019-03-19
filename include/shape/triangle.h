#ifndef SHAPE_TRIANGLE_H
#define SHAPE_TRIANGLE_H

#include "shape.h"
#include "util.h"

#include <vector>
#include <string>


class Triangle : public Shape {
public:
    Triangle();

    double a, b, c, d;

    Vector3f faceNormal;

    std::string materialName;

    std::vector<Point3f> vertices;
    std::vector<Vector3f> normals;

    Vec3<int> maxVerticesIndices = Vec3<int>(-1, -1, -1);
    Vec3<int> minVerticesIndices = Vec3<int>(-1, -1, -1);


    Vec3f emission;
    Vec3f KDiffuse;
    Vec3f Kspecular;
    // 折射 todo

    double intersect(const Ray &ray) const;

};

#endif //SHAPE_TRACING_TRIANGLE_H
