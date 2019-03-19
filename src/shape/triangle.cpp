#include "shape/triangle.h"

Triangle::Triangle() {}

double Triangle::intersect(const Ray &ray) const {
    // intersection with mesh
    Vec3f rMax, rMin;

    double triMaxx = vertices[maxVerticesIndices.x].x;
    double triMaxy = vertices[maxVerticesIndices.y].y;
    double triMaxz = vertices[maxVerticesIndices.z].z;
    rMax = Vec3f(triMaxx - ray.o.x, triMaxy - ray.o.y, triMaxz - ray.o.z) / ray.d;

    double triMinx = vertices[maxVerticesIndices.x].x;
    double triMiny = vertices[maxVerticesIndices.y].y;
    double triMinz = vertices[maxVerticesIndices.z].z;
    Vec3f temp = Vec3f(triMinx - ray.o.x, triMiny - ray.o.y, triMinz - ray.o.z) / ray.d;

    if (temp.x > rMax.x) {
        rMin.x = rMax.x;
        rMax.x = temp.x;
    } else
        rMin.x = temp.x;

    if (temp.y > rMax.y) {
        rMin.y = rMax.y;
        rMax.y = temp.y;
    } else
        rMin.y = temp.y;

    if (temp.z > rMax.z) {
        rMin.z = rMax.z;
        rMax.z = temp.z;
    } else
        rMin.z = temp.z;

    if (rMax.x <= 0 || rMax.y <= 0 || rMax.z <= 0)
        return 0;
    rMin.x = std::max(rMin.x, 0.);
    rMin.y = std::max(rMin.y, 0.);
    rMin.z = std::max(rMin.z, 0.);

    double range1[2];
    double range2[2];
    double range3[2];
    range1[0] = rMin.x;
    range1[1] = rMax.x;
    range2[0] = rMin.y;
    range2[1] = rMax.y;
    range3[0] = rMin.z;
    range3[1] = rMax.z;

    double range12[2];
    if (range1[1] >= range2[0] && range1[0] <= range2[1]) {
        range12[0] = std::max(range1[0], range2[0]);
        range12[1] = std::max(range1[0], range2[1]);
        if (range12[1] >= range3[0] && range12[0] <= range3[1]) {
            // detail process
            // resolve equation
            if (dot(faceNormal, ray.d) == 0)
                return 0;

            double t = -(a * ray.o.x + b * ray.o.y + c * ray.o.z + d) /
                       dot(faceNormal, ray.d);
            Point3f pp = ray.o + ray.d * t;
            Point3f pa = vertices[0];
            Point3f pb = vertices[1];
            Point3f pc = vertices[2];

            Vector3f vab = pb - pa;
            Vector3f vac = pc - pa;
            Vector3f vap = pp - pa;
            double m = -1, n = -1;
            if (vab.x * vac.y - vac.x * vab.y != 0) {
                m = (vap.x * vac.y - vac.x * vap.y) /
                    (vab.x * vac.y - vac.x * vab.y);
                n = (vab.x * vap.y - vap.x * vab.y) /
                    (vab.x * vac.y - vac.x * vab.y);
            } else if (vab.x * vac.z - vac.x * vab.z != 0) {
                m = (vap.x * vac.z - vac.x * vap.z) /
                    (vab.x * vac.z - vac.x * vab.z);
                n = (vab.x * vap.z - vap.x * vab.z) /
                    (vab.x * vac.z - vac.x * vab.z);
            } else if (vab.y * vac.z - vac.y * vab.z != 0) {
                m = (vap.y * vac.z - vac.y * vap.z) /
                    (vab.y * vac.z - vac.y * vab.z);
                n = (vab.y * vap.z - vap.y * vab.z) /
                    (vab.y * vac.z - vac.y * vab.z);
            }

            if (m + n <= 1 && m >= 0 && n >= 0) {
                return t;
            }
        }
    }
    return 0;
}