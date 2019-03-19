#include "render.h"

//Render::Render() {
//    srand((unsigned) time(NULL));
//}

Render::Render(Model &model, Camera &camera, const int sampleNum, Vec2<int> imageSize)
        : camera(camera), renderModel(model) {

    this->sampleNum = sampleNum;
    srand((unsigned) time(NULL));
}
//double normalPdf(Vector3f normal,double  x) {
//    return 0;
//}

void Render::run() {
    int total = camera.filmSize.x * camera.filmSize.y;
    for (unsigned int y = 0; y < camera.filmSize.y; y++) {
        for (unsigned int x = 0; x < camera.filmSize.x; x++) {
            fprintf(stderr, "\r%5.4f%%", 100. * (y * camera.filmSize.x + x) / total);

            Vec3f c;
            for (int i = 0; i < sampleNum; i++) {
                // filter
                double r1 = 2 * (RANDNUM);
                double r2 = 2 * (RANDNUM);
                double dx = r1 < 1 ? std::sqrt(r1) - 1 : 1 - std::sqrt(2 - r1);
                double dy = r2 < 1 ? std::sqrt(r2) - 1 : 1 - std::sqrt(2 - r2);

//                Vector3f d = camera.raster2World(Vector3f(x + (dx + .5) * .5, y + (dy + .5) * .5, 1.));
                Vector3f d = camera.camera2World(camera.raster2Camera(Point3f(x + (dx + .5) * .5, y + (dy + .5) * .5, 1.)) - Point3f(0,0,0));
                c = c + radiance(Ray(camera.position, d.normalize()), 0) * (1. / sampleNum);
            }
            camera.film[y * camera.filmSize.x + x] = c;
        }
    }
}

double Render::intersect(const Face &face, const Ray &ray) {
    // intersection with mesh
    Vec3f rMax, rMin;

    double faceMaxx = renderModel.scene.mVertices[face.maxVerticesIndices.x].x;
    double faceMaxy = renderModel.scene.mVertices[face.maxVerticesIndices.y].y;
    double faceMaxz = renderModel.scene.mVertices[face.maxVerticesIndices.z].z;


    rMax = Vec3f((faceMaxx - ray.o.x) / ray.d.x, (faceMaxy - ray.o.y) / ray.d.y, (faceMaxz - ray.o.z) / ray.d.z);

    double faceMinx = renderModel.scene.mVertices[face.minVerticesIndices.x].x;
    double faceMiny = renderModel.scene.mVertices[face.minVerticesIndices.y].y;
    double faceMinz = renderModel.scene.mVertices[face.minVerticesIndices.z].z;
    Vec3f temp = Vec3f((faceMinx - ray.o.x) / ray.d.x, (faceMiny - ray.o.y) / ray.d.y, (faceMinz - ray.o.z) / ray.d.z);

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
            if (dot(face.faceNormal, ray.d) == 0)
                return 0;

            double t = -(face.a * ray.o.x + face.b * ray.o.y + face.c * ray.o.z + face.d) /
                       dot(face.faceNormal, ray.d);
            Point3f pp = ray.o + ray.d * t;
            // 这里的x，y，z表示第1，2，3个顶点坐标id
            Point3f pa = renderModel.scene.mVertices[face.verticesIndices.x];
            Point3f pb = renderModel.scene.mVertices[face.verticesIndices.y];
            Point3f pc = renderModel.scene.mVertices[face.verticesIndices.z];

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

Vec3f Render::radiance(const Ray &ray, int depth) {
    double t = std::numeric_limits<double>::infinity();
    Face face;
    int id = -1;
    enum InterType {
        FACE,
        LIGHTSOURCE
    };

    InterType interType;

    // obj meshes
    for (unsigned int i = 0; i < renderModel.scene.mNumMeshes; i++) {
        Mesh mesh = renderModel.scene.mMeshes[i];
        for (unsigned int j = 0; j < mesh.numFaces; j++) {
            double tt = intersect(mesh.faces[j], ray);
            if (tt > 1e-10 && tt < t) {
                t = tt;
                id = j;
                face = mesh.faces[j];
                interType = FACE;
            }
        }
    }

    // self defined shapes such as light
    // todo
    for (unsigned int i = 0; i < renderModel.config.sphereLights.size(); i++) {
        double tt = renderModel.config.sphereLights[i].intersect(ray);
        if (tt > 1e-10 && tt < t) {
            t = tt;
            id = i;
            interType = LIGHTSOURCE;
        }
    }

    Vec3f xyzColor(0, 0, 0);
    if (id < 0)
        return xyzColor;

    // obj meshes
    if (interType == FACE) {
        Material material = renderModel.scene.mMaterials[renderModel.scene.mtlName2ID[face.materialName]];
        if (depth > 5) {
            if (depth > 10) {
                return xyzColor;
            }
            double rNum = (RANDNUM);
            if (material.KDiffuse.maxCor < rNum && material.KSpecular.maxCor < rNum) {
                return xyzColor;
            }
        }

        Point3f p = ray.o + ray.d * t;
        Vector3f n = face.faceNormal;

        if (dot(n, ray.d) > 0) n = n * -1;

        if (material.KDiffuse.maxCor > 1e-10) {

            double r1 = 2 * M_PI * (RANDNUM);
            double r2 = (RANDNUM);
            double r2s = std::sqrt(r2);

            Vector3f w = n;
            Vector3f u;
            if (std::abs(w.x) > std::abs(w.y)) {
                u = cross(Vector3f(0, 1, 0), w).normalize();
            } else {
                u = cross(Vector3f(1, 0, 0), w).normalize();
            }
            Vector3f v = cross(w, u);

            Vector3f d = (u * std::cos(r1) * r2s + v * std::sin(r1) * r2s + w * std::sqrt(1 - r2)).normalize();

            xyzColor = xyzColor + mul(material.KDiffuse, radiance(Ray(p, d), ++depth));
        }

        if (material.KSpecular.maxCor > 1e-10) {
            xyzColor = xyzColor + mul(material.KSpecular, radiance(Ray(p, ray.d - n * 2 * dot(n, ray.d)), ++depth));
        }
    }
        // other shapes
    else if (interType == LIGHTSOURCE) {
        SphereLightSource light = renderModel.config.sphereLights[id];
        xyzColor = xyzColor + light.emission;
        if (depth > 5) {
            if (depth > 10)
                return xyzColor;
            double rNum = (RANDNUM);
            if (light.KDiffuse.maxCor < rNum && light.KSpecular.maxCor < rNum) {
                return xyzColor;
            }
        }
        Point3f p = ray.o + ray.d * t;
        Vector3f n = (p - light.position).normalize();

        if (dot(n, ray.d) > 0) n = n * -1;

        if (light.KDiffuse.maxCor > 1e-10) {

            double r1 = 2 * M_PI * (RANDNUM);
            double r2 = (RANDNUM);
            double r2s = std::sqrt(r2);

            Vector3f w = n;
            Vector3f u;
            if (std::abs(w.x) > std::abs(w.y)) {
                u = cross(Vector3f(0, 1, 0), w).normalize();
            } else {
                u = cross(Vector3f(1, 0, 0), w).normalize();
            }
            Vector3f v = cross(w, u);

            Vector3f d = (u * std::cos(r1) * r2s + v * std::sin(r1) * r2s + w * std::sqrt(1 - r2)).normalize();

            xyzColor = xyzColor + mul(light.KDiffuse, radiance(Ray(p, d), ++depth));
        }

        if (light.KSpecular.maxCor > 1e-10) {
            xyzColor = xyzColor + mul(light.KSpecular, radiance(Ray(p, ray.d - n * 2 * dot(n, ray.d)), ++depth));
        }
    }

    return xyzColor;
}
/*
bool Render::rayTrace(Point3f origin, Vector3f normal, Point3f &iPointLog, Color &color, int &debugDepth) {
    debugDepth++;
//    printf("depth:%d\n", debugDepth);
    if(debugDepth > 100) return false;
    Ray gRay = generateRay(origin, normal);


    bool isLightSource;
    Point3f iPoint;
    Face iFace;

    bool iFlag = getIntersection(gRay, iPoint, iFace, isLightSource);


    if (!iFlag) return false;
//    printf("gegerate ray o = (%f, %f, %f), d = (%f, %f, %f)\n", gRay.o.x, gRay.o.y, gRay.o.z, gRay.d.x, gRay.d.y, gRay.d.z);
//    printf("intersection (%f, %f, %f)\n", iPoint.x, iPoint.y, iPoint.z);
    // if intersection is a light source return;
    if (isLightSource) {
        // return the light color
        color.r = renderModel.scene.mLights[0].Le[0];
        color.g = renderModel.scene.mLights[0].Le[1];
        color.b = renderModel.scene.mLights[0].Le[2];
        return true;
    }

    bool rFlag = rayTrace(iPoint, iFace.faceNormal, iPointLog, color, debugDepth);
    if (!rFlag) return false;

    // save the intersection of current ray
    iPointLog = iPoint;

    // if intersection is not light source
    std::string debugstr = iFace.materialName;
    printf("intersection face : %s\n", debugstr.c_str());
    double debugkd0 = renderModel.scene.mMaterials[renderModel.scene.mtlName2ID[iFace.materialName]].Kd[0];
    double debugks0 = renderModel.scene.mMaterials[renderModel.scene.mtlName2ID[iFace.materialName]].Ks[0];
    double debugkd1 = renderModel.scene.mMaterials[renderModel.scene.mtlName2ID[iFace.materialName]].Kd[1];
    double debugks1 = renderModel.scene.mMaterials[renderModel.scene.mtlName2ID[iFace.materialName]].Ks[1];
    double debugkd2 = renderModel.scene.mMaterials[renderModel.scene.mtlName2ID[iFace.materialName]].Kd[2];
    double debugks2 = renderModel.scene.mMaterials[renderModel.scene.mtlName2ID[iFace.materialName]].Ks[2];
    color.r = color.r * renderModel.scene.mMaterials[renderModel.scene.mtlName2ID[iFace.materialName]].Kd[0]
              + color.r * renderModel.scene.mMaterials[renderModel.scene.mtlName2ID[iFace.materialName]].Ks[0];
    color.g = color.g * renderModel.scene.mMaterials[renderModel.scene.mtlName2ID[iFace.materialName]].Kd[1]
              + color.g * renderModel.scene.mMaterials[renderModel.scene.mtlName2ID[iFace.materialName]].Ks[1];
    color.b = color.b * renderModel.scene.mMaterials[renderModel.scene.mtlName2ID[iFace.materialName]].Kd[2]
              + color.b * renderModel.scene.mMaterials[renderModel.scene.mtlName2ID[iFace.materialName]].Ks[2];;

    return true;
}

void Render::run() {
    double diffuse[3] = {1.f, 1.f, 1.f};
    Point3f cameraPos;
    Vector3f cameraDirect;
    int maxI = 1000;
    for (int i = 0; i < maxI; i++) {

        Color sampleColor;
        Point3f iPoint;
        int debugcnt = 0;
        bool flag = rayTrace(camera.position, camera.look - camera.position, iPoint, sampleColor, debugcnt);
        if (flag) {
            Point3f rasterPos;
            Transform world2Raster = camera.raster2Camera.inverse() * camera.camera2World.inverse();
            rasterPos = world2Raster(iPoint);
            printf("raster(%d,%d)", (int)rasterPos.y, (int)rasterPos.x);
            camera.film.at<cv::Vec3b>((int) rasterPos.y, (int) rasterPos.x) =
                    cv::Vec3b((uchar) sampleColor.b, (uchar) sampleColor.g, (uchar) sampleColor.r);
            printf("process %d/%d---b %d g %d r %d\n", i, maxI, (uchar) sampleColor.b, (uchar) sampleColor.g, (uchar) sampleColor.r);
        } else
            i--;
    }
}

Ray Render::generateRay(const Point3f &origin, const Vector3f &normal) {
    Vector3f direct = sample(normal);

    if (dot(normal, direct) < 0) {
        std::cout << "WARN: the angle of ray and normal overs 90" << std::endl;
        return Ray(origin, normal);
    }

    return Ray(origin, direct);
}

Vector3f Render::sample(const Vector3f &normal) {
    Vector3f sampleVec;

    if (normal.x != 0) {
        sampleVec.y = RANDNUM;
        sampleVec.z = RANDNUM;
        if (normal.x > 0) {
            sampleVec.x = -(sampleVec.y * normal.y + sampleVec.z * normal.z) / normal.x + 0.5;
        } else {
            sampleVec.x = -(sampleVec.y * normal.y + sampleVec.z * normal.z) / normal.x - 0.5;
        }
    } else if (normal.y != 0) {
        sampleVec.x = RANDNUM;
        sampleVec.z = RANDNUM;
        if (normal.y > 0) {
            sampleVec.y = -(sampleVec.x * normal.x + sampleVec.z * normal.z) / normal.y + 0.5;
        } else {
            sampleVec.y = -(sampleVec.x * normal.x + sampleVec.z * normal.z) / normal.y - 0.5;
        }
    } else if (normal.z != 0) {
        sampleVec.x = RANDNUM;
        sampleVec.y = RANDNUM;
        if (normal.z > 0) {
            sampleVec.z = -(sampleVec.x * normal.x + sampleVec.y * normal.y) / normal.z + 0.5;
        } else {
            sampleVec.z = -(sampleVec.x * normal.x + sampleVec.y * normal.y) / normal.z - 0.5;
        }
    } else {
        std::cout << "ERROR: invalid normal" << std::endl;
    }

    return sampleVec.normalize();
}
*/

/*
bool Render::getIntersection(const Ray &ray, Point3f &iPoint, Face &iFace, bool &isLightSource) {
    double tMin = ray.tMax;
    isLightSource = false;
    bool intersectionFlag = false;
    // intersection with mesh
    for (int i = 0; i < renderModel.scene.mNumMeshes; i++) {
        Mesh mesh = renderModel.scene.mMeshes[i];
        for (int j = 0; j < mesh.numFaces; j++) {
            Face face = mesh.faces[j];
            Vector3f rMax, rMin;

//            rMax = (renderModel.scene.mVertices[face.maxVerticesIndices[0]] - ray.o) / ray.d;
//            Vector3f temp = (renderModel.scene.mVertices[face.minVerticesIndices[0]] - ray.o) / ray.d;
double faceMaxx = renderModel.scene.mVertices[face.maxVerticesIndices[0]].x;
double faceMaxy = renderModel.scene.mVertices[face.maxVerticesIndices[1]].y;
double faceMaxz = renderModel.scene.mVertices[face.maxVerticesIndices[2]].z;
            rMax = (Point3f(faceMaxx, faceMaxy, faceMaxz) - ray.o) / ray.d;

            double faceMinx = renderModel.scene.mVertices[face.minVerticesIndices[0]].x;
            double faceMiny = renderModel.scene.mVertices[face.minVerticesIndices[1]].y;
            double faceMinz = renderModel.scene.mVertices[face.minVerticesIndices[2]].z;
            Vector3f temp = (Point3f(faceMinx, faceMiny, faceMinz) - ray.o) / ray.d;

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
                continue;
            rMin.x = std::max(rMin.x, 0.f);
            rMin.y = std::max(rMin.y, 0.f);
            rMin.z = std::max(rMin.z, 0.f);

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
                    if (dot(face.faceNormal, ray.d) == 0)
                        continue;

                    double t = -(face.a * ray.o.x + face.b * ray.o.y + face.c * ray.o.z + face.d) /
                              dot(face.faceNormal, ray.d);
                    Point3f pp = ray.o + ray.d * t;
                    Point3f pa = renderModel.scene.mVertices[face.mVerticesIndices[0]];
                    Point3f pb = renderModel.scene.mVertices[face.mVerticesIndices[1]];
                    Point3f pc = renderModel.scene.mVertices[face.mVerticesIndices[2]];

                    Vector3f vab = pb - pa;
                    Vector3f vac = pc - pa;
                    Vector3f vap = pp - pa;

                    double m = (vap.x * vac.y - vac.x * vap.y) / (vab.x * vac.y - vac.x * vab.y);
                    double n = (vab.x * vap.y - vap.x * vab.y) / (vab.x * vac.y - vac.x * vab.y);

                    if (m + n <= 1 && m >= 0 && n >= 0) {
                        if (t < tMin && t > 0.001) {
                            iFace = face;
                            iPoint = pp;
                            intersectionFlag = true;
//                            printf("DEBUG: (%f,%f,%f)+(%f,%f,%f)*%f=(%f,%f,%f)\n",ray.o.x, ray.o.y, ray.o.z,ray.d.x, ray.d.y,ray.d.z,t,pp.x,pp.y,pp.z);
                        }
                    }
                } else
                    continue;
            } else
                continue;
        }
    }

    // intersection with light source
    if (intersectionFlag)
        isLightSource = renderModel.scene.mMeshes[iFace.meshId].isLightSource;

    return intersectionFlag;
}

*/