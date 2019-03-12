#include "render.h"

Render::Render() {
    srand((unsigned) time(NULL));
}

Render::Render(const Model &model, const Camera &camera) {
    renderModel = model;
    this->camera = camera;
    srand((unsigned) time(NULL));
}

float normalPdf(Vector3f normal, float x) {
    return 0;
}

bool Render::rayTrace(Point3f origin, Vector3f normal, Point3f &iPointLog, Color &color, int &debugDepth) {
    debugDepth++;
    Ray gRay = generateRay(origin, normal);

    bool isLightSource;
    Point3f iPoint;
    Face iFace;

    bool iFlag = getIntersection(gRay, iPoint, iFace, isLightSource);

    if (!iFlag) return false;

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
    color.r = color.r * renderModel.scene.mMaterials[renderModel.scene.mtlName2ID[iFace.materialName]].Kd[0]
              + color.r * renderModel.scene.mMaterials[renderModel.scene.mtlName2ID[iFace.materialName]].Ks[0];
    color.g = color.g * renderModel.scene.mMaterials[renderModel.scene.mtlName2ID[iFace.materialName]].Kd[1]
              + color.g * renderModel.scene.mMaterials[renderModel.scene.mtlName2ID[iFace.materialName]].Ks[1];
    color.b = color.b * renderModel.scene.mMaterials[renderModel.scene.mtlName2ID[iFace.materialName]].Kd[2]
              + color.g * renderModel.scene.mMaterials[renderModel.scene.mtlName2ID[iFace.materialName]].Ks[2];;

    return true;
}

void Render::run() {
    float diffuse[3] = {1.f, 1.f, 1.f};
    Point3f cameraPos;
    Vector3f cameraDirect;
    int maxI = 10;
    for (int i = 0; i < maxI; i++) {

        Color sampleColor;
        Point3f iPoint;
        int debugcnt = 0;
        bool flag = rayTrace(camera.position, camera.look - camera.position, iPoint, sampleColor, debugcnt);
        if (flag) {
            Point3f rasterPos;
            Transform world2Raster = camera.raster2Camera.inverse() * camera.camera2World.inverse();
            rasterPos = world2Raster(iPoint);
            camera.film.at<cv::Vec3b>((int) rasterPos.y, (int) rasterPos.x) =
                    cv::Vec3b((uchar) sampleColor.b, (uchar) sampleColor.g, (uchar) sampleColor.r);
            printf("process %d/%d---b %d g %d r %d\n", i, maxI, (int) sampleColor.b, (int) sampleColor.g, (int) sampleColor.r);
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

bool Render::getIntersection(const Ray &ray, Point3f &iPoint, Face &iFace, bool &isLightSource) {
    float tMin = ray.tMax;
    isLightSource = false;
    bool intersectionFlag = false;
    // intersection with mesh
    for (int i = 0; i < renderModel.scene.mNumMeshes; i++) {
        Mesh mesh = renderModel.scene.mMeshes[i];
        for (int j = 0; j < mesh.numFaces; j++) {
            Face face = mesh.faces[j];
            Vector3f rMax, rMin;
            rMax = (renderModel.scene.mVertices[face.maxVerticesIndices[0]] - ray.o) / ray.d;
            Vector3f temp = (renderModel.scene.mVertices[face.minVerticesIndices[0]] - ray.o) / ray.d;

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

            float range1[2];
            float range2[2];
            float range3[3];
            range1[0] = rMin.x;
            range1[1] = rMax.x;
            range2[0] = rMin.y;
            range2[1] = rMax.y;
            range3[0] = rMin.z;
            range3[1] = rMax.z;

            float range12[2];
            if (range1[1] >= range2[0] && range1[0] <= range2[1]) {
                range12[0] = std::max(range1[0], range2[0]);
                range12[1] = std::max(range1[0], range2[1]);
                if (range12[1] >= range3[0] && range12[0] <= range3[1]) {
                    // detail process
                    // resolve equation
                    if (dot(face.faceNormal, ray.d) == 0)
                        continue;
                    float t = -(face.a * ray.o.x + face.b * ray.o.y + face.c * ray.o.z + face.d) /
                              dot(face.faceNormal, ray.d);
                    Point3f pp = ray.o + ray.d * t;
                    Point3f pa = renderModel.scene.mVertices[face.mVerticesIndices[0]];
                    Point3f pb = renderModel.scene.mVertices[face.mVerticesIndices[1]];
                    Point3f pc = renderModel.scene.mVertices[face.mVerticesIndices[2]];

                    Vector3f vab = pb - pa;
                    Vector3f vac = pc - pa;
                    Vector3f vap = pp - pa;

                    float m = (vap.x * vac.y - vac.x * vap.y) / (vab.x * vac.y - vac.x * vab.y);
                    float n = (vab.x * vap.y - vap.x * vab.y) / (vab.x * vac.y - vac.x * vab.y);

                    if (m + n <= 1 && m >= 0 && n >= 0) {
                        if (t < tMin) {
                            iFace = face;
                            iPoint = pp;
                            intersectionFlag = true;
                        }
                    }
                } else
                    continue;
            } else
                continue;
        }
    }

    // intersection with light source
    isLightSource = renderModel.scene.mMeshes[iFace.meshId].isLightSource;

    return intersectionFlag;
}