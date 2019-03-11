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

void Render::rayTrace(Point3f origin, Vector3f normal, Point3f &iPointLog, Color &color) {
    Ray gRay = generateRay(origin, normal);

    bool isLightSource;
    Point3f iPoint;
    Face iFace;

    getIntersection(gRay, iPoint, iFace, isLightSource);

    // if intersection is a light source return;
    if (isLightSource) {
        // return the light color
        color.r = renderModel.scene.mLights[0].Le[0];
        color.g = renderModel.scene.mLights[0].Le[1];
        color.b = renderModel.scene.mLights[0].Le[2];
        return;
    }

    rayTrace(iPoint, iFace.faceNormal, iPointLog, color);

    // save the intersection of current ray
    iPointLog = iPoint;

    // if intersection is not light source
    color.r *= renderModel.scene.mMaterials[renderModel.scene.mtlName2ID[iFace.materialName]].Kd[0];
    color.g *= renderModel.scene.mMaterials[renderModel.scene.mtlName2ID[iFace.materialName]].Kd[1];
    color.b *= renderModel.scene.mMaterials[renderModel.scene.mtlName2ID[iFace.materialName]].Kd[2];
}

void Render::run() {
    float diffuse[3] = {1.f, 1.f, 1.f};
    Point3f cameraPos;
    Vector3f cameraDirect;
    // 现在看来，camera需要有自己的generateRay以保存从camera出发光线与场景交点的颜色，trace第一个交点之后的光线
    // todo
    Color sampleColor;
    Point3f iPoint;
    rayTrace(camera.position, camera.look - camera.position, iPoint, sampleColor);
    Point3f rasterPos;
    rasterPos = (camera.raster2Camera.inverse() * camera.camera2World.inverse())(iPoint);
    camera.film.at<cv::Vec3i>((int) rasterPos.y, (int) rasterPos.x) =
            cv::Vec3i((int) sampleColor.b, (int) sampleColor.g, (int) sampleColor.r);
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
            sampleVec.x = -(sampleVec.y * normal.y + sampleVec.z * normal.z) + 0.5;
        } else {
            sampleVec.x = -(sampleVec.y * normal.y + sampleVec.z * normal.z) - 0.5;
        }
    } else if (normal.y != 0) {
        sampleVec.x = RANDNUM;
        sampleVec.z = RANDNUM;
        if (normal.y > 0) {
            sampleVec.y = -(sampleVec.x * normal.x + sampleVec.z * normal.z) + 0.5;
        } else {
            sampleVec.y = -(sampleVec.x * normal.x + sampleVec.z * normal.z) - 0.5;
        }
    } else if (normal.z != 0) {
        sampleVec.x = RANDNUM;
        sampleVec.y = RANDNUM;
        if (normal.z > 0) {
            sampleVec.z = -(sampleVec.x * normal.x + sampleVec.y * normal.y) + 0.5;
        } else {
            sampleVec.z = -(sampleVec.x * normal.x + sampleVec.y * normal.y) - 0.5;
        }
    } else {
        std::cout << "ERROR: invalid normal" << std::endl;
    }

    return sampleVec;
}

void Render::getIntersection(const Ray &ray, Point3f &iPoint, Face &iFace, bool &isLightSource) {
    float tMin = ray.tMax;
    isLightSource = false;
    // intersection with mesh
    for (int i = 0; i < renderModel.scene.mNumMeshes; i++) {
        Mesh mesh = renderModel.scene.mMeshes[i];
        for (int j = 0; j < mesh.numFaces; j++) {
            Face face = mesh.faces[j];
            Vector3f rMax, rMin;
            rMax = (mesh.mVertices[face.maxVecticesIndices[0]] - ray.o) / ray.d;
            Vector3f temp = (mesh.mVertices[face.minVecticesIndices[0]] - ray.o) / ray.d;

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
                    // tode
                    if (dot(face.faceNormal, ray.d) == 0)
                        continue;
                    float t = -(face.a * ray.o.x + face.b * ray.o.y + face.c * ray.o.z + face.d) /
                              dot(face.faceNormal, ray.d);
                    Point3f pp = ray.o + ray.d * t;
                    Point3f pa = mesh.mVertices[face.mVerticesIndices[0]];
                    Point3f pb = mesh.mVertices[face.mVerticesIndices[1]];
                    Point3f pc = mesh.mVertices[face.mVerticesIndices[2]];

                    Vector3f vab = pb - pa;
                    Vector3f vac = pc - pa;
                    Vector3f vap = pp - pa;

                    float m = (vap.x * vac.y - vac.x * vap.y) / (vab.x * vac.y - vac.x * vab.y);
                    float n = (vab.x * vap.y - vap.x * vab.y) / (vab.x * vac.y - vac.x * vab.y);

                    if (m + n <= 1 && m >= 0 && n >= 0) {
                        if (t < tMin) {
                            iFace = face;
                            iPoint = pp;
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
}