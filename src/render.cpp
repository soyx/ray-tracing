#include "render.h"

Render::Render(Model &model, Camera &camera, int sampleNum)
    : camera(camera), renderModel(model) {
    this->sampleNum = sampleNum;
    srand((unsigned)time(NULL));
}

void Render::run() {
    time_t start = time(NULL);
    int total = camera.filmSize.x * camera.filmSize.y;

    for (int y = 0; y < camera.filmSize.y; y++) {
#pragma omp parallel for schedule(dynamic, 1)
        for (int x = 0; x < camera.filmSize.x; x++) {
            fprintf(stderr, "\r%5.4f%%",
                    100. * (y * camera.filmSize.x + x) / total);
            Vec3f c;
            // #pragma omp parallel for schedule(dynamic, 1)
            for (int i = 0; i < sampleNum; i++) {
                // filter
                double r1 = 2 * (RANDNUM);
                double r2 = 2 * (RANDNUM);
                double dx = r1 < 1 ? std::sqrt(r1) - 1 : 1 - std::sqrt(2 - r1);
                double dy = r2 < 1 ? std::sqrt(r2) - 1 : 1 - std::sqrt(2 - r2);

                Vector3f d = camera.camera2World(
                    camera.raster2Camera(
                        Point3f(x + (dx + .5) * .5, y + (dy + .5) * .5, 1.)) -
                    Point3f(0, 0, 0));
                c = c + radiance(Ray(camera.position, d.normalize()), 0) *
                            (1. / sampleNum);
            }
            camera.film[y * camera.filmSize.x + x] = c;
        }
    }
    renderTime += (time(NULL) - start);
}

double Render::intersect(const Face &face, const Ray &ray) {
    time_t start = time(NULL);
    // intersection with mesh
    Vec3f rMax, rMin;

    double faceMaxx = renderModel.scene.mVertices[face.maxVerticesIndices.x].x;
    double faceMaxy = renderModel.scene.mVertices[face.maxVerticesIndices.y].y;
    double faceMaxz = renderModel.scene.mVertices[face.maxVerticesIndices.z].z;

    rMax = Vec3f((faceMaxx - ray.o.x) / ray.d.x, (faceMaxy - ray.o.y) / ray.d.y,
                 (faceMaxz - ray.o.z) / ray.d.z);

    double faceMinx = renderModel.scene.mVertices[face.minVerticesIndices.x].x;
    double faceMiny = renderModel.scene.mVertices[face.minVerticesIndices.y].y;
    double faceMinz = renderModel.scene.mVertices[face.minVerticesIndices.z].z;
    Vec3f temp =
        Vec3f((faceMinx - ray.o.x) / ray.d.x, (faceMiny - ray.o.y) / ray.d.y,
              (faceMinz - ray.o.z) / ray.d.z);

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

    if (rMax.x <= 0 || rMax.y <= 0 || rMax.z <= 0) {
        interTime += (time(NULL)-start);
        return 0;}
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
            if (dot(face.faceNormal, ray.d) == 0) {
                interTime += (time(NULL) - start);
                return 0;
            }
            double t = -(face.a * ray.o.x + face.b * ray.o.y +
                         face.c * ray.o.z + face.d) /
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
                interTime += (time(NULL) - start);
                return t;
            }
        }
    }
    interTime += (time(NULL) - start);
    return 0;
}

Vec3f Render::radiance(const Ray &ray, int depth) {
    double t = std::numeric_limits<double>::infinity();
    Face face;
    int id = -1;
    enum InterType { OTHER, FACE, SPHERE_LIGHTSOURCE, QUAD_LIGHTSOURCE };

    InterType interType = OTHER;

    // obj meshes
    for (int i = 0; i < renderModel.scene.mNumMeshes; i++) {
        Mesh mesh = renderModel.scene.mMeshes[i];
        if (!mesh.isIntersect(ray)) continue;
        for (int j = 0; j < mesh.numFaces; j++) {
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
    for (unsigned int i = 0; i < renderModel.scene.sphereLights.size(); i++) {
        double tt = renderModel.scene.sphereLights[i].intersect(ray);
        if (tt > 1e-10 && tt < t) {
            t = tt;
            id = i;
            interType = SPHERE_LIGHTSOURCE;
        }
    }
    for (unsigned int i = 0; i < renderModel.scene.quadLights.size(); i++) {
        double tt = renderModel.scene.quadLights[i].intersect(ray);
        if (tt > 1e-10 && tt < t) {
            t = tt;
            id = i;
            interType = QUAD_LIGHTSOURCE;
        }
    }

    Vec3f xyzColor(0, 0, 0);
    if (id < 0) return xyzColor;

    // obj meshes
    if (interType == FACE) {
        Material material =
            renderModel.scene
                .mMaterials[renderModel.scene.mtlName2ID[face.materialName]];
        if (depth > 5) {
            if (depth > 10) {
                return xyzColor;
            }
            double rNum = (RANDNUM);
            if (material.KDiffuse.maxCor < rNum &&
                material.KSpecular.maxCor < rNum) {
                return xyzColor;
            }
        }

        Point3f p = ray.o + ray.d * t;
        Vector3f n = face.faceNormal;
        Vector3f nl;

        if (dot(n, ray.d) > 0)
            nl = n * -1;
        else
            nl = n;

        if (material.KDiffuse.maxCor > 1e-10) {
            double r1 = 2 * M_PI * (RANDNUM);
            double r2 = (RANDNUM);
            double r2s = std::sqrt(r2);
            Vector3f w = nl;
            Vector3f u;
            if (std::abs(w.x) > std::abs(w.y)) {
                u = cross(Vector3f(0, 1, 0), w).normalize();
            } else {
                u = cross(Vector3f(1, 0, 0), w).normalize();
            }
            Vector3f v = cross(w, u);

            Vector3f d = (u * std::cos(r1) * r2s + v * std::sin(r1) * r2s +
                          w * std::sqrt(1 - r2))
                             .normalize();

            xyzColor = xyzColor +
                       mul(material.KDiffuse, radiance(Ray(p, d), depth + 1)) *
                           dot(d, nl);
        }

        if (material.KSpecular.maxCor > 1e-10) {
            Vector3f w = nl;
            Vector3f u;
            if (std::abs(w.x) > std::abs(w.y))
                u = cross(Vector3f(0, 1, 0), w).normalize();
            else
                u = cross(Vector3f(1, 0, 0), w).normalize();
            Vector3f v = cross(w, u);

            
            double cosTheta = std::pow((RANDNUM), 1 / (material.Ns + 1));
            double sinTheta = std::sqrt(1 - cosTheta * cosTheta);
            double phi = 2 * M_PI * (RANDNUM);

            Vector3f L = ray.d * -1;
            Vector3f H = u * sinTheta * std::cos(phi) +
                         v * sinTheta * std::sin(phi) +
                         w * cosTheta;
            Vector3f N = nl;
            Vector3f V = H * 2 * dot(H, L) - L;

            xyzColor =
                xyzColor +
                mul(material.KSpecular *
                        std::pow(dot(H, N) * (material.Ns + 1), material.Ns) *
                        2 * M_PI *
                        (1 / ((material.Ns + 1) *
                              std::pow(cosTheta, material.Ns) *
                              sinTheta)),
                    radiance(Ray(p, V), depth + 1));

            // xyzColor =
            //    xyzColor +
            //    mul(material.KSpecular,
            //        radiance(Ray(p, ray.d - nl * 2 * dot(nl, ray.d)), depth +
            //        1));
        }

        if (material.Tf.maxCor <= 1 - 1e-10) {
            // refr
            Ray reflRay(p, ray.d - nl * 2 * dot(nl, ray.d));
            bool into = dot(nl, n) > 0;

            double cosTheta1 = -dot(nl, ray.d);
            double cos2Theta1 = cosTheta1 * cosTheta1;
            double sinTheta1 = std::sqrt(1 - cos2Theta1);

            Vec3f kdegradation = mul(Vec3f(1, 1, 1) - material.KDiffuse,
                                     Vec3f(1, 1, 1) - material.Tf);

            // inside->outside
            if (!into) {
                double n12 = material.Ni;
                double sinThresh = 1 / n12;
                if (sinTheta1 > sinThresh) {
                    // all reflect
                    xyzColor = xyzColor + face.emission +
                               mul(kdegradation, radiance(reflRay, depth + 1));
                } else {
                    // light plane
                    Vector3f nt = cross(nl, cross(ray.d, nl)).normalize();
                    if (dot(ray.d, nt) < 0) nt = nt * -1;
                    double sin2Theta2 = (1 - cos2Theta1) * n12 * n12;
                    double cos2Theta2 = 1 - sin2Theta2;

                    Vector3f tdir = nl * -std::sqrt(cos2Theta2) +
                                    nt * std::sqrt(sin2Theta2);

                    double a = n12 * std::sqrt(1 - (n12 * sinTheta1) *
                                                       (n12 * sinTheta1));
                    double b = cosTheta1;
                    double Rp = (a - b) / (a + b);
                    Rp *= Rp;
                    double c = a / n12;
                    double d = b * n12;
                    double Rs = (d - c) / (d + c);
                    Rs *= Rs;
                    double R = (Rp + Rs) * 0.5;
                    Vec3f kr = mul(Vec3f(R, R, R), kdegradation);
                    double T = 1 - R;
                    Vec3f kt = mul(Vec3f(T, T, T), kdegradation);
                    xyzColor = xyzColor + face.emission +
                               mul(kr, radiance(reflRay, depth + 1)) +
                               mul(kt, radiance(Ray(p, tdir), depth + 1));
                }
            }
            // outside -> inside
            else {
                double n12 = 1 / material.Ni;
                Vector3f nt = cross(nl, cross(ray.d, nl)).normalize();
                if (dot(ray.d, nt) < 0) nt = nt * -1;
                double sin2Theta2 = (1 - cos2Theta1) * n12 * n12;
                double cos2Theta2 = 1 - sin2Theta2;

                Vector3f tdir =
                    nl * -std::sqrt(cos2Theta2) + nt * std::sqrt(sin2Theta2);

                double a =
                    n12 * std::sqrt(1 - (n12 * sinTheta1) * (n12 * sinTheta1));
                double b = cosTheta1;
                double Rp = (a - b) / (a + b);
                Rp *= Rp;
                double c = a / n12;
                double d = b * n12;
                double Rs = (d - c) / (d + c);
                Rs *= Rs;
                double R = (Rp + Rs) * 0.5;
                Vec3f kr = mul(Vec3f(R, R, R), kdegradation);
                double T = 1 - R;
                Vec3f kt = mul(Vec3f(T, T, T), kdegradation);
                xyzColor = xyzColor + face.emission +
                           mul(kr, radiance(reflRay, depth + 1)) +
                           mul(kt, radiance(Ray(p, tdir), depth + 1));
            }
        }
    }
    // other shapes
    else if (interType == SPHERE_LIGHTSOURCE) {
        SphereLightSource light = renderModel.scene.sphereLights[id];
        xyzColor = xyzColor + light.emission;
        if (depth > 5) {
            if (depth > 10) return xyzColor;
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

            Vector3f d = (u * std::cos(r1) * r2s + v * std::sin(r1) * r2s +
                          w * std::sqrt(1 - r2))
                             .normalize();

            xyzColor =
                xyzColor + mul(light.KDiffuse, radiance(Ray(p, d), depth + 1));
        }

        if (light.KSpecular.maxCor > 1e-10) {
            xyzColor =
                xyzColor +
                mul(light.KSpecular,
                    radiance(Ray(p, ray.d - n * 2 * dot(n, ray.d)), depth + 1));
        }
    } else if (interType == QUAD_LIGHTSOURCE) {
        QuadLightSource light = renderModel.scene.quadLights[id];
        xyzColor = xyzColor + light.emission;
        if (depth > 5) {
            if (depth > 10) return xyzColor;
            double rNum = (RANDNUM);
            if (light.KDiffuse.maxCor < rNum && light.KSpecular.maxCor < rNum) {
                return xyzColor;
            }
        }
        Point3f p = ray.o + ray.d * t;
        Vector3f n = light.normal;

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

            Vector3f d = (u * std::cos(r1) * r2s + v * std::sin(r1) * r2s +
                          w * std::sqrt(1 - r2))
                             .normalize();

            xyzColor =
                xyzColor + mul(light.KDiffuse, radiance(Ray(p, d), depth + 1));
        }

        if (light.KSpecular.maxCor > 1e-10) {
            xyzColor =
                xyzColor +
                mul(light.KSpecular,
                    radiance(Ray(p, ray.d - n * 2 * dot(n, ray.d)), depth + 1));
        }
    }

    return xyzColor;
}
