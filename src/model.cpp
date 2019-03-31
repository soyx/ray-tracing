
#include <model.h>

#include "model.h"

Model::Model() {}

Model::Model(std::string objPath, std::string mtlPath) {
    load(objPath, mtlPath);
}

bool Model::load(std::string objPath, std::string mtlPath,
                 std::string cfgPath) {
    time_t start = time(NULL);
    scene.mMaterials.clear();
    scene.mMeshes.clear();
    scene.mNumLights = 0;
    scene.mNumMaterials = 0;
    scene.mNumMeshes = 0;

    if (cfgPath.empty()) {
        cfgPath = objPath.substr(0, objPath.find_last_of('.'));
        cfgPath += ".cfg";
        std::cout << "use defualt config path:" << cfgPath << std::endl;
        if (!loadCfg(cfgPath)) {
            std::cout << "ERROR: NO CFG FILE!" << std::endl;
            return false;
        }
    } else if (!loadCfg(cfgPath)) {
        std::cout << "ERROR: NO CFG FILE!" << std::endl;
        return false;
    }

    if (objPath.empty()) {
        std::cout << "ERROR: NO OBJ FILE!" << std::endl;
        return false;
    } else if (!loadObj(objPath))
        return false;

    if (mtlPath.empty()) {
        mtlPath = objPath.substr(0, objPath.find_last_of('.'));
        mtlPath += ".mtl";
        std::cout << "use defualt material path:" << mtlPath << std::endl;
        if (!loadMtl(mtlPath)) {
            std::cout << "ERROR: NO MTL FILE!" << std::endl;
            return false;
        }
    } else if (!loadMtl(mtlPath)) {
        std::cout << "ERROR: NO MTL FILE!" << std::endl;
        return false;
    }

    scene.mNumMaterials = scene.mMaterials.size();
    scene.mNumMeshes = scene.mMeshes.size();

    loadTime = time(NULL) - start;
    return true;
}

bool Model::loadObj(std::string objPath) {
    std::ifstream ifs(objPath);
    if (!ifs) {
        std::cout << "ERROR: CANNOT OPEN " << objPath << std::endl;
        return false;
    }

    std::string lineBuf;
    Mesh mesh;
    std::string mtlName = "default";
    mesh.maxVertices = Vec3f(-INF, -INF, -INF);
    mesh.minVertices = Vec3f(INF, INF, INF);
    while (getline(ifs, lineBuf)) {
        std::cout << "Read from file: " << lineBuf << std::endl;

        if (!lineBuf.empty())
            if (lineBuf[lineBuf.size() - 1] == '\n')
                lineBuf.erase(lineBuf.end() - 1);

        if (!lineBuf.empty())
            if (lineBuf[lineBuf.size() - 1] == '\r')
                lineBuf.erase(lineBuf.end() - 1);

        if (lineBuf.empty()) continue;

        // remove space and tab at begin
        while (lineBuf[0] == ' ' || lineBuf[0] == '\t') {
            lineBuf.erase(lineBuf.begin());
            if (lineBuf.empty()) break;
        }
        if (lineBuf.empty()) continue;
        // remove  space and tab at end
        while (lineBuf[lineBuf.size() - 1] == ' ' ||
               lineBuf[lineBuf.size() - 1] == ' ') {
            lineBuf.erase(lineBuf.begin());
            if (lineBuf.empty()) break;
        }
        if (lineBuf.empty()) continue;

        // group
        if (lineBuf[0] == 'g') {
            // new mesh
            if (lineBuf == "g default") {
                mesh.numFaces = mesh.faces.size();

                if (mesh.numFaces > 0) scene.mMeshes.push_back(mesh);

                mesh.maxVertices = Vec3f(-INF, -INF, -INF);
                mesh.minVertices = Vec3f(INF, INF, INF);

                mesh.faces.clear();
                mesh.name.clear();
            } else {
                std::string groupName = lineBuf.substr(2, lineBuf.size());
                mesh.name = groupName;
            }
        }

        // comment
        if (lineBuf[0] == '#') continue;

        // vertex
        if (lineBuf[0] == 'v' && lineBuf[1] == ' ') {
            std::string data = lineBuf.substr(2, lineBuf.size());
            std::stringstream ss(data);
            double vx, vy, vz;
            ss >> vx >> vy >> vz;
            scene.mVertices.push_back(Point3f(vx, vy, vz));

            if (vx > mesh.maxVertices.x) mesh.maxVertices.x = vx;
            if (vy > mesh.maxVertices.y) mesh.maxVertices.y = vy;
            if (vz > mesh.maxVertices.z) mesh.maxVertices.z = vz;
            if (vx < mesh.minVertices.x) mesh.minVertices.x = vx;
            if (vy < mesh.minVertices.y) mesh.minVertices.y = vy;
            if (vz < mesh.minVertices.z) mesh.minVertices.z = vz;
        }

        // normal
        if (lineBuf[0] == 'v' && lineBuf[1] == 'n') {
            std::string data = lineBuf.substr(3, lineBuf.size());
            std::stringstream ss(data);
            double vnx, vny, vnz;
            ss >> vnx >> vny >> vnz;
            scene.mNormals.push_back(Vector3f(vnx, vny, vnz));
        }

        // texture coord
        if (lineBuf[0] == 'v' && lineBuf[1] == 't') {
            std::string data = lineBuf.substr(3, lineBuf.size());
            std::stringstream ss(data);
            double vtx, vty;
            ss >> vtx >> vty;
            scene.mTextureCoords.push_back(Point2f(vtx, vty));
        }

        // update mtl of face
        if (lineBuf.substr(0, 6) == "usemtl") {
            mtlName = lineBuf.substr(7, lineBuf.size());
        }

        // face
        if (lineBuf[0] == 'f' && lineBuf[1] == ' ') {
            std::string data = lineBuf.substr(2, lineBuf.size());

            int spaceCnt = 0;
            for (auto c : data) {
                if (c == ' ') spaceCnt++;
            }

            if (spaceCnt < 2) return false;

            std::vector<std::string> vertexData;

            while (!data.empty()) {
                vertexData.push_back(data.substr(0, data.find_first_of(' ')));

                data.erase(0, vertexData[vertexData.size() - 1].size());

                if (!data.empty()) data.erase(0, 1);
            }

            Face face;

            for (int i = 0; i < 3; i++) {
                // a//c
                if (vertexData[i].find_last_of('/') -
                        vertexData[i].find_first_of('/') ==
                    1) {
                    sscanf(vertexData[i].c_str(), "%d//%d",
                           &face.verticesIndices[i], &face.normalsIndices[i]);
                    face.verticesIndices[i]--;
                    face.normalsIndices[i]--;
                }
                // a/b/c
                if (vertexData[i].find_last_of('/') -
                        vertexData[i].find_first_of('/') >
                    1) {
                    sscanf(vertexData[i].c_str(), "%d/%d/%d",
                           &face.verticesIndices[i],
                           &face.textureCoordsIndices[i],
                           &face.normalsIndices[i]);
                    face.verticesIndices[i]--;
                    face.textureCoordsIndices[i]--;
                    face.normalsIndices[i]--;
                }
            }

            face.materialName = mtlName;
            getMaxIndices(face, mesh);
            getMinIndices(face, mesh);
            computeFaceNormal(face, mesh);
            face.meshId = scene.mMeshes.size();
            mesh.faces.push_back(face);

            // convert origin faces to trigle faces
            for (int i = 3; i < vertexData.size(); i++) {
                face.verticesIndices[1] = face.verticesIndices[2];
                face.normalsIndices[1] = face.normalsIndices[2];
                // a//c
                if (vertexData[i].find_last_of('/') -
                        vertexData[i].find_first_of('/') ==
                    1) {
                    sscanf(vertexData[i].c_str(), "%d//%d",
                           &face.verticesIndices[2], &face.normalsIndices[2]);

                    face.verticesIndices[2]--;
                    face.normalsIndices[2]--;
                }
                // a/b/c
                if (vertexData[i].find_last_of('/') -
                        vertexData[i].find_first_of('/') >
                    1) {
                    face.textureCoordsIndices[1] = face.textureCoordsIndices[2];
                    sscanf(vertexData[i].c_str(), "%d/%d/%d",
                           &face.verticesIndices[2],
                           &face.textureCoordsIndices[2],
                           &face.normalsIndices[2]);
                    face.verticesIndices[2]--;
                    face.textureCoordsIndices[2]--;
                    face.normalsIndices[2]--;
                }

                face.materialName = mtlName;
                getMaxIndices(face, mesh);
                getMinIndices(face, mesh);
                computeFaceNormal(face, mesh);
                face.meshId = scene.mMeshes.size();
                mesh.faces.push_back(face);
            }
        }
    }
    // last
    mesh.numFaces = mesh.faces.size();
    if (mesh.numFaces > 0) scene.mMeshes.push_back(mesh);

    return true;
}

bool Model::loadMtl(std::string mtlPath) {
    std::ifstream ifs(mtlPath);
    if (!ifs) {
        std::cout << "ERROR: CANNOT OPEN " << mtlPath << std::endl;
        return false;
    }

    std::string lineBuf;
    Material material;
    while (getline(ifs, lineBuf)) {
        std::cout << "Read from file: " << lineBuf << std::endl;

        if (!lineBuf.empty())
            if (lineBuf[lineBuf.size() - 1] == '\n')
                lineBuf.erase(lineBuf.end() - 1);

        if (!lineBuf.empty())
            if (lineBuf[lineBuf.size() - 1] == '\r')
                lineBuf.erase(lineBuf.end() - 1);

        if (lineBuf.empty()) continue;

        // remove space and tab at begin
        while (lineBuf[0] == ' ' || lineBuf[0] == '\t') {
            lineBuf.erase(lineBuf.begin());
            if (lineBuf.empty()) break;
        }
        if (lineBuf.empty()) continue;
        // remove  space and tab at end
        while (lineBuf[lineBuf.size() - 1] == ' ' ||
               lineBuf[lineBuf.size() - 1] == ' ') {
            lineBuf.erase(lineBuf.begin());
            if (lineBuf.empty()) break;
        }
        if (lineBuf.empty()) continue;

        // comment
        if (lineBuf[0] == '#') continue;

        // new material
        if (lineBuf.substr(0, 6) == "newmtl") {
            if (!material.name.empty()) {
                scene.mMaterials.push_back(material);
                // map
                unsigned long s = scene.mtlName2ID.size();
                scene.mtlName2ID[material.name] = (int)s;
            }
            material.name = lineBuf.substr(7, lineBuf.size() - 1);
            material.illum = 0;
            material.Ni = material.Ns = 0;
            material.KSpecular = Vec3f();
            material.KDiffuse = Vec3f();
        }

        if (lineBuf.substr(0, 6) == "illum") {
            std::string data = lineBuf.substr(7, lineBuf.size() - 1);
            sscanf(data.c_str(), "%d", &material.illum);
        }

        if (lineBuf[0] == 'K' && lineBuf[1] == 'd') {
            double a, b, c;
            sscanf(lineBuf.substr(3, lineBuf.size() - 1).c_str(), "%lf%lf%lf",
                   &a, &b, &c);
            material.KDiffuse = Vec3f(a, b, c);
        }

        // if (lineBuf[0] == 'K' && lineBuf[1] == 'a')
        //     sscanf(lineBuf.substr(3, lineBuf.size() - 1).c_str(),
        //     "%lf%lf%lf", material.Ka, material.Ka + 1,
        //            material.Ka + 2);

        if (lineBuf[0] == 'K' && lineBuf[1] == 's') {
            double a, b, c;
            sscanf(lineBuf.substr(3, lineBuf.size() - 1).c_str(), "%lf%lf%lf",
                   &a, &b, &c);
            material.KSpecular = Vec3f(a, b, c);
        }
        if (lineBuf[0] == 'T' && lineBuf[1] == 'f') {
            double a, b, c;
            sscanf(lineBuf.substr(3, lineBuf.size() - 1).c_str(), "%lf%lf%lf",
                   &a, &b, &c);
            material.Tf = Vec3f(a, b, c);
        }

        if (lineBuf[0] == 'N' && lineBuf[1] == 's') {
            sscanf(lineBuf.substr(3, lineBuf.size() - 1).c_str(), "%lf",
                   &material.Ns);
        }

        if (lineBuf[0] == 'N' && lineBuf[1] == 'i') {
            sscanf(lineBuf.substr(3, lineBuf.size() - 1).c_str(), "%lf",
                   &material.Ni);
        }
    }
    scene.mMaterials.push_back(material);
    unsigned long s = scene.mtlName2ID.size();
    scene.mtlName2ID[material.name] = (int)s;

    return true;
}

bool Model::loadCfg(std::string cfgPath) {
    std::ifstream ifs(cfgPath);
    if (!ifs) {
        std::cout << "ERROR: CANNOT OPEN " << cfgPath << std::endl;
        return false;
    }

    std::string lineBuf;
    enum DataType { OTHER, CAMERA_PARAMS, SPHERE_LIGHT, QUAD_LIGHT };

    DataType type = OTHER;

    SphereLightSource spherelight;
    QuadLightSource quadLight;

    while (getline(ifs, lineBuf)) {
        std::cout << "Read from file: " << lineBuf << std::endl;

        if (!lineBuf.empty())
            if (lineBuf[lineBuf.size() - 1] == '\n')
                lineBuf.erase(lineBuf.end() - 1);

        if (!lineBuf.empty())
            if (lineBuf[lineBuf.size() - 1] == '\r')
                lineBuf.erase(lineBuf.end() - 1);

        if (lineBuf.empty()) continue;

        // remove space and tab at begin
        while (lineBuf[0] == ' ' || lineBuf[0] == '\t') {
            lineBuf.erase(lineBuf.begin());
            if (lineBuf.empty()) break;
        }
        if (lineBuf.empty()) continue;
        // remove  space and tab at end
        while (lineBuf[lineBuf.size() - 1] == ' ' ||
               lineBuf[lineBuf.size() - 1] == ' ') {
            lineBuf.erase(lineBuf.begin());
            if (lineBuf.empty()) break;
        }
        if (lineBuf.empty()) continue;

        // comment
        if (lineBuf[0] == '#') continue;

        if (lineBuf.substr(0, lineBuf.find_first_of(' ')) == "resolution")
            sscanf(
                lineBuf.substr(lineBuf.find_first_of(' ') + 1, lineBuf.size())
                    .c_str(),
                "%d*%d", &config.resolution.width, &config.resolution.height);

        else if (lineBuf == "cameraparams") {
            if (type == SPHERE_LIGHT) {
                this->scene.sphereLights.push_back(spherelight);
                spherelight.clear();
            }
            if (type == QUAD_LIGHT) {
                this->scene.quadLights.push_back(quadLight);
                quadLight.clear();
            }
            type = CAMERA_PARAMS;
        } else if (lineBuf == "spherelight") {
            if (type == SPHERE_LIGHT) {
                this->scene.sphereLights.push_back(spherelight);
                spherelight.clear();
            }
            if (type == QUAD_LIGHT) {
                this->scene.quadLights.push_back(quadLight);
                quadLight.clear();
            }
            type = SPHERE_LIGHT;
        } else if( lineBuf == "quadlight"){
            if (type == SPHERE_LIGHT) {
                this->scene.sphereLights.push_back(spherelight);
                spherelight.clear();
            }
            if (type == QUAD_LIGHT) {
                this->scene.quadLights.push_back(quadLight);
                quadLight.clear();
            }
            type = QUAD_LIGHT;
        }

        if (type == CAMERA_PARAMS) {
            std::string data = lineBuf.substr(lineBuf.find_first_of(' ') + 1);
            std::stringstream ss(data);
            if (lineBuf[0] == 'p') {
                double px, py, pz;
                ss >> px >> py >> pz;
                config.cameraparams.position = Point3f(px, py, pz);
            } else if (lineBuf[0] == 'l') {
                double lx, ly, lz;
                ss >> lx >> ly >> lz;
                config.cameraparams.lookat = Point3f(lx, ly, lz);
            } else if (lineBuf[0] == 'u') {
                double ux, uy, uz;
                ss >> ux >> uy >> uz;
                config.cameraparams.up = Vector3f(ux, uy, uz);
            } else if (lineBuf[0] == 'f') {
                ss >> config.cameraparams.fovy;
            }
        }

        if (type == SPHERE_LIGHT) {
            std::string data = lineBuf.substr(lineBuf.find_first_of(' ') + 1);
            std::stringstream ss(data);
            if (lineBuf[0] == 'c') {
                double cx, cy, cz;
                ss >> cx >> cy >> cz;
                spherelight.position = Point3f(cx, cy, cz);
            } else if (lineBuf[0] == 'r') {
                ss >> spherelight.radius;
            } else if (lineBuf[0] == 'L') {
                double ex, ey, ez;
                ss >> ex >> ey >> ez;
                spherelight.emission = Vec3f(ex, ey, ez);
            }
        }

        if (type == QUAD_LIGHT) {
            std::string data = lineBuf.substr(lineBuf.find_first_of(' ') + 1);
            std::stringstream ss(data);
            if (lineBuf[0] == 'c') {
                double cx, cy, cz;
                ss >> cx >> cy >> cz;
                quadLight.center = Point3f(cx, cy, cz);
            } else if (lineBuf[0] == 'n') {
                double nx, ny, nz;
                ss >> nx >> ny >> nz;
                quadLight.normal = Vector3f(nx, ny, nz);
            } else if (lineBuf[0] == 's') {
                double sx, sy;
                ss >> sx >> sy;
                quadLight.size = Vec2f(sx, sy);
            } else if (lineBuf[0] == 'L') {
                double ex, ey, ez;
                ss >> ex >> ey >> ez;
                quadLight.emission = Vec3f(ex, ey, ez);
            }
        }
    }
    if (type == SPHERE_LIGHT) this->scene.sphereLights.push_back(spherelight);
    if (type == QUAD_LIGHT) this->scene.quadLights.push_back(quadLight);

    return true;
}

void Model::getMaxIndices(Face &face, const Mesh &mesh) {
    // xMax
    int xMaxIndices = -1;
    double xMax = -std::numeric_limits<double>::max();
    for (int i = 0; i < 3; i++) {
        if (scene.mVertices[face.verticesIndices[i]].x > xMax) {
            xMax = scene.mVertices[face.verticesIndices[i]].x;
            xMaxIndices = face.verticesIndices[i];
        }
    }
    face.maxVerticesIndices[0] = xMaxIndices;

    // yMax
    int yMaxIndices = -1;
    double yMax = -std::numeric_limits<double>::max();
    for (int i = 0; i < 3; i++) {
        if (scene.mVertices[face.verticesIndices[i]].y > yMax) {
            yMax = scene.mVertices[face.verticesIndices[i]].y;
            yMaxIndices = face.verticesIndices[i];
        }
    }
    face.maxVerticesIndices[1] = yMaxIndices;

    // zMax
    int zMaxIndices = -1;
    double zMax = -std::numeric_limits<double>::max();
    for (int i = 0; i < 3; i++) {
        if (scene.mVertices[face.verticesIndices[i]].z > zMax) {
            zMax = scene.mVertices[face.verticesIndices[i]].z;
            zMaxIndices = face.verticesIndices[i];
        }
    }
    face.maxVerticesIndices[2] = zMaxIndices;
    if (face.maxVerticesIndices[0] == -1 || face.maxVerticesIndices[1] == -1 ||
        face.maxVerticesIndices[2] == -1)
        printf("DEBUG: maxVertices error!\n");
}

void Model::getMinIndices(Face &face, const Mesh &mesh) {
    // xMin
    int xMinIndices = -1;
    double xMin = std::numeric_limits<double>::max();
    for (int i = 0; i < 3; i++) {
        if (scene.mVertices[face.verticesIndices[i]].x < xMin) {
            xMin = scene.mVertices[face.verticesIndices[i]].x;
            xMinIndices = face.verticesIndices[i];
        }
    }
    face.minVerticesIndices[0] = xMinIndices;

    // yMin
    int yMinIndices = -1;
    double yMin = std::numeric_limits<double>::max();
    for (int i = 0; i < 3; i++) {
        if (scene.mVertices[face.verticesIndices[i]].y < yMin) {
            yMin = scene.mVertices[face.verticesIndices[i]].y;
            yMinIndices = face.verticesIndices[i];
        }
    }
    face.minVerticesIndices[1] = yMinIndices;

    // zMin
    int zMinIndices = -1;
    double zMin = std::numeric_limits<double>::max();
    for (int i = 0; i < 3; i++) {
        if (scene.mVertices[face.verticesIndices[i]].z < zMin) {
            zMin = scene.mVertices[face.verticesIndices[i]].z;
            zMinIndices = face.verticesIndices[i];
        }
    }
    face.minVerticesIndices[2] = zMinIndices;
    if (face.maxVerticesIndices[0] == -1 || face.maxVerticesIndices[1] == -1 ||
        face.maxVerticesIndices[2] == -1)
        printf("DEBUG: maxVertices error!\n");
}

void Model::computeFaceNormal(Face &face, const Mesh &mesh) {
    Point3f fv1 = scene.mVertices[face.verticesIndices[0]];
    Point3f fv2 = scene.mVertices[face.verticesIndices[1]];
    Point3f fv3 = scene.mVertices[face.verticesIndices[2]];

    Vector3f fv12 = fv2 - fv1;
    Vector3f fv13 = fv3 - fv1;

    face.faceNormal = cross(fv12, fv13).normalize();
    // ensure the normal point to the outside
    if (dot(scene.mNormals[face.normalsIndices[0]], face.faceNormal) < 0)
        face.faceNormal = face.faceNormal * (-1);

    face.a = face.faceNormal.x;
    face.b = face.faceNormal.y;
    face.c = face.faceNormal.z;
    face.d = -(face.a * fv1.x + face.b * fv1.y + face.c * fv1.z);
}

bool Mesh::isIntersect(const Ray &ray) {
    Vec3f rMax, rMin;

    rMax = Vec3f((maxVertices.x - ray.o.x) / ray.d.x,
                 (maxVertices.y - ray.o.y) / ray.d.y,
                 (maxVertices.z - ray.o.z) / ray.d.z);

    Vec3f temp = Vec3f((minVertices.x - ray.o.x) / ray.d.x,
                       (minVertices.y - ray.o.y) / ray.d.y,
                       (minVertices.z - ray.o.z) / ray.d.z);

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

    if (rMax.x <= 0 || rMax.y <= 0 || rMax.z <= 0) return false;
    if (rMax.x == INF || rMax.y == INF || rMax.z == INF) return false;
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
        if (range12[1] >= range3[0] && range12[0] <= range3[1]) return true;
    }
    return false;
}
