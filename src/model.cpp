
#include <model.h>

#include "model.h"

Model::Model() {}

Model::Model(std::string objPath, std::string mtlPath) {
    load(objPath, mtlPath);
}

bool Model::load(std::string objPath, std::string mtlPath, std::string cfgPath) {
    scene.mLights.clear();
    scene.mMaterials.clear();
    scene.mMeshes.clear();
    scene.mNumLights = 0;
    scene.mNumMaterials = 0;
    scene.mNumMeshes = 0;

    if(cfgPath.empty()){
        cfgPath = objPath.substr(0, objPath.find_last_of('.'));
        cfgPath += ".cfg";
        std::cout << "use defualt config path:" << cfgPath << std::endl;
        if (!loadCfg(cfgPath)) {
            std::cout << "ERROR: NO CFG FILE!" << std::endl;
            return false;
        }
    }
    else if(!loadCfg(cfgPath)){
        std::cout << "ERROR: NO CFG FILE!" << std::endl;
        return false;
    }

    if (objPath.empty()) {
        std::cout << "ERROR: NO OBJ FILE!" << std::endl;
        return false;
    } else if (!loadObj(objPath))
        return false;

    if (mtlPath.empty()) {
        mtlPath = objPath.substr(0, objPath.find_first_of('.'));
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

    scene.mLights.push_back(config.light);

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
    while (getline(ifs, lineBuf)) {
        std::cout << "Read from file: " << lineBuf << std::endl;

        if (!lineBuf.empty())
            if (lineBuf[lineBuf.size() - 1] == '\n')
                lineBuf.erase(lineBuf.end() - 1);

        if (!lineBuf.empty())
            if (lineBuf[lineBuf.size() - 1] == '\r')
                lineBuf.erase(lineBuf.end() - 1);

        if (lineBuf.empty())
            continue;

        // remove space and tab at begin
        while (lineBuf[0] == ' ' || lineBuf[0] == '\t') {
            lineBuf.erase(lineBuf.begin());
            if (lineBuf.empty())
                break;
        }
        if (lineBuf.empty())
            continue;
        // remove  space and tab at end
        while (lineBuf[lineBuf.size() - 1] == ' ' || lineBuf[lineBuf.size() - 1] == ' ') {
            lineBuf.erase(lineBuf.begin());
            if (lineBuf.empty())
                break;
        }
        if (lineBuf.empty())
            continue;

        // group
        if (lineBuf[0] == 'g') {
            // new mesh
            if (lineBuf == "g default") {
                mesh.numFaces = mesh.faces.size();

                if (mesh.numFaces > 0)
                    scene.mMeshes.push_back(mesh);

                mesh.faces.clear();
                mesh.name.clear();
            } else {
                std::string groupName = lineBuf.substr(2, lineBuf.size());
                mesh.name = groupName;
                if(mesh.name == config.light.groupname){
                    mesh.isLightSource = true;
                }
                else{
                    mesh.isLightSource = false;
                }
            }
        }

        // comment
        if (lineBuf[0] == '#')
            continue;

        // vertex
        if (lineBuf[0] == 'v' && lineBuf[1] == ' ') {
            std::string data = lineBuf.substr(2, lineBuf.size());
            std::stringstream ss(data);
            float vx, vy, vz;
            ss >> vx >> vy >> vz;
            scene.mVertices.push_back(Point3f(vx, vy, vz));
        }

        // normal
        if (lineBuf[0] == 'v' && lineBuf[1] == 'n') {
            std::string data = lineBuf.substr(3, lineBuf.size());
            std::stringstream ss(data);
            float vnx, vny, vnz;
            ss >> vnx >> vny >> vnz;
            scene.mNormals.push_back(Vector3f(vnx, vny, vnz));
        }

        // texture coord
        if (lineBuf[0] == 'v' && lineBuf[1] == 't') {
            std::string data = lineBuf.substr(3, lineBuf.size());
            std::stringstream ss(data);
            float vtx, vty;
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
                if (c == ' ')
                    spaceCnt++;
            }

            if (spaceCnt < 2)
                return false;

            std::vector<std::string> vertexData;

            while (!data.empty()) {
                vertexData.push_back(data.substr(0, data.find_first_of(' ')));

                data.erase(0, vertexData[vertexData.size() - 1].size());

                if (!data.empty())
                    data.erase(0, 1);
            }

            Face face;

            for (int i = 0; i < 3; i++) {
                // a//c
                if (vertexData[i].find_last_of('/') - vertexData[i].find_first_of('/') == 1) {
                    sscanf(vertexData[i].c_str(), "%d//%d", &face.mVerticesIndices[i], &face.mNormalsIndices[i]);
                    face.mVerticesIndices[i]--;
                    face.mNormalsIndices[i]--;
                }
                // a/b/c
                if (vertexData[i].find_last_of('/') - vertexData[i].find_first_of('/') > 1) {
                    sscanf(vertexData[i].c_str(), "%d/%d/%d", &face.mVerticesIndices[i], &face.mTextureCoordsIndices[i],
                           &face.mNormalsIndices[i]);
                    face.mVerticesIndices[i]--;
                    face.mTextureCoordsIndices[i]--;
                    face.mNormalsIndices[i]--;
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
                face.mVerticesIndices[1] = face.mVerticesIndices[2];
                face.mNormalsIndices[1] = face.mNormalsIndices[2];
                // a//c
                if (vertexData[i].find_last_of('/') - vertexData[i].find_first_of('/') == 1) {
                    sscanf(vertexData[i].c_str(), "%d//%d", &face.mVerticesIndices[2], &face.mNormalsIndices[2]);

                    face.mVerticesIndices[2]--;
                    face.mNormalsIndices[2]--;
                }
                // a/b/c
                if (vertexData[i].find_last_of('/') - vertexData[i].find_first_of('/') > 1) {
                    face.mTextureCoordsIndices[1] = face.mTextureCoordsIndices[2];
                    sscanf(vertexData[i].c_str(), "%d/%d/%d", &face.mVerticesIndices[2], &face.mTextureCoordsIndices[2],
                           &face.mNormalsIndices[2]);
                    face.mVerticesIndices[2]--;
                    face.mTextureCoordsIndices[2]--;
                    face.mNormalsIndices[2]--;
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

        if (lineBuf.empty())
            continue;

        // remove space and tab at begin
        while (lineBuf[0] == ' ' || lineBuf[0] == '\t') {
            lineBuf.erase(lineBuf.begin());
            if (lineBuf.empty())
                break;
        }
        if (lineBuf.empty())
            continue;
        // remove  space and tab at end
        while (lineBuf[lineBuf.size() - 1] == ' ' || lineBuf[lineBuf.size() - 1] == ' ') {
            lineBuf.erase(lineBuf.begin());
            if (lineBuf.empty())
                break;
        }
        if (lineBuf.empty())
            continue;

        // comment
        if (lineBuf[0] == '#')
            continue;

        // new material
        if (lineBuf.substr(0, 6) == "newmtl") {
            if (!material.name.empty()) {
                scene.mMaterials.push_back(material);
                // map

                scene.mtlName2ID[material.name] = scene.mtlName2ID.size();
            }
            material.name = lineBuf.substr(7, lineBuf.size() - 1);
            material.illum = 0;
            // only 0 is correct
            std::memset(material.Kd, 0, sizeof(material.Kd));
            std::memset(material.Ka, 0, sizeof(material.Ka));
            std::memset(material.Ks, 0, sizeof(material.Ks));
            std::memset(material.Tf, 0, sizeof(material.Tf));
            material.Ni = material.Ns = 0;
        }

        if (lineBuf.substr(0, 6) == "illum") {
            std::string data = lineBuf.substr(7, lineBuf.size() - 1);
            sscanf(data.c_str(), "%d", &material.illum);
        }

        if (lineBuf[0] == 'K' && lineBuf[1] == 'd')
            sscanf(lineBuf.substr(3, lineBuf.size() - 1).c_str(), "%f%f%f", material.Kd, material.Kd + 1,
                   material.Kd + 2);

        if (lineBuf[0] == 'K' && lineBuf[1] == 'a')
            sscanf(lineBuf.substr(3, lineBuf.size() - 1).c_str(), "%f%f%f", material.Ka, material.Ka + 1,
                   material.Ka + 2);

        if (lineBuf[0] == 'K' && lineBuf[1] == 's')
            sscanf(lineBuf.substr(3, lineBuf.size() - 1).c_str(), "%f%f%f", material.Ks, material.Ks + 1,
                   material.Ks + 2);

        if (lineBuf[0] == 'T' && lineBuf[1] == 'f')
            sscanf(lineBuf.substr(3, lineBuf.size() - 1).c_str(), "%f%f%f", material.Tf, material.Tf + 1,
                   material.Tf + 2);

        if (lineBuf[0] == 'N' && lineBuf[1] == 's') {
            sscanf(lineBuf.substr(3, lineBuf.size() - 1).c_str(), "%f", &material.Ns);
        }

        if (lineBuf[0] == 'N' && lineBuf[1] == 'i') {
            sscanf(lineBuf.substr(3, lineBuf.size() - 1).c_str(), "%f", &material.Ni);
        }
    }
    return true;
}

bool Model::loadCfg(std::string cfgPath) {
    std::ifstream ifs(cfgPath);
    if (!ifs) {
        std::cout << "ERROR: CANNOT OPEN " << cfgPath << std::endl;
        return false;
    }

    std::string lineBuf;
    bool cameraParamsFlag = false;
    bool lightFlag = false;
    while(getline(ifs, lineBuf))
    {
        std::cout << "Read from file: " << lineBuf << std::endl;

        if (!lineBuf.empty())
            if (lineBuf[lineBuf.size() - 1] == '\n')
                lineBuf.erase(lineBuf.end() - 1);

        if (!lineBuf.empty())
            if (lineBuf[lineBuf.size() - 1] == '\r')
                lineBuf.erase(lineBuf.end() - 1);

        if (lineBuf.empty())
            continue;

        // remove space and tab at begin
        while (lineBuf[0] == ' ' || lineBuf[0] == '\t') {
            lineBuf.erase(lineBuf.begin());
            if (lineBuf.empty())
                break;
        }
        if (lineBuf.empty())
            continue;
        // remove  space and tab at end
        while (lineBuf[lineBuf.size() - 1] == ' ' || lineBuf[lineBuf.size() - 1] == ' ') {
            lineBuf.erase(lineBuf.begin());
            if (lineBuf.empty())
                break;
        }
        if (lineBuf.empty())
            continue;

        // comment
        if (lineBuf[0] == '#')
            continue;

        if (lineBuf.substr(0, lineBuf.find_first_of(' ')) == "resolution")
            sscanf(lineBuf.substr(lineBuf.find_first_of(' ') + 1, lineBuf.size()).c_str(),
                    "%d*%d", &config.resolution.width, &config.resolution.height);

        else if(lineBuf == "cameraparams"){
            cameraParamsFlag = true;
            lightFlag = false;
        }
        else if(lineBuf == "light")
        {
            lightFlag = true;
            cameraParamsFlag = false;
        }

        if(cameraParamsFlag && !lightFlag){
            std::string data = lineBuf.substr(lineBuf.find_first_of(' ') + 1);
            std::stringstream ss(data);
            if(lineBuf[0] == 'p'){
                float px, py, pz;
                ss >> px >> py >> pz;
                config.cameraparams.position = Point3f(px, py, pz);
            }
            else if(lineBuf[0] == 'l'){
                float lx, ly, lz;
                ss >> lx >> ly >> lz;
                config.cameraparams.lookat = Point3f(lx, ly, lz);
            }
            else if(lineBuf[0] == 'u'){
                float ux, uy, uz;
                ss >> ux >> uy >> uz;
                config.cameraparams.up = Vector3f(ux, uy, uz);
            }
            else if(lineBuf[0] == 'f'){
                ss >> config.cameraparams.fovy;
            }
        }

        if(lightFlag && !cameraParamsFlag){
            std::string data = lineBuf.substr(lineBuf.find_first_of(' ') + 1);
            std::stringstream ss(data);
            if(lineBuf[0] == 'g'){
                ss >> config.light.groupname;
            }
            else if(lineBuf[0] == 'c'){
                float cx, cy, cz;
                ss >> cx >> cy >> cz;
                config.light.center = Point3f(cx, cy, cz);
            }
            else if(lineBuf[0] == 'r'){
                ss >> config.light.radius;
            }
            else if(lineBuf[0] == 'L'){
                ss >> config.light.Le[0] >> config.light.Le[1] >> config.light.Le[2];
            }
        }
    }

    return true;
}

void Model::getMaxIndices(Face &face, const Mesh &mesh) {
    // xMax
    int xMaxIndices = -1;
    float xMax = std::numeric_limits<float>::min();
    for (int i = 0; i < 3; i++) {
        if (scene.mVertices[face.mVerticesIndices[i]].x > xMax) {
            xMax = scene.mVertices[face.mVerticesIndices[i]].x;
            xMaxIndices = face.mVerticesIndices[i];
        }
    }
    face.maxVerticesIndices[0] = xMaxIndices;

    // yMax
    int yMaxIndices = -1;
    float yMax = std::numeric_limits<float>::min();
    for (int i = 0; i < 3; i++) {
        if (scene.mVertices[face.mVerticesIndices[i]].y > yMax) {
            yMax = scene.mVertices[face.mVerticesIndices[i]].y;
            yMaxIndices = face.mVerticesIndices[i];
        }
    }
    face.maxVerticesIndices[1] = yMaxIndices;

    // zMax
    int zMaxIndices = -1;
    float zMax = std::numeric_limits<float>::min();
    for (int i = 0; i < 3; i++) {
        if (scene.mVertices[face.mVerticesIndices[i]].z > zMax) {
            zMax = scene.mVertices[face.mVerticesIndices[i]].z;
            zMaxIndices = face.mVerticesIndices[i];
        }
    }
    face.maxVerticesIndices[2] = zMaxIndices;
}

void Model::getMinIndices(Face &face, const Mesh &mesh) {
    // xMin
    int xMinIndices = -1;
    float xMin = std::numeric_limits<float>::max();
    for (int i = 0; i < 3; i++) {
        if (scene.mVertices[face.mVerticesIndices[i]].x < xMin) {
            xMin = scene.mVertices[face.mVerticesIndices[i]].x;
            xMinIndices = face.mVerticesIndices[i];
        }
    }
    face.minVerticesIndices[0] = xMinIndices;

    // yMin
    int yMinIndices = -1;
    float yMin = std::numeric_limits<float>::max();
    for (int i = 0; i < 3; i++) {
        if (scene.mVertices[face.mVerticesIndices[i]].y < yMin) {
            yMin = scene.mVertices[face.mVerticesIndices[i]].y;
            yMinIndices = face.mVerticesIndices[i];
        }
    }
    face.minVerticesIndices[1] = yMinIndices;

    // zMin
    int zMinIndices = -1;
    float zMin = std::numeric_limits<float>::max();
    for (int i = 0; i < 3; i++) {
        if (scene.mVertices[face.mVerticesIndices[i]].z < zMin) {
            zMin = scene.mVertices[face.mVerticesIndices[i]].z;
            zMinIndices = face.mVerticesIndices[i];
        }
    }
    face.minVerticesIndices[2] = zMinIndices;
}

void Model::computeFaceNormal(Face &face, const Mesh &mesh) {
    Point3f fv1 = scene.mVertices[face.mVerticesIndices[0]];
    Point3f fv2 = scene.mVertices[face.mVerticesIndices[1]];
    Point3f fv3 = scene.mVertices[face.mVerticesIndices[2]];

    Vector3f fv12 = fv2 - fv1;
    Vector3f fv13 = fv3 - fv1;

    face.faceNormal = cross(fv12, fv13).normalize();
    // ensure the normal point to the outside
    if (dot(scene.mNormals[face.mNormalsIndices[0]], face.faceNormal) < 0)
        face.faceNormal = face.faceNormal * (-1);

    face.a = face.faceNormal.x;
    face.b = face.faceNormal.y;
    face.c = face.faceNormal.z;
    face.d = -(face.a * fv1.x + face.b * fv1.y + face.c * fv1.z);
}


