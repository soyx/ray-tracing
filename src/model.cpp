#include "model.h"

Model::Model() {}

Model::Model(std::string objPath, std::string mtlPath)
{
    load(objPath, mtlPath);
}

bool Model::load(std::string objPath, std::string mtlPath)
{
    scene.mLights.clear();
    scene.mMaterials.clear();
    scene.mMeshes.clear();
    scene.mNumLights = 0;
    scene.mNumMaterials = 0;
    scene.mNumMeshes = 0;
    if (objPath.empty())
    {
        std::cout << "ERROR: NO OBJ FILE!" << std::endl;
        return false;
    }
    else if (!loadObj(objPath))
        return false;
    if (objPath.empty())
    {
        mtlPath = objPath.substr(0, objPath.find_first_of('/'));
        mtlPath += objPath.substr(objPath.find_last_of('/'), objPath.find_last_of('.'));
        mtlPath += ".mtl";
        std::cout << "use defualt material path:" << mtlPath << std::endl;
        if (!loadMtl(mtlPath))
        {
            std::cout << "ERROR: NO MTL FILE!" << std::endl;
            return false;
        }
    }
    else if (!loadMtl(mtlPath))
    {
        std::cout << "ERROR: NO MTL FILE!" << std::endl;
        return false;
    }

    scene.mNumMaterials = scene.mMaterials.size();
    scene.mNumMeshes = scene.mMeshes.size();

    return true;
}

bool Model::loadObj(std::string objPath)
{
    std::ifstream ifs(objPath);
    if (!ifs)
    {
        std::cout << "ERROR: CANNOT OPEN " << objPath << std::endl;
        return false;
    }

    std::string lineBuf;
    Mesh mesh;
    std::string mtlName = "default";
    while (getline(ifs, lineBuf))
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
        while (lineBuf[0] == ' ' || lineBuf[0] == '\t')
        {
            lineBuf.erase(lineBuf.begin());
            if (lineBuf.empty())
                break;
        }
        if (lineBuf.empty())
            continue;
        // remove  space and tab at end
        while (lineBuf[lineBuf.size() - 1] == ' ' || lineBuf[lineBuf.size() - 1] == ' ')
        {
            lineBuf.erase(lineBuf.begin());
            if (lineBuf.empty())
                break;
        }
        if (lineBuf.empty())
            continue;

        // group
        if (lineBuf[0] == 'g')
        {
            // new mesh
            if (lineBuf == "g default")
            {
                mesh.mNumVertices = mesh.mVertices.size();
                mesh.mNumNormals = mesh.mNormals.size();
                mesh.mNumTextureCoords = mesh.mTextureCoords.size();
                mesh.numFaces = mesh.faces.size();

                if (mesh.mNumVertices > 0)
                    scene.mMeshes.push_back(mesh);

                mesh.mNormals.clear();
                mesh.mTextureCoords.clear();
                mesh.mVertices.clear();
                mesh.faces.clear();
                mesh.name.clear();
            }
            else
            {
                std::string groupName = lineBuf.substr(2, lineBuf.size());
                mesh.name = groupName;
            }
        }

        // comment
        if (lineBuf[0] == '#')
            continue;

        // vertex
        if (lineBuf[0] == 'v' && lineBuf[1] == ' ')
        {
            std::string data = lineBuf.substr(2, lineBuf.size());
            std::stringstream ss(data);
            float vx, vy, vz;
            ss >> vx >> vy >> vz;
            mesh.mVertices.push_back(Point3f(vx, vy, vz));
        }

        // normal
        if (lineBuf[0] == 'v' && lineBuf[1] == 'n')
        {
            std::string data = lineBuf.substr(3, lineBuf.size());
            std::stringstream ss(data);
            float vnx, vny, vnz;
            ss >> vnx >> vny >> vnz;
            mesh.mNormals.push_back(Vector3f(vnx, vny, vnz));
        }

        // texture coord
        if (lineBuf[0] == 'v' && lineBuf[1] == 't')
        {
            std::string data = lineBuf.substr(3, lineBuf.size());
            std::stringstream ss(data);
            float vtx, vty;
            ss >> vtx >> vty;
            mesh.mTextureCoords.push_back(Point2f(vtx, vty));
        }

        // update mtl of face
        if (lineBuf.substr(0, 6) == "usemtl")
        {
            mtlName = lineBuf.substr(7, lineBuf.size());
        }

        // face
        if (lineBuf[0] == 'f' && lineBuf[1] == ' ')
        {
            std::string data = lineBuf.substr(2, lineBuf.size());

            int spaceCnt = 0;
            for (auto c : data)
            {
                if (c == ' ')
                    spaceCnt++;
            }

            if (spaceCnt < 2)
                return false;

            std::vector<std::string> vertexData;

            while (!data.empty())
            {
                vertexData.push_back(data.substr(0, data.find_first_of(' ')));

                data.erase(0, vertexData[vertexData.size() - 1].size());

                if (!data.empty())
                    data.erase(0, 1);
            }

            Face face;

            for (int i = 0; i < 3; i++)
            {
                // a//c
                if (vertexData[i].find_last_of('/') - vertexData[i].find_first_of('/') == 1){
                    sscanf(vertexData[i].c_str(), "%d//%d", &face.mVerticesIndices[i], &face.mNormalsIndices[i]);
                    face.mVerticesIndices[i]--;
                    face.mNormalsIndices[i]--;
                }
                // a/b/c
                if (vertexData[i].find_last_of('/') - vertexData[i].find_first_of('/') > 1){
                    sscanf(vertexData[i].c_str(), "%d/%d/%d", &face.mVerticesIndices[i], &face.mTextureCoordsIndices[i], &face.mNormalsIndices[i]);
                    face.mVerticesIndices[i]--;
                    face.mTextureCoordsIndices[i]--;
                    face.mNormalsIndices[i]--;
                }
            }

            face.materialName = mtlName;
            getMaxIndices(face);
            getMinIndices(face);
            mesh.faces.push_back(face);

            // convert origin faces to trigle faces
            for (int i = 3; i < vertexData.size(); i++)
            {
                face.mVerticesIndices[1] = face.mVerticesIndices[2];
                face.mNormalsIndices[1] = face.mNormalsIndices[2];
                // a//c
                if (vertexData[i].find_last_of('/') - vertexData[i].find_first_of('/') == 1)
                {
                    sscanf(vertexData[i].c_str(), "%d//%d", &face.mVerticesIndices[2], &face.mNormalsIndices[2]);

                    face.mVerticesIndices[2]--;
                    face.mNormalsIndices[2]--;
                }
                // a/b/c
                if (vertexData[i].find_last_of('/') - vertexData[i].find_first_of('/') > 1)
                {
                    face.mTextureCoordsIndices[1] = face.mTextureCoordsIndices[2];
                    sscanf(vertexData[i].c_str(), "%d/%d/%d", &face.mVerticesIndices[2], &face.mTextureCoordsIndices[2], &face.mNormalsIndices[2]);
                    face.mVerticesIndices[2]--;
                    face.mTextureCoordsIndices[2]--;
                    face.mNormalsIndices[2]--;
                }

                face.materialName = mtlName;
                getMaxIndices(face);
                getMinIndices(face);
                mesh.faces.push_back(face);
            }
        }
    }

    return true;
}

bool Model::loadMtl(std::string mtlPath)
{

    std::ifstream ifs(mtlPath);
    if (!ifs)
    {
        std::cout << "ERROR: CANNOT OPEN " << mtlPath << std::endl;
        return false;
    }

    std::string lineBuf;
    Material material;
    while (getline(ifs, lineBuf))
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
        while (lineBuf[0] == ' ' || lineBuf[0] == '\t')
        {
            lineBuf.erase(lineBuf.begin());
            if (lineBuf.empty())
                break;
        }
        if (lineBuf.empty())
            continue;
        // remove  space and tab at end
        while (lineBuf[lineBuf.size() - 1] == ' ' || lineBuf[lineBuf.size() - 1] == ' ')
        {
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
        if (lineBuf.substr(0, 6) == "newmtl")
        {
            if(!material.name.empty()){
                scene.mMaterials.push_back(material);
                // map

                scene.mtlName2ID[material.name] = scene.mtlName2ID.size();
            }
            material.name = lineBuf.substr(7, lineBuf.size() - 1);
            material.illum = 0;
            material.Ka = material.Kd = material.Ks = material.Tf = Vector3f(0, 0, 0);
            material.Ni = material.Ns = 0;
            
        }

        if(lineBuf.substr(0, 6) == "illum"){
            std::string data = lineBuf.substr(7, lineBuf.size() -1);
            sscanf(data.c_str(), "%d", &material.illum);
        }

        if(lineBuf[0] == 'K' && lineBuf[1] == 'd'){
            float a, b, c;
            sscanf(lineBuf.substr(3, lineBuf.size() - 1).c_str(), "%f%f%f", &a, &b, &c);
            material.Kd = Vector3f(a, b,c);
        }

        if(lineBuf[0] == 'K' && lineBuf[1] == 'a'){
            float a, b, c;
            sscanf(lineBuf.substr(3, lineBuf.size() - 1).c_str(), "%f%f%f", &a, &b, &c);
            material.Ka = Vector3f(a, b, c);
        }

        if (lineBuf[0] == 'K' && lineBuf[1] == 's')
        {
            float a, b, c;
            sscanf(lineBuf.substr(3, lineBuf.size() - 1).c_str(), "%f%f%f", &a, &b, &c);
            material.Ks = Vector3f(a, b, c);
        }

        if(lineBuf[0] =='T' && lineBuf[1] == 'f')
        {
            float a, b, c;
            sscanf(lineBuf.substr(3, lineBuf.size() - 1).c_str(), "%f%f%f", &a, &b, &c);
            material.Tf = Vector3f(a, b, c);
        }

        if(lineBuf[0] == 'N' && lineBuf[1] == 's')
        {
            sscanf(lineBuf.substr(3, lineBuf.size() - 1).c_str(), "%f", &material.Ns);
        }

        if(lineBuf[0] == 'N' && lineBuf[1] == 'i')
        {
            sscanf(lineBuf.substr(3, lineBuf.size() - 1).c_str(), "%f", &material.Ni);
        }
    
        
    }
    return true;
}

void getMaxIndices(Face &face, const Mesh &mesh){
    // xMax
    int xMaxIndices = -1;
    int xMax = std::numeric_limits<int>::min();
    for(int i = 0; i < 3; i++){
        if(mesh.mVertices[face.mVerticesIndices[i]].x > xMax)
        {
            xMax = mesh.mVertices[face.mVerticesIndices[i]].x;
            xMaxIndices = face.maxVecticesIndices[i];
        }
    }
    face.maxVecticesIndices[0] = xMaxIndices;

    // yMax
    int yMaxIndices = -1;
    int yMax = std::numeric_limits<int>::min();
    for (int i = 0; i < 3; i++)
    {
        if (mesh.mVertices[face.mVerticesIndices[i]].y > yMax)
        {
            yMax = mesh.mVertices[face.mVerticesIndices[i]].y;
            yMaxIndices = face.maxVecticesIndices[i];
        }
    }
    face.maxVecticesIndices[1] = yMaxIndices;

    // zMax
    int zMaxIndices = -1;
    int zMax = std::numeric_limits<int>::min();
    for (int i = 0; i < 3; i++)
    {
        if (mesh.mVertices[face.mVerticesIndices[i]].z > zMax)
        {
            zMax = mesh.mVertices[face.mVerticesIndices[i]].z;
            zMaxIndices = face.maxVecticesIndices[i];
        }
    }
    face.maxVecticesIndices[2] = zMaxIndices;
}

void getMinIndices(Face &face, const Mesh &mesh)
{
    // xMin
    int xMinIndices = -1;
    int xMin = std::numeric_limits<int>::max();
    for (int i = 0; i < 3; i++)
    {
        if (mesh.mVertices[face.mVerticesIndices[i]].x < xMin)
        {
            xMin = mesh.mVertices[face.mVerticesIndices[i]].x;
            xMinIndices = face.minVecticesIndices[i];
        }
    }
    face.minVecticesIndices[0] = xMinIndices;

    // yMin
    int yMinIndices = -1;
    int yMin = std::numeric_limits<int>::min();
    for (int i = 0; i < 3; i++)
    {
        if (mesh.mVertices[face.mVerticesIndices[i]].y < yMin)
        {
            yMin = mesh.mVertices[face.mVerticesIndices[i]].y;
            yMinIndices = face.minVecticesIndices[i];
        }
    }
    face.minVecticesIndices[1] = yMinIndices;

    // zMin
    int zMinIndices = -1;
    int zMin = std::numeric_limits<int>::lowest();
    for (int i = 0; i < 3; i++)
    {
        if (mesh.mVertices[face.mVeråticesIndices[i]].z < zMin)
        {
            zMin = mesh.mVertices[face.mVerticesIndices[i]].z;
            zMinIndices = face.minVecticesIndices[i];
        }
    }
    face.minVecticesIndices[2] = zMinIndices;
}