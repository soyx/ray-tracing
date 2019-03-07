#include "render.h"

Render::Render()
{
    srand((unsigned)time(NULL));
}

Render::Render(Model &model)
{
    renderModel = model;
    srand((unsigned)time(NULL));
}

float normalPdf(Vector3f normal, float x)
{
}

Ray Render::generateRay(const Point3f &origin, const Vector3f &normal)
{
    Vector3f direct = sample(normal);

    if (dot(normal, direct) < 0)
    {
        std::cout << "WARN: the angle of ray and normal overs 90" << std::endl;
        return Ray(origin, normal);
    }

    return Ray(origin, direct);
}

Vector3f Render::sample(const Vector3f &normal)
{
    Vector3f sampleVec;

    if (normal.x != 0)
    {
        sampleVec.y = RANDNUM;
        sampleVec.z = RANDNUM;
        if (normal.x > 0)
        {
            sampleVec.x = -(sampleVec.y * normal.y + sampleVec.z * normal.z) + 0.5;
        }
        else
        {
            sampleVec.x = -(sampleVec.y * normal.y + sampleVec.z * normal.z) - 0.5;
        }
    }
    else if (normal.y != 0)
    {
        sampleVec.x = RANDNUM;
        sampleVec.z = RANDNUM;
        if (normal.y > 0)
        {
            sampleVec.y = -(sampleVec.x * normal.x + sampleVec.z * normal.z) + 0.5;
        }
        else
        {
            sampleVec.y = -(sampleVec.x * normal.x + sampleVec.z * normal.z) - 0.5;
        }
    }
    else if (normal.z != 0)
    {
        sampleVec.x = RANDNUM;
        sampleVec.y = RANDNUM;
        if (normal.z > 0)
        {
            sampleVec.z = -(sampleVec.x * normal.x + sampleVec.y * normal.y) + 0.5;
        }
        else
        {
            sampleVec.z = -(sampleVec.x * normal.x + sampleVec.y * normal.y) - 0.5;
        }
    }
    else
    {
        std::cout << "ERROR: invalid normal" << std::endl;
    }

    return sampleVec;
}

void Render::getIntersection(const Ray &ray, Point3f &iPoint, Face &iFace)
{
    for (int i = 0; i < renderModel.scene.mNumMeshes; i++)
    {
        Mesh mesh = renderModel.scene.mMeshes[i];
        for (int j = 0; j < mesh.numFaces; j++)
        {
            Face face = mesh.faces[j];
            Vector3f rMax, rMin;
            rMax = (mesh.mVertices[face.maxVecticesIndices[0]] - ray.o) / ray.d;
            Vector3f temp = (mesh.mVertices[face.minVecticesIndices[0]] - ray.o) / ray.d;

            if (temp.x > rMax.x)
            {
                rMin.x = rMax.x;
                rMax.x = temp.x;
            }
            else
                rMin.x = temp.x;

            if (temp.y > rMax.y)
            {
                rMin.y = rMax.y;
                rMax.y = temp.y;
            }
            else
                rMin.y = temp.y;

            if (temp.z > rMax.z)
            {
                rMin.z = rMax.z;
                rMax.z = temp.z;
            }
            else
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
            if (range1[1] >= range2[0] && range1[0] <= range2[1])
            {
                range12[0] = std::max(range1[0], range2[0]);
                range12[1] = std::max(range1[0], range2[1]);
                if (range12[1] >= range3[0] && range12[0] <= range3[1])
                {
                    // detail process
                    // resolve equation
                    // tode
                }
                else
                    continue;
            }
            else
                continue;
        }
    }
}