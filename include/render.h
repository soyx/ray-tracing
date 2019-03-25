#ifndef RENDER_H
#define RENDER_H

#include <ctime>
#include <cstdlib>
#include <cmath>

#include <cuda_runtime.h>

#include "util.h"
#include "model.h"
#include "camera.h"

#define RANDNUM std::rand() / (double)(RAND_MAX)

class Render {
public:
    __host__ __device__ Render(Model &model, Camera &camera, int sampleNum = 50);
    __host__ __device__ Render(Model &model, Camera &camera, int argc, char *argv[], int sampleNum = 50, bool useGPU = true);

    __host__ __device__ void run();

private:

    __host__ __device__ double intersect(const Face &face, const Ray &ray);

    __host__ __device__ Vec3f radiance(const Ray &ray, int depth);

    __device__ void runGPU(int sampleNum);

    Model &renderModel;

    Camera &camera;

    int sampleNum;

    bool useGPU;

    Camera *d_camera;
    Model *d_model;
};


#endif // RENDER_H
