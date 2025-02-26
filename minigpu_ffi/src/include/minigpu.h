#ifndef MINIGPU_H
#define MINIGPU_H

#include "export.h"
#ifdef __cplusplus
#include "../include/buffer.h"
#include "../include/compute_shader.h"

extern "C"
{
#endif

    typedef struct MGPUComputeShader MGPUComputeShader;
    typedef struct MGPUBuffer MGPUBuffer;

    EXPORT void mgpuInitializeContext();
    typedef void (*MGPUCallback)(void);
    EXPORT void mgpuInitializeContextAsync(MGPUCallback callback);
    EXPORT void mgpuDestroyContext();
    EXPORT MGPUComputeShader *mgpuCreateComputeShader();
    EXPORT void mgpuDestroyComputeShader(MGPUComputeShader *shader);
    EXPORT void mgpuLoadKernel(MGPUComputeShader *shader, const char *kernelString);
    EXPORT int mgpuHasKernel(MGPUComputeShader *shader);
    EXPORT MGPUBuffer *mgpuCreateBuffer(int size, int memSize);
    EXPORT void mgpuDestroyBuffer(MGPUBuffer *buffer);
    EXPORT void mgpuSetBuffer(MGPUComputeShader *shader, int tag, MGPUBuffer *buffer);
    EXPORT void mgpuDispatch(MGPUComputeShader *shader, int groupsX, int groupsY, int groupsZ);
    EXPORT void mgpuDispatchAsync(MGPUComputeShader *shader, int groupsX, int groupsY, int groupsZ, MGPUCallback callback);
    EXPORT void mgpuReadBufferSync(MGPUBuffer *buffer, float *outputData, size_t size, size_t offset);
    EXPORT void mgpuReadBufferAsync(MGPUBuffer *buffer,
        float *outputData,
        size_t size,
        size_t offset,
        MGPUCallback callback);
    EXPORT void mgpuSetBufferData(MGPUBuffer *buffer, const float *inputData, size_t size);

#ifdef __cplusplus
}
#endif

#endif // MINIGPU_H
