#ifndef MINIGPU_H
#define MINIGPU_H

#include "export.h"
#ifdef __cplusplus
#include "buffer.h"

extern "C"
{
#endif

    typedef struct MGPUComputeShader MGPUComputeShader;
    typedef struct MGPUBuffer MGPUBuffer;

    EXPORT void mgpuInitializeContext();
    EXPORT void mgpuDestroyContext();
    EXPORT MGPUComputeShader *mgpuCreateComputeShader();
    EXPORT void mgpuDestroyComputeShader(MGPUComputeShader *shader);
    EXPORT void mgpuLoadKernelString(MGPUComputeShader *shader, const char *kernelString);
    EXPORT void mgpuLoadKernelFile(MGPUComputeShader *shader, const char *path);
    EXPORT int mgpuHasKernel(MGPUComputeShader *shader);
    EXPORT MGPUBuffer *mgpuCreateBuffer(MGPUComputeShader *shader, int size, int memSize);
    EXPORT void mgpuDestroyBuffer(MGPUBuffer *buffer);
    EXPORT void mgpuSetBuffer(MGPUComputeShader *shader, const char *kernel, const char *tag, MGPUBuffer *buffer);
    EXPORT void mgpuDispatch(MGPUComputeShader *shader, const char *kernel, int groupsX, int groupsY, int groupsZ);
    EXPORT void mgpuReadBufferSync(MGPUBuffer *buffer, MGPUBuffer *otherBuffer);
    EXPORT void mgpuReadBufferAsync(MGPUBuffer *buffer, MGPUBuffer *otherBuffer, void (*callback)(void *), void *userData);
#ifdef __cplusplus
}
#endif

#endif // MINIGPU_H
