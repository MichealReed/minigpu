#ifndef MINIGPU_H
#define MINIGPU_H

#include <cstdint>
#include <functional>
#include <string>
#include <vector>
#include <fstream>
#include "../external/include/gpu/gpu.h"
#include "webgpu/webgpu.h"

#include "export.h"

typedef struct MGPUComputeShader MGPUComputeShader;
typedef struct MGPUBuffer MGPUBuffer;

extern "C"
{
    EXPORT void mgpuInitializeContext();
    EXPORT void mgpuDestroyContext();
    EXPORT MGPUComputeShader *mgpuCreateComputeShader();
    EXPORT void mgpuDestroyComputeShader(MGPUComputeShader *shader);
    EXPORT void mgpuLoadKernelString(MGPUComputeShader *shader, const char *kernelString);
    EXPORT void mgpuLoadKernelFile(MGPUComputeShader *shader, const char *path);
    EXPORT int mgpuHasKernel(MGPUComputeShader *shader);
    EXPORT MGPUBuffer *mgpuCreateBuffer(MGPUComputeShader *shader, uint32_t size, uint32_t memSize);
    EXPORT void mgpuDestroyBuffer(MGPUBuffer *buffer);
    EXPORT void mgpuSetBuffer(MGPUComputeShader *shader, const char *kernel, const char *tag, MGPUBuffer *buffer);
    EXPORT void mgpuDispatch(MGPUComputeShader *shader, const char *kernel, uint32_t groupsX, uint32_t groupsY, uint32_t groupsZ);
    EXPORT void mgpuReadBufferSync(MGPUBuffer *buffer, MGPUBuffer *otherBuffer);
    EXPORT void mgpuReadBufferAsync(MGPUBuffer *buffer, MGPUBuffer *otherBuffer, void (*callback)(void *), void *userData);
}

namespace mgpu
{
    void initializeContext();
    void destroyContext();
    extern gpu::Context ctx;
    class Buffer
    {
    public:
        Buffer(gpu::Array data);
        void readSync(const Buffer &otherBuffer);
        void requestAsync(const Buffer &otherBuffer, std::function<void(void *)> callback, void *userData);
        void release();

        gpu::Array data;
    };

    class ComputeShader
    {
    public:
        void loadKernelString(const std::string &kernelString);
        void loadKernelFile(const std::string &path);
        bool hasKernel() const;
        gpu::Array createBuffer(uint32_t size, uint32_t memSize);
        void setBuffer(const std::string &kernel, const std::string &tag, const Buffer &buffer);
        void dispatch(const std::string &kernel, uint32_t groupsX, uint32_t groupsY, uint32_t groupsZ);

    private:
        gpu::KernelCode code;
        std::vector<gpu::Tensor> bindings;
    };

} // namespace mgpu

#endif // MINIGPU_H
