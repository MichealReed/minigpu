#ifndef MINIGPU_H
#define MINIGPU_H

#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

    typedef struct MGPUBuffer MGPUBuffer;
    typedef struct MGPUComputeShader MGPUComputeShader;

    // GPUComputeShader
    MGPUComputeShader *mgpuCreateComputeShader();
    void mgpuDestroyComputeShader(MGPUComputeShader *shader);
    void mgpuLoadKernelString(MGPUComputeShader *shader, const char *kernelString);
    void mgpuLoadKernelFile(MGPUComputeShader *shader, const char *path);
    int mgpuHasKernel(MGPUComputeShader *shader);

    // GPUBuffer
    MGPUBuffer *mgpuCreateBuffer(MGPUComputeShader *shader, uint32_t size, uint32_t memSize);
    void mgpuDestroyBuffer(MGPUBuffer *buffer);
    void mgpuSetBuffer(MGPUComputeShader *shader, const char *kernel, const char *tag, MGPUBuffer *buffer);
    void mgpuDispatch(MGPUComputeShader *shader, const char *kernel, uint32_t groupsX, uint32_t groupsY, uint32_t groupsZ);
    void mgpuReadBufferSync(MGPUBuffer *buffer, MGPUBuffer *otherBuffer);
    void mgpuReadBufferAsync(MGPUBuffer *buffer, MGPUBuffer *otherBuffer, void (*callback)(void *), void *userData);

#ifdef __cplusplus
} // extern "C"
#endif

#ifdef __cplusplus

#include "gpu.h"

namespace mgpu
{

    class Buffer
    {
    public:
        Buffer(gpu::Array data) : data(data) {}
        ~Buffer() { /* Cleanup */ }
        void readSync(Buffer &otherBuffer);
        void requestAsync(Buffer &otherBuffer, std::function<void(void *)> callback, void *userData);
        void release();
        gpu::Array getData() { return data; }

    private:
        gpu::Array data;
    };

    class ComputeShader
    {
        std::vector<gpu::Tensor> bindings;

    public:
        ComputeShader() {}
        ~ComputeShader() { /* Cleanup */ }
        void loadKernelString(const std::string &kernelString);
        void loadKernelFile(const std::string &path);
        bool hasKernel();
        Buffer createBuffer(uint32_t size, uint32_t memSize);
        void setBuffer(const std::string &kernel, const std::string &tag, Buffer &buffer);
        void dispatch(const std::string &kernel, uint32_t groupsX, uint32_t groupsY, uint32_t groupsZ);

    private:
        // Implementation details
    };

} // namespace mgpu

#endif // __cplusplus

#endif // MINIGPU_H
