#include "../include/minigpu.h"
#ifdef __cplusplus
using namespace mgpu;
using namespace gpu;
extern "C"
{
#endif

    void mgpuInitializeContext()
    {
        initializeContext();
    }

    void mgpuDestroyContext()
    {
        destroyContext();
    }

    MGPUComputeShader *mgpuCreateComputeShader()
    {
        return reinterpret_cast<MGPUComputeShader *>(new mgpu::ComputeShader());
    }

    void mgpuDestroyComputeShader(MGPUComputeShader *shader)
    {
        delete reinterpret_cast<mgpu::ComputeShader *>(shader);
    }

    void mgpuLoadKernel(MGPUComputeShader *shader, const char *kernelString)
    {
        if (shader && kernelString)
        {
            reinterpret_cast<mgpu::ComputeShader *>(shader)->loadKernelString(kernelString);
        }
        else
        {
            gpu::LOG(kDefLog, kError, "Invalid shader or kernelString pointer");
        }
    }

    int mgpuHasKernel(MGPUComputeShader *shader)
    {
        if (shader)
        {
            return reinterpret_cast<mgpu::ComputeShader *>(shader)->hasKernel();
        }
        else
        {
            LOG(kDefLog, kError, "Invalid shader pointer");
            return 0;
        }
    }

    MGPUBuffer *mgpuCreateBuffer(MGPUBuffer *buffer, int size, int memSize)
    {
        if (buffer)
        {
            gpu::Array array = reinterpret_cast<mgpu::Buffer *>(buffer)->createBuffer(size, memSize);
            return reinterpret_cast<MGPUBuffer *>(new mgpu::Buffer(std::move(array)));
        }
        else
        {
            LOG(kDefLog, kError, "Invalid buffer pointer");
            return nullptr;
        }
    }

    void mgpuDestroyBuffer(MGPUBuffer *buffer)
    {
        if (buffer)
        {
            reinterpret_cast<mgpu::Buffer *>(buffer)->release();
            delete reinterpret_cast<mgpu::Buffer *>(buffer);
        }
        else
        {
            LOG(kDefLog, kError, "Invalid buffer pointer");
        }
    }

    void mgpuSetBuffer(MGPUComputeShader *shader, const char *kernel, const char *tag, MGPUBuffer *buffer)
    {
        if (shader && kernel && tag && buffer)
        {
            reinterpret_cast<mgpu::ComputeShader *>(shader)->setBuffer(kernel, tag, *reinterpret_cast<mgpu::Buffer *>(buffer));
        }
        else
        {
            LOG(kDefLog, kError, "Invalid shader, kernel, tag, or buffer pointer");
        }
    }

    void mgpuDispatch(MGPUComputeShader *shader, const char *kernel, int groupsX, int groupsY, int groupsZ)
    {
        if (shader && kernel)
        {
            reinterpret_cast<mgpu::ComputeShader *>(shader)->dispatch(kernel, groupsX, groupsY, groupsZ);
        }
        else
        {
            LOG(kDefLog, kError, "Invalid shader or kernel pointer");
        }
    }

    void mgpuReadBufferSync(MGPUBuffer *buffer, void *outputData, size_t size)
    {
        if (buffer && outputData)
        {
            reinterpret_cast<mgpu::Buffer *>(buffer)->readSync(outputData, size);
        }
        else
        {
            LOG(kDefLog, kError, "Invalid buffer or outputData pointer");
        }
    }

    void mgpuReadBufferAsync(MGPUBuffer *buffer, void *outputData, size_t size, void (*callback)(void *), void *userData)
    {
        if (buffer && outputData && callback)
        {
            reinterpret_cast<mgpu::Buffer *>(buffer)->readAsync(outputData, size, callback, userData);
        }
        else
        {
            LOG(kDefLog, kError, "Invalid buffer, outputData, or callback pointer");
        }
    }

    void mgpuSetBufferData(MGPUBuffer *buffer, const void *inputData, size_t size)
    {
        if (buffer && inputData)
        {
            reinterpret_cast<mgpu::Buffer *>(buffer)->setData(inputData, size);
        }
        else
        {
            LOG(kDefLog, kError, "Invalid buffer or inputData pointer");
        }
    }

#ifdef __cplusplus
}
#endif// extern "C"
