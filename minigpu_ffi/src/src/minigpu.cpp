#include "../include/minigpu.h"
#ifdef __cplusplus
using namespace mgpu;
using namespace gpu;
extern "C"
{
#endif

    MGPU minigpu;

    void mgpuInitializeContext()
    {
        minigpu.initializeContext();
    }

    void mgpuDestroyContext()
    {
        minigpu.destroyContext();
    }

    MGPUComputeShader *mgpuCreateComputeShader()
    {
        return reinterpret_cast<MGPUComputeShader *>(new mgpu::ComputeShader(minigpu));
    }

    void mgpuDestroyComputeShader(MGPUComputeShader *shader)
    {
        delete reinterpret_cast<mgpu::ComputeShader *>(shader);
    }

    void mgpuLoadKernel(MGPUComputeShader *shader, const char *kernelString)
    {
        if (!shader)
        {
            gpu::LOG(kDefLog, kError, "Invalid shader pointer (null)");
            return;
        }

        if (!kernelString)
        {
            gpu::LOG(kDefLog, kError, "Invalid kernelString pointer (null)");
            return;
        }

        if (strlen(kernelString) == 0)
        {
            gpu::LOG(kDefLog, kError, "Empty kernel string provided");
            return;
        }

        // Log first 100 characters of the kernel string for debugging
        char previewStr[101] = {0};
        strncpy(previewStr, kernelString, 100);
        gpu::LOG(kDefLog, kInfo, "Loading kernel (preview): %s%s",
                 previewStr, strlen(kernelString) > 100 ? "..." : "");

        reinterpret_cast<mgpu::ComputeShader *>(shader)->loadKernelString(kernelString);
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

    MGPUBuffer *mgpuCreateBuffer(int size, int memSize)
    {
        auto *buf = new mgpu::Buffer(minigpu);
        buf->createBuffer(size, memSize);
        return reinterpret_cast<MGPUBuffer *>(buf);
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

    void mgpuSetBuffer(MGPUComputeShader *shader, const char *tag, MGPUBuffer *buffer)
    {
        if (shader && tag && buffer)
        {
            reinterpret_cast<mgpu::ComputeShader *>(shader)->setBuffer(tag, *reinterpret_cast<mgpu::Buffer *>(buffer));
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

    void mgpuReadBufferSync(MGPUBuffer *buffer, float *outputData, size_t size)
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

    void mgpuSetBufferData(MGPUBuffer *buffer, const float *inputData, size_t size)
    {
        printf("mgpuSetBufferData called\n");
        printf("buffer: %p\n", buffer);
        if (buffer && inputData)
        {
            reinterpret_cast<mgpu::Buffer *>(buffer)->setData(inputData, size);
            LOG(kDefLog, kInfo, "mgpuSetBufferData called inputData last: %f", inputData[size - 1]);
        }
        else
        {
            LOG(kDefLog, kError, "Invalid buffer or inputData pointer");
        }
    }

#ifdef __cplusplus
}
#endif // extern "C"
