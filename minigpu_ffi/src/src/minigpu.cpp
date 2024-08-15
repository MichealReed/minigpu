#include "../include/minigpu.h"


using namespace mgpu;
using namespace gpu;



extern "C"
{
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

    void mgpuLoadKernelString(MGPUComputeShader *shader, const char *kernelString)
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

    void mgpuLoadKernelFile(MGPUComputeShader *shader, const char *path)
    {
        if (shader && path)
        {
            reinterpret_cast<mgpu::ComputeShader *>(shader)->loadKernelFile(path);
        }
        else
        {
            LOG(kDefLog, kError, "Invalid shader or path pointer");
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

    MGPUBuffer *mgpuCreateBuffer(MGPUComputeShader *shader, int size, int memSize)
    {
        if (shader)
        {
            gpu::Array array = reinterpret_cast<mgpu::ComputeShader *>(shader)->createBuffer(size, memSize);
            return reinterpret_cast<MGPUBuffer *>(new mgpu::Buffer(array));
        }
        else
        {
            LOG(kDefLog, kError, "Invalid shader pointer");
            return nullptr;
        }
    }

    void mgpuDestroyBuffer(MGPUBuffer *buffer)
    {
        if (buffer)
        {
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

    void mgpuReadBufferSync(MGPUBuffer *buffer, MGPUBuffer *otherBuffer)
    {
        if (buffer && otherBuffer)
        {
            reinterpret_cast<mgpu::Buffer *>(buffer)->readSync(*reinterpret_cast<mgpu::Buffer *>(otherBuffer));
        }
        else
        {
            LOG(kDefLog, kError, "Invalid buffer or otherBuffer pointer");
        }
    }

    void mgpuReadBufferAsync(MGPUBuffer *buffer, MGPUBuffer *otherBuffer, void (*callback)(void *), void *userData)
    {
        if (buffer && otherBuffer && callback)
        {
            reinterpret_cast<mgpu::Buffer *>(buffer)->requestAsync(*reinterpret_cast<mgpu::Buffer *>(otherBuffer), callback, userData);
        }
        else
        {
            LOG(kDefLog, kError, "Invalid buffer, otherBuffer, or callback pointer");
        }
    }

} // extern "C"