

#include "../include/buffer.h"
#include "../include/compute_shader.h"
#include "../include/gpuh.h"

using namespace gpu;

namespace mgpu
{
    void MGPU::initializeContext() {
        try {
          // Wrap context in a unique_ptr.
          ctx = std::make_unique<gpu::Context>(std::move(gpu::createContext()));
          LOG(kDefLog, kInfo, "GPU context initialized successfully.");
        } catch (const std::exception &ex) {
          LOG(kDefLog, kError, "Failed to create GPU context: %s", ex.what());
        }
      }

    void MGPU::destroyContext()
    {
    }

    Buffer::Buffer(MGPU &mgpu) : mgpu(mgpu)
    {
        bufferData.buffer = nullptr;
        bufferData.usage = 0;
        bufferData.size = 0;
    }
    void Buffer::createBuffer(int numElements, int memSize)
    {
        WGPUBufferUsage usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst | WGPUBufferUsage_CopySrc;
        WGPUBufferDescriptor descriptor = {};
        descriptor.usage = usage;
        descriptor.size = static_cast<uint64_t>(memSize);
        descriptor.mappedAtCreation = false;
        descriptor.label = {.data = nullptr, .length = 0};

        LOG(kDefLog, kInfo, "Creating buffer with elements: %d, bytes: %d", numElements, memSize);

        WGPUBuffer buffer = wgpuDeviceCreateBuffer(this->mgpu.getContext().device, &descriptor);
        if (buffer == nullptr)
        {
            LOG(kDefLog, kError, "Failed to create buffer");
            return;
        }

        bufferData = gpu::Array{
            .buffer = buffer,
            .usage = usage,
            .size = static_cast<size_t>(memSize),
        };
    }

    void Buffer::readSync(void *outputData, size_t size)
    {
        // Log entry into readSync along with the pointer and size.
        LOG(kDefLog, kInfo, "Entering Buffer::readSync, outputData: %p, size: %zu", outputData, size);

        gpu::Tensor tensor{bufferData, gpu::Shape{bufferData.size}};

        // Perform the copy from GPU to CPU.
        gpu::toCPU(this->mgpu.getContext(), tensor, outputData, bufferData.size);

        // Cast outputData to a float pointer to log some values.
        float *data = reinterpret_cast<float *>(outputData);
        size_t numFloats = size / sizeof(float);
        if (numFloats > 0)
        {
            LOG(kDefLog, kInfo, "readSync: First float: %f, Last float: %f", data[0], data[numFloats - 1]);
            // log all floats in single concatenated string
            std::string floatString = "readSync: Floats: ";
            for (size_t i = 0; i < numFloats; i++)
            {
                floatString += std::to_string(data[i]);
                if (i < numFloats - 1)
                {
                    floatString += ", ";
                }
            }
            LOG(kDefLog, kInfo, floatString.c_str());
        }
        else
        {
            LOG(kDefLog, kInfo, "readSync: Not enough data to display float values");
        }

        LOG(kDefLog, kInfo, "Exiting Buffer::readSync");
    }

    void Buffer::readAsync(void *outputData, size_t size,
                           std::function<void(void *)> callback,
                           void *userData)
    {
        // NOT IMPLEMENTED
        LOG(kDefLog, kError, "Buffer::readAsync is not implemented");
    }

    void Buffer::setData(const float *inputData, size_t size)
    {
        // Check if we need to create or resize the buffer
        if (bufferData.buffer == nullptr || size > bufferData.size)
        {
            createBuffer(size / sizeof(float), size);
        }

        LOG(kDefLog, kInfo, "mgpuSetBufferData called buffer: %p, size: %zu", (void *)bufferData.buffer, size);
        // log elements of buffer as single concatenated string
        std::string bufferString = "mgpuSetBufferData: Buffer: ";
        for (size_t i = 0; i < size / sizeof(float); i++)
        {
            bufferString += std::to_string(inputData[i]);
            if (i < size / sizeof(float) - 1)
            {
                bufferString += ", ";
            }
        }
        LOG(kDefLog, kInfo, bufferString.c_str());

        // Copy the input data to the buffer using gpu::toGPU
        gpu::toGPU(this->mgpu.getContext(), inputData, bufferData.buffer, size);

        LOG(kDefLog, kInfo, "mgpuSetBufferData called inputData last: %f", inputData[size - 1]);
    }

    void Buffer::release()
    {
        wgpuBufferRelease(bufferData.buffer);
    }

} // namespace mgpu