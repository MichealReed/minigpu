

#include "../include/buffer.h"
#include "../include/compute_shader.h"
#include "../include/gpuh.h"

using namespace gpu;

namespace mgpu
{
    void MGPU::initializeContext()
    {
        // Initialize the context using gpu::createContext
        gpu::Context *rawContext = new gpu::Context(gpu::createContext());
        if (rawContext)
        {
            ctx = std::unique_ptr<gpu::Context>(rawContext);
        }
        else
        {
            LOG(kDefLog, kError, "Failed to create GPU context");
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
        WGPUBufferUsageFlags usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst | WGPUBufferUsage_CopySrc;
        // Zero‐initialize the descriptor.
        WGPUBufferDescriptor descriptor = {};
        descriptor.usage = usage;
        descriptor.size = static_cast<uint64_t>(memSize);
        descriptor.mappedAtCreation = false;
        descriptor.label = nullptr; // or you can set a string if desired

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
        gpu::Tensor tensor{bufferData, gpu::Shape{bufferData.size}};
        gpu::toCPU(this->mgpu.getContext(), tensor, outputData, size);
    }

    void Buffer::readAsync(void *outputData, size_t size, std::function<void(void *)> callback, void *userData)
    {
        gpu::Tensor tensor{bufferData, gpu::Shape{bufferData.size}};
        gpu::CopyData op;
        op.future = op.promise.get_future();
        {
            WGPUBufferDescriptor readbackBufferDescriptor = {
                .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead,
                .size = size,
            };
            op.readbackBuffer = wgpuDeviceCreateBuffer(this->mgpu.getContext().device, &readbackBufferDescriptor);
        }
        {
            WGPUCommandEncoder commandEncoder;
            WGPUComputePassEncoder computePassEncoder;
            commandEncoder = wgpuDeviceCreateCommandEncoder(this->mgpu.getContext().device, nullptr);
            wgpuCommandEncoderCopyBufferToBuffer(commandEncoder, tensor.data.buffer, 0,
                                                 op.readbackBuffer, 0, size);
            op.commandBuffer = wgpuCommandEncoderFinish(commandEncoder, nullptr);
            check(op.commandBuffer, "Create command buffer", __FILE__, __LINE__);
        }
        gpu::CallbackData callbackData = {op.readbackBuffer, size, outputData, &op.promise, &op.future};
        wgpuQueueSubmit(this->mgpu.getContext().queue, 1, &op.commandBuffer);
        wgpuQueueOnSubmittedWorkDone(this->mgpu.getContext().queue, [](WGPUQueueWorkDoneStatus status, void *callbackData)
                                     {
            check(status == WGPUQueueWorkDoneStatus_Success, "Queue work done", __FILE__, __LINE__);
            const auto *data = static_cast<gpu::CallbackData *>(callbackData);
            wgpuBufferMapAsync(data->buffer, WGPUMapMode_Read, 0, data->bufferSize,
                               [](WGPUBufferMapAsyncStatus status, void *captureData) {
                                   const auto *data = static_cast<gpu::CallbackData *>(captureData);
                                   check(status == WGPUBufferMapAsyncStatus_Success, "Map readbackBuffer", __FILE__, __LINE__);
                                   const void *mappedData = wgpuBufferGetConstMappedRange(data->buffer, 0, data->bufferSize);
                                   check(mappedData, "Get mapped range", __FILE__, __LINE__);
                                   memcpy(data->output, mappedData, data->bufferSize);
                                   wgpuBufferUnmap(data->buffer);
                                   data->promise->set_value();
                               },
                               callbackData); }, &callbackData);
        gpu::wait(this->mgpu.getContext(), op.future);
        callback(userData);
    }

    void Buffer::setData(const float *inputData, size_t size)
    {
        setLogLevel(0);
        // Check if we need to create or resize the buffer
        if (bufferData.buffer == nullptr || size > bufferData.size)
        {
            createBuffer(size / sizeof(float), size);
        }

        LOG(kDefLog, kInfo, "mgpuSetBufferData called buffer: %p, size: %zu", (void *)bufferData.buffer, size);

        // Copy the input data to the buffer using gpu::toGPU
        gpu::toGPU(this->mgpu.getContext(), inputData, bufferData.buffer, size);

        LOG(kDefLog, kInfo, "mgpuSetBufferData called inputData last: %f", inputData[size - 1]);
    }

    void Buffer::release()
    {
        wgpuBufferRelease(bufferData.buffer);
    }

} // namespace mgpu