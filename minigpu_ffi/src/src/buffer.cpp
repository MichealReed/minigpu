

#include "../include/buffer.h"
#include "../include/compute_shader.h"
#include "../include/gpuh.h"

using namespace gpu;

namespace mgpu
{
    void MGPU::initializeContext()
    {
        // Initialize the context using gpu::createContext
        gpu::Context* rawContext = new gpu::Context(gpu::createContext());
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
        delete &this->ctx;
    }

    Buffer::Buffer(MGPU &mgpu) : mgpu(mgpu) {}
    void Buffer::createBuffer(int size, int memSize)
    {

        // Proceed with buffer creation
        WGPUBufferUsageFlags usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst | WGPUBufferUsage_CopySrc;
        WGPUBufferDescriptor descriptor = {
            .usage = usage,
            .size = static_cast<uint64_t>(memSize),
        };

        LOG(kDefLog, kInfo, "mgpuCreateBuffer called with size: %d, memSize: %d", size, memSize);
        LOG(kDefLog, kInfo, "mgpuCreateBuffer called with device: %p", (void *)this->mgpu.getContext().device);
        LOG(kDefLog, kInfo, "mgpuCreateBuffer called with descriptor size: %llu, usage: %u",
            descriptor.size, descriptor.usage);

        WGPUBuffer buffer = wgpuDeviceCreateBuffer(this->mgpu.getContext().device, &descriptor);
        if (buffer == nullptr)
        {
            LOG(kDefLog, kError, "wgpuDeviceCreateBuffer failed to create buffer");
            return; // Handle buffer creation failure
        }

        gpu::Array array = {
            .buffer = buffer,
            .usage = usage,
            .size = static_cast<size_t>(memSize),
        };
        this->bufferData = array; // Assuming bufferDat assignment was incomplete in the snippet
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
        setLogLevel(4);
        // Ensure the buffer size is sufficient
        if (size > bufferData.size)
        {
            LOG(kDefLog, kError, "Buffer size is insufficient");
            return;
        }

        createBuffer(size / 4, size);

        LOG(kDefLog, kInfo, "mgpuSetBufferData called buffer last: %d, size: %d", bufferData.buffer, size);
        LOG(kDefLog, kInfo, "mgpuSetBufferData called inputData last: %d, size: %d", inputData, size);

        // Copy the input data to the buffer using gpu::toGPU
        gpu::toGPU(this->mgpu.getContext(), inputData, bufferData.buffer, size * sizeof(float));
    }

    void Buffer::release()
    {
        wgpuBufferRelease(bufferData.buffer);
    }

} // namespace mgpu