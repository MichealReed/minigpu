

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
        WGPUBufferUsage usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst | WGPUBufferUsage_CopySrc;
        // Zero‐initialize the descriptor.
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
        gpu::Tensor tensor{bufferData, gpu::Shape{bufferData.size}};
        gpu::toCPU(this->mgpu.getContext(), tensor, outputData, size);
    }

    void Buffer::readAsync(void *outputData, size_t size,
                           std::function<void(void *)> callback,
                           void *userData)
    {
        // Create a tensor for the GPU buffer.
        gpu::Tensor tensor{bufferData, gpu::Shape{bufferData.size}};

        // Prepare our copy data operation.
        gpu::CopyData op;
        op.future = op.promise.get_future();

        // Create a readback buffer with CopyDst and MapRead usages.
        {
            WGPUBufferDescriptor readbackBufferDescriptor = {
                .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead,
                .size = size,
            };
            op.readbackBuffer = wgpuDeviceCreateBuffer(
                this->mgpu.getContext().device, &readbackBufferDescriptor);
        }

        // Build a command buffer to copy from the source tensor buffer to the readback buffer.
        {
            WGPUCommandEncoder commandEncoder =
                wgpuDeviceCreateCommandEncoder(this->mgpu.getContext().device, nullptr);
            wgpuCommandEncoderCopyBufferToBuffer(commandEncoder,
                                                 tensor.data.buffer, 0,
                                                 op.readbackBuffer, 0, size);
            op.commandBuffer = wgpuCommandEncoderFinish(commandEncoder, nullptr);
            check(op.commandBuffer, "Create command buffer", __FILE__, __LINE__);
        }

        // Create a CallbackData instance which holds all state required by the callbacks.
        gpu::CallbackData callbackData = {op.readbackBuffer, size, outputData,
                                          &op.promise, &op.future};

        // Submit the command buffer.
        wgpuQueueSubmit(this->mgpu.getContext().queue, 1, &op.commandBuffer);

        // Set up the work-done callback info.
        WGPUQueueWorkDoneCallbackInfo workDoneCallbackInfo = {
            .mode = WGPUCallbackMode_AllowSpontaneous,
            .callback = [](WGPUQueueWorkDoneStatus status, void *userdata1, void *userdata2)
            {
                // Ensure that the submitted work completed successfully.
                check(status == WGPUQueueWorkDoneStatus_Success, "Queue work done",
                      __FILE__, __LINE__);
                auto *data = static_cast<gpu::CallbackData *>(userdata1);

                // Set up the buffer mapping callback info.
                WGPUBufferMapCallbackInfo mapCallbackInfo = {
                    .mode = WGPUCallbackMode_AllowSpontaneous,
                    .callback = [](WGPUMapAsyncStatus status, WGPUStringView message,
                                   void *userdata1, void *userdata2)
                    {
                        auto *data = static_cast<gpu::CallbackData *>(userdata1);
                        check(status == WGPUMapAsyncStatus_Success, "Map readbackBuffer",
                              __FILE__, __LINE__);
                        const void *mappedData = wgpuBufferGetConstMappedRange(
                            data->buffer, /*offset=*/0, data->bufferSize);
                        check(mappedData, "Get mapped range", __FILE__, __LINE__);
                        memcpy(data->output, mappedData, data->bufferSize);
                        wgpuBufferUnmap(data->buffer);
                        data->promise->set_value(); // Signal that the copy is done.
                    },
                    .userdata1 = data,
                    .userdata2 = nullptr};

                // Request the buffer to be mapped asynchronously.
                wgpuBufferMapAsync(data->buffer, WGPUMapMode_Read, 0, data->bufferSize,
                                   mapCallbackInfo);
            },
            .userdata1 = &callbackData,
            .userdata2 = nullptr};

        // Schedule the work-done callback.
        wgpuQueueOnSubmittedWorkDone(this->mgpu.getContext().queue, workDoneCallbackInfo);

        // Block until the asynchronous copy operation has completed.
        gpu::wait(this->mgpu.getContext(), op.future);

        // Finally, invoke the user callback.
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