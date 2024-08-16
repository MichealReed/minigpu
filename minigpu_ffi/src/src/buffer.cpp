

#include "../include/buffer.h"
#include "../include/compute_shader.h"
#include "../include/gpuh.h"

using namespace gpu;

namespace mgpu
{
    gpu::Context ctx;

    void initializeContext()
    {
        ctx = gpu::createContext({});
    }

    void destroyContext()
    {
        delete &ctx;
    }

    Buffer::Buffer(gpu::Array bufferData) : bufferData(bufferData) {}

    void Buffer::readSync(void *outputData, size_t size)
    {
        gpu::Tensor tensor{bufferData, gpu::Shape{bufferData.size}};
        gpu::toCPU(ctx, tensor, outputData, size);
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
            op.readbackBuffer = wgpuDeviceCreateBuffer(ctx.device, &readbackBufferDescriptor);
        }
        {
            WGPUCommandEncoder commandEncoder;
            WGPUComputePassEncoder computePassEncoder;
            commandEncoder = wgpuDeviceCreateCommandEncoder(ctx.device, nullptr);
            wgpuCommandEncoderCopyBufferToBuffer(commandEncoder, tensor.data.buffer, 0,
                                                 op.readbackBuffer, 0, size);
            op.commandBuffer = wgpuCommandEncoderFinish(commandEncoder, nullptr);
            check(op.commandBuffer, "Create command buffer", __FILE__, __LINE__);
        }
        gpu::CallbackData callbackData = {op.readbackBuffer, size, outputData, &op.promise, &op.future};
        wgpuQueueSubmit(ctx.queue, 1, &op.commandBuffer);
        wgpuQueueOnSubmittedWorkDone(ctx.queue, [](WGPUQueueWorkDoneStatus status, void *callbackData)
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
        gpu::wait(ctx, op.future);
        callback(userData);
    }

    void Buffer::setData(const void *inputData, size_t size)
    {
        // Ensure the buffer size is sufficient
        if (size > bufferData.size)
        {
            LOG(kDefLog, kError, "Buffer size is insufficient");
            return;
        }

        // Copy the input data to the buffer using gpu::toGPU
        gpu::toGPU(ctx, inputData, bufferData.buffer, size);
    }

    void Buffer::release()
    {
        wgpuBufferRelease(bufferData.buffer);
    }

} // namespace mgpu