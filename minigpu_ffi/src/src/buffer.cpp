

#include "../include/buffer.h"
#include "../include/gpuh.h"

using namespace gpu;

namespace mgpu
{
    gpu::Context ctx;

    void initializeContext()
    {
        ctx = gpu::createContext();
    }

    void destroyContext()
    {
        delete &ctx;
    }

    Buffer::Buffer(gpu::Array data) : data(data) {}

    void Buffer::readSync(const Buffer &otherBuffer)
    {
        gpu::Tensor tensor{data, gpu::Shape{data.size}};
        gpu::toCPU(ctx, tensor, otherBuffer.data.buffer, otherBuffer.data.size);
    }
    void Buffer::requestAsync(const Buffer &otherBuffer, std::function<void(void *)> callback, void *userData)
    {
        gpu::CopyData op;
        op.future = op.promise.get_future();
        {
            WGPUBufferUsageFlags usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead;
            WGPUBufferDescriptor readbackBufferDescriptor = {
                .usage = usage,
                .size = otherBuffer.data.size,
            };
            op.readbackBuffer =
                wgpuDeviceCreateBuffer(ctx.device, &readbackBufferDescriptor);
        }
        {
            WGPUCommandEncoder commandEncoder;
            WGPUComputePassEncoder computePassEncoder;
            commandEncoder = wgpuDeviceCreateCommandEncoder(ctx.device, nullptr);
            wgpuCommandEncoderCopyBufferToBuffer(commandEncoder, data.buffer, 0,
                                                 op.readbackBuffer, 0, otherBuffer.data.size);
            op.commandBuffer = wgpuCommandEncoderFinish(commandEncoder, nullptr);
            gpu::check(op.commandBuffer, "Create command buffer", __FILE__, __LINE__);
        }
        gpu::CallbackData callbackData = {op.readbackBuffer, otherBuffer.data.size, otherBuffer.data.buffer,
                                          &op.promise, &op.future};
        wgpuQueueSubmit(ctx.queue, 1, &op.commandBuffer);
        wgpuQueueOnSubmittedWorkDone(
            ctx.queue,
            [](WGPUQueueWorkDoneStatus status, void *callbackData)
            {
                gpu::check(status == WGPUQueueWorkDoneStatus_Success, "Queue work done",
                           __FILE__, __LINE__);
                const auto *data = static_cast<gpu::CallbackData *>(callbackData);
                wgpuBufferMapAsync(
                    data->buffer, WGPUMapMode_Read, 0, data->bufferSize,
                    [](WGPUBufferMapAsyncStatus status, void *captureData)
                    {
                        const auto *data = static_cast<gpu::CallbackData *>(captureData);
                        gpu::check(status == WGPUBufferMapAsyncStatus_Success,
                                   "Map readbackBuffer", __FILE__, __LINE__);
                        const void *mappedData = wgpuBufferGetConstMappedRange(
                            data->buffer, /*offset=*/0, data->bufferSize);
                        gpu::check(mappedData, "Get mapped range", __FILE__, __LINE__);
                        memcpy(data->output, mappedData, data->bufferSize);
                        wgpuBufferUnmap(data->buffer);
                        data->promise->set_value();
                    },
                    callbackData);
            },
            &callbackData);
        gpu::wait(ctx, op.future);
        callback(userData);
    }

    void Buffer::release()
    {
        wgpuBufferRelease(data.buffer);
    }

    void ComputeShader::loadKernelString(const std::string &kernelString)
    {
        code = gpu::KernelCode{kernelString, {256, 1, 1}};
    }

    void ComputeShader::loadKernelFile(const std::string &path)
    {
        std::ifstream file(path);
        std::string kernelString((std::istreambuf_iterator<char>(file)),
                                 std::istreambuf_iterator<char>());
        loadKernelString(kernelString);
    }

    bool ComputeShader::hasKernel() const
    {
        return !code.data.empty();
    }

    gpu::Array ComputeShader::createBuffer(int size, int memSize)
    {
        WGPUBufferUsageFlags usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst | WGPUBufferUsage_CopySrc;
        WGPUBufferDescriptor descriptor = {
            .usage = usage,
            .size = static_cast<uint64_t>(memSize),
        };
        WGPUBuffer buffer = wgpuDeviceCreateBuffer(ctx.device, &descriptor);
        gpu::Array array = {
            .buffer = buffer,
            .usage = usage,
            .size = static_cast<size_t>(memSize),
        };
        return array;
    }

    void ComputeShader::setBuffer(const std::string &kernel, const std::string &tag, const Buffer &buffer)
    {
        // Find the binding index for the given tag in the kernel
        size_t bindingIndex = std::string::npos;
        for (size_t i = 0; i < code.data.size(); ++i)
        {
            std::string pattern = "@group(0) @binding(" + std::to_string(i) + ")";
            if (code.data.find(pattern) != std::string::npos)
            {
                bindingIndex = i;
                break;
            }
        }

        if (bindingIndex == std::string::npos)
        {
            LOG(kDefLog, kError, "Binding tag '%s' not found in kernel '%s'", tag.c_str(), kernel.c_str());
            return;
        }

        // Resize the bindings vector if necessary
        if (bindingIndex >= bindings.size())
        {
            bindings.resize(bindingIndex + 1);
        }
        // Store the Buffer object in the bindings vector at the corresponding index
        bindings[bindingIndex] = gpu::Tensor{buffer.data, gpu::Shape{buffer.data.size}};
    }

    void ComputeShader::dispatch(const std::string &kernel, int groupsX, int groupsY, int groupsZ)
    {
        gpu::Kernel op = gpu::createKernel(ctx, code, bindings.data(), bindings.size(), nullptr, {static_cast<size_t>(groupsX), static_cast<size_t>(groupsY), static_cast<size_t>(groupsZ)});
        std::promise<void> promise;
        gpu::dispatchKernel(ctx, op, promise);
        auto future = promise.get_future();
        gpu::wait(ctx, future);
    }
} // namespace mgpu