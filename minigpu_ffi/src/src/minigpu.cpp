#include "minigpu.h"
#include "gpu.h"

using namespace gpu;

extern "C"
{

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
        reinterpret_cast<mgpu::ComputeShader *>(shader)->loadKernelString(kernelString);
    }

    void mgpuLoadKernelFile(MGPUComputeShader *shader, const char *path)
    {
        reinterpret_cast<mgpu::ComputeShader *>(shader)->loadKernelFile(path);
    }

    int mgpuHasKernel(MGPUComputeShader *shader)
    {
        return reinterpret_cast<mgpu::ComputeShader *>(shader)->hasKernel();
    }

    MGPUBuffer *mgpuCreateBuffer(MGPUComputeShader *shader, uint32_t size, uint32_t memSize)
    {
        mgpu::Buffer buffer = reinterpret_cast<mgpu::ComputeShader *>(shader)->createBuffer(size, memSize);
        return reinterpret_cast<MGPUBuffer *>(new mgpu::Buffer(buffer));
    }

    void mgpuDestroyBuffer(MGPUBuffer *buffer)
    {
        delete reinterpret_cast<mgpu::Buffer *>(buffer);
    }

    void mgpuSetBuffer(MGPUComputeShader *shader, const char *kernel, const char *tag, MGPUBuffer *buffer)
    {
        reinterpret_cast<mgpu::ComputeShader *>(shader)->setBuffer(kernel, tag, *reinterpret_cast<mgpu::Buffer *>(buffer));
    }

    void mgpuDispatch(MGPUComputeShader *shader, const char *kernel, uint32_t groupsX, uint32_t groupsY, uint32_t groupsZ)
    {
        reinterpret_cast<mgpu::ComputeShader *>(shader)->dispatch(kernel, groupsX, groupsY, groupsZ);
    }

    void mgpuReadBufferSync(MGPUBuffer *buffer, MGPUBuffer *otherBuffer)
    {
        reinterpret_cast<mgpu::Buffer *>(buffer)->readSync(*reinterpret_cast<mgpu::Buffer *>(otherBuffer));
    }

    void mgpuReadBufferAsync(MGPUBuffer *buffer, MGPUBuffer *otherBuffer, void (*callback)(void *), void *userData)
    {
        reinterpret_cast<mgpu::Buffer *>(buffer)->requestAsync(*reinterpret_cast<mgpu::Buffer *>(otherBuffer), callback, userData);
    }

} // extern "C"

namespace mgpu
{

    Buffer::Buffer(gpu::Array data) : data(data) {}

    void Buffer::readSync(Buffer &otherBuffer)
    {
        gpu::toCPU(ctx, data, otherBuffer.data.buffer, otherBuffer.data.size);
    }

    void Buffer::requestAsync(Buffer &otherBuffer, std::function<void(void *)> callback, void *userData)
    {
        gpu::CopyData op;
        op.future = op.promise.get_future();
        {
            WGPUBufferDescriptor readbackBufferDescriptor = {
                .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead,
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
        gpu::CallbackData callbackData = {op.readbackBuffer, otherBuffer.data.size, otherBuffer.data.buffer, &op.promise,
                                          &op.future};
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

    void ComputeShader::loadKernelString(const std::string& kernelString) {
  code = gpu::KernelCode(kernelString);
}

void ComputeShader::loadKernelFile(const std::string& path) {
  std::ifstream file(path);
  std::string kernelString((std::istreambuf_iterator<char>(file)),
                            std::istreambuf_iterator<char>());
  loadKernelString(kernelString);
}

bool ComputeShader::hasKernel() {
  return !code.data.empty();
}

Buffer ComputeShader::createBuffer(uint32_t size, uint32_t memSize) {
  gpu::Array data = gpu::createTensor(ctx, {size}, gpu::kf32);
  return Buffer(data);
}

void ComputeShader::setBuffer(const std::string& kernel, const std::string& tag, Buffer& buffer) {
  // Find the binding index for the given tag in the kernel
  size_t bindingIndex = -1;
  for (size_t i = 0; i < code.data.size(); ++i) {
    std::string pattern = "@group(0) @binding(" + std::to_string(i) + ")";
    if (code.data.find(pattern) != std::string::npos) {
      bindingIndex = i;
      break;
    }
  }

  if (bindingIndex == -1) {
    LOG(kDefLog, kError, "Binding tag '%s' not found in kernel '%s'", tag.c_str(), kernel.c_str());
    return;
  }

  // Resize the bindings vector if necessary
  if (bindingIndex >= bindings.size()) {
    bindings.resize(bindingIndex + 1);
  }

  // Store the Buffer object in the bindings vector at the corresponding index
  bindings[bindingIndex] = gpu::Tensor{buffer.data, gpu::Shape{buffer.data.size}};
}



void ComputeShader::dispatch(const std::string& kernel, uint32_t groupsX, uint32_t groupsY, uint32_t groupsZ) {
  gpu::Kernel op = gpu::createKernel(ctx, code, bindings, {groupsX, groupsY, groupsZ});
  std::promise<void> promise;
  std::future<void> future = promise.get_future();
  gpu::dispatchKernel(ctx, op, promise);
  gpu::wait(ctx, future);
}

} // namespace mgpu
