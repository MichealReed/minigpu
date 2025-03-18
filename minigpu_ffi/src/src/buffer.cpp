

#include "../include/buffer.h"
#include "../include/compute_shader.h"
#include "../include/gpuh.h"

using namespace gpu;

namespace mgpu {
void MGPU::initializeContext() {
  try {
    // Wrap context in a unique_ptr.
    ctx = std::make_unique<gpu::Context>(std::move(gpu::createContext()));
    LOG(kDefLog, kInfo, "GPU context initialized successfully.");
  } catch (const std::exception &ex) {
    LOG(kDefLog, kError, "Failed to create GPU context: %s", ex.what());
  }
}

void MGPU::initializeContextAsync(std::function<void()> callback) {
  try {
    initializeContext();
    if (callback) {
      callback();
    }
  } catch (const std::exception &ex) {
    LOG(kDefLog, kError, "Failed to create GPU context: %s", ex.what());
  }
}

void MGPU::destroyContext() {}

Buffer::Buffer(MGPU &mgpu) : mgpu(mgpu) {
  bufferData.buffer = nullptr;
  bufferData.usage = 0;
  bufferData.size = 0;
}
void Buffer::createBuffer(int bufferSize) {
  WGPUBufferUsage usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst |
                          WGPUBufferUsage_CopySrc;
  WGPUBufferDescriptor descriptor = {};
  descriptor.usage = usage;
  descriptor.size = static_cast<uint64_t>(bufferSize);
  descriptor.mappedAtCreation = false;
  descriptor.label = {.data = nullptr, .length = 0};

  WGPUBuffer buffer =
      wgpuDeviceCreateBuffer(this->mgpu.getContext().device, &descriptor);
  if (buffer == nullptr) {
    LOG(kDefLog, kError, "Failed to create buffer");
    return;
  }

  bufferData = gpu::Array{
      .buffer = buffer,
      .usage = usage,
      .size = static_cast<size_t>(bufferSize),
  };
}

void Buffer::readSync(void *outputData, size_t size, size_t offset) {

  LOG(kDefLog, kInfo, "readSync: Reading %zu bytes from buffer", size);

  gpu::Tensor tensor{bufferData, gpu::Shape{bufferData.size}};

  // Perform the copy from GPU to CPU.
  gpu::toCPU(this->mgpu.getContext(), tensor, outputData, bufferData.size,
             offset);

  // Cast outputData to a float pointer to log some values.
  float *data = reinterpret_cast<float *>(outputData);
  size_t numFloats = size / sizeof(float);
  if (numFloats > 0) {
    // log all floats in single concatenated string
    std::string floatString = "readSync: Floats: ";
    for (size_t i = 0; i < numFloats; i++) {
      floatString += std::to_string(data[i]);
      if (i < numFloats - 1) {
        floatString += ", ";
      }
    }
    LOG(kDefLog, kInfo, floatString.c_str());
  } else {
    LOG(kDefLog, kInfo, "readSync: Not enough data to display float values");
  }

}

void Buffer::readAsync(void *outputData, size_t size, size_t offset,
                       std::function<void()> callback) {
  std::thread([=]() {
    // Perform the synchronous read. (May include any blocking call, etc.)
    readSync(outputData, size, offset);

    // Once readSync completes, if a callback was provided, notify the caller.
    if (callback) {
      callback();
    }
  }).detach();
}

void Buffer::setData(const float *inputData, size_t byteSize) {
  // Check if we need to create or resize the buffer
  if (bufferData.buffer == nullptr || byteSize > bufferData.size) {
    createBuffer(byteSize);
  }
  std::string bufferString = "mgpuSetBufferData: Buffer: ";
  for (size_t i = 0; i < byteSize / sizeof(float); i++) {
    bufferString += std::to_string(inputData[i]);
    if (i < byteSize / sizeof(float) - 1) {
      bufferString += ", ";
    }
  }
  LOG(kDefLog, kInfo, bufferString.c_str());

  // Copy the input data to the buffer using gpu::toGPU
  gpu::toGPU(this->mgpu.getContext(), inputData, bufferData.buffer, byteSize);

}

void Buffer::release() { wgpuBufferRelease(bufferData.buffer); }

} // namespace mgpu