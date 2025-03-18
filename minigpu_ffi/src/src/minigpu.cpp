#include "../include/minigpu.h"
#ifdef __cplusplus
using namespace mgpu;
using namespace gpu;
extern "C" {
#endif

MGPU minigpu;

void mgpuInitializeContext() {
  minigpu.initializeContext();
  setLogLevel(4);
}

void mgpuInitializeContextAsync(MGPUCallback callback) {

  minigpu.initializeContextAsync(callback);
  setLogLevel(4);
}

void mgpuDestroyContext() { minigpu.destroyContext(); }

MGPUComputeShader *mgpuCreateComputeShader() {
  return reinterpret_cast<MGPUComputeShader *>(
      new mgpu::ComputeShader(minigpu));
}

void mgpuDestroyComputeShader(MGPUComputeShader *shader) {
  delete reinterpret_cast<mgpu::ComputeShader *>(shader);
}

void mgpuLoadKernel(MGPUComputeShader *shader, const char *kernelString) {
  if (!shader) {
    gpu::LOG(kDefLog, kError, "Invalid shader pointer (null)");
    return;
  }

  if (!kernelString) {
    gpu::LOG(kDefLog, kError, "Invalid kernelString pointer (null)");
    return;
  }

  if (strlen(kernelString) == 0) {
    gpu::LOG(kDefLog, kError, "Empty kernel string provided");
    return;
  }

  reinterpret_cast<mgpu::ComputeShader *>(shader)->loadKernelString(
      kernelString);
}

int mgpuHasKernel(MGPUComputeShader *shader) {
  if (shader) {
    return reinterpret_cast<mgpu::ComputeShader *>(shader)->hasKernel();
  } else {
    LOG(kDefLog, kError, "Invalid shader pointer");
    return 0;
  }
}

MGPUBuffer *mgpuCreateBuffer(int bufferSize) {
  auto *buf = new mgpu::Buffer(minigpu);
  buf->createBuffer(bufferSize);
  return reinterpret_cast<MGPUBuffer *>(buf);
}

void mgpuDestroyBuffer(MGPUBuffer *buffer) {
  if (buffer) {
    reinterpret_cast<mgpu::Buffer *>(buffer)->release();
    delete reinterpret_cast<mgpu::Buffer *>(buffer);
  } else {
    LOG(kDefLog, kError, "Invalid buffer pointer");
  }
}

void mgpuSetBuffer(MGPUComputeShader *shader, int tag, MGPUBuffer *buffer) {
  if (shader && tag >= 0 && buffer) {
    reinterpret_cast<mgpu::ComputeShader *>(shader)->setBuffer(
        tag, *reinterpret_cast<mgpu::Buffer *>(buffer));
  } else {
    LOG(kDefLog, kError, "Invalid shader, or buffer pointer");
  }
}

void mgpuDispatch(MGPUComputeShader *shader, int groupsX, int groupsY,
                  int groupsZ) {
  if (shader) {
    reinterpret_cast<mgpu::ComputeShader *>(shader)->dispatch(groupsX, groupsY,
                                                              groupsZ);
  } else {
    LOG(kDefLog, kError, "Invalid shader or kernel pointer");
  }
}

void mgpuDispatchAsync(MGPUComputeShader *shader, int groupsX, int groupsY,
                       int groupsZ, MGPUCallback callback) {
  if (shader) {
    reinterpret_cast<mgpu::ComputeShader *>(shader)->dispatchAsync(
        groupsX, groupsY, groupsZ, callback);
  } else {
    LOG(kDefLog, kError, "Invalid shader or kernel pointer");
  }
}

void mgpuReadBufferSync(MGPUBuffer *buffer, float *outputData, size_t size,
                        size_t offset) {
  if (buffer && outputData) {
    reinterpret_cast<mgpu::Buffer *>(buffer)->readSync(outputData, size,
                                                       offset ? offset : 0);
  } else {
    LOG(kDefLog, kError, "Invalid buffer or outputData pointer");
  }
}

void mgpuReadBufferAsync(MGPUBuffer *buffer, float *outputData, size_t size,
                         size_t offset, MGPUCallback callback) {
  if (buffer && outputData && callback) {
    reinterpret_cast<mgpu::Buffer *>(buffer)->readAsync(outputData, size,
                                                        offset, callback);
  } else {
    LOG(kDefLog, kError, "Invalid buffer, outputData, or callback pointer");
  }
}

void mgpuSetBufferData(MGPUBuffer *buffer, const float *inputData,
                       size_t byteSize) {
  if (buffer && inputData) {
    reinterpret_cast<mgpu::Buffer *>(buffer)->setData(inputData, byteSize);
  } else {
    LOG(kDefLog, kError, "Invalid buffer or inputData pointer");
  }
}

#ifdef __cplusplus
}
#endif // extern "C"
