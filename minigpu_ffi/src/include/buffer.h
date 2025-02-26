#ifndef BUFFER_H
#define BUFFER_H

#include "gpuh.h"
#include <fstream>
#include <future>
#include <string>
#include <vector>

namespace mgpu {
class MGPU {
public:
  void initializeContext();
  void initializeContextAsync(std::function<void()> callback);
  void destroyContext();

  gpu::Context &getContext() { return *ctx; }

private:
  std::unique_ptr<gpu::Context> ctx;
};

class Buffer {
public:
  Buffer(MGPU &mgpu);
  void createBuffer(int size, int memSize);
  void readSync(void *outputData, size_t size, size_t offset = 0);
  void readAsync(void *outputData, size_t size, size_t offset,
                 std::function<void()> callback);
  void setData(const float *inputData, size_t size);
  void release();

  gpu::Array bufferData;

private:
  MGPU &mgpu;
};

} // namespace mgpu
#endif // BUFFER_H