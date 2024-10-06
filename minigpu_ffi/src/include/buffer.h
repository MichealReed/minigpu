#ifndef BUFFER_H
#define BUFFER_H

#include <string>
#include <future>
#include <vector>
#include <fstream>
#include "gpuh.h"

namespace mgpu
{
    class MGPU
    {
    public:
        void initializeContext();
        void destroyContext();

        gpu::Context &getContext() { return *ctx; }

    private:
        std::unique_ptr<gpu::Context> ctx;
    };

    class Buffer
    {
    public:
        Buffer(MGPU &mgpu);
        void createBuffer(int size, int memSize);
        void readSync(void *outputData, size_t size);
        void readAsync(void *outputData, size_t size, std::function<void(void *)> callback, void *userData);
        void setData(const float *inputData, size_t size);
        void release();

        gpu::Array bufferData;

    private:
        MGPU &mgpu;
    };

}
#endif // BUFFER_H