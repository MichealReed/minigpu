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
        gpu::Context ctx;
        void initializeContext();
        void destroyContext();
    };

    class Buffer
    {
    public:
        Buffer(gpu::Array buffer, MGPU &mgpu);
        gpu::Array createBuffer(int size, int memSize);
        void readSync(void *outputData, size_t size);
        void readAsync(void *outputData, size_t size, std::function<void(void *)> callback, void *userData);
        void setData(const void *inputData, size_t size);
        void release();

        gpu::Array bufferData;

    private:
        MGPU &mgpu;
    };

}
#endif // BUFFER_H