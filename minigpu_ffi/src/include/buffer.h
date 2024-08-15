#ifndef BUFFER_H
#define BUFFER_H

#include <string>
#include <future>
#include <vector>
#include <fstream>
#include "gpuh.h"

namespace mgpu
{
    extern gpu::Context ctx;
    void initializeContext();
    void destroyContext();
    class Buffer
    {
    public:
        Buffer(gpu::Array data);
        void readSync(const Buffer &otherBuffer);
        void requestAsync(const Buffer &otherBuffer, std::function<void(void *)> callback, void *userData);
        void release();

        gpu::Array data;
    };

    class ComputeShader
    {
    public:
        void loadKernelString(const std::string &kernelString);
        void loadKernelFile(const std::string &path);
        bool hasKernel() const;
        gpu::Array createBuffer(int size, int memSize);
        void setBuffer(const std::string &kernel, const std::string &tag, const Buffer &buffer);
        void dispatch(const std::string &kernel, int groupsX, int groupsY, int groupsZ);

    private:
        gpu::KernelCode code;
        std::vector<gpu::Tensor> bindings;
    };
}
#endif // BUFFER_H