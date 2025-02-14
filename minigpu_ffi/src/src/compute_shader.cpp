#include "../include/compute_shader.h"
#include <sstream>
#include <stdexcept>

using namespace gpu;

namespace mgpu
{

    ComputeShader::ComputeShader(MGPU &mgpu) : mgpu(mgpu) {}

    void ComputeShader::loadKernelString(const std::string &kernelString)
    {
        code = KernelCode{
            kernelString,
            Shape{256, 1, 1},
            kf32};
    }

    void ComputeShader::loadKernelFile(const std::string &path)
    {
        std::ifstream file(path);
        if (!file.is_open())
        {
            throw std::runtime_error("Failed to open kernel file: " + path);
        }
        std::string kernelString((std::istreambuf_iterator<char>(file)),
                                 std::istreambuf_iterator<char>());
        loadKernelString(kernelString);
    }

    bool ComputeShader::hasKernel() const
    {
        return !code.data.empty();
    }

    void ComputeShader::setBuffer(const std::string &tag, const Buffer &buffer)
    {
        size_t bindingIndex = (tag == "inp") ? 0 : (tag == "out") ? 1
                                                                  : throw std::runtime_error("Invalid buffer tag");

        // For the input buffer, we assume it's already been created properly.
        // For output, if buffer.bufferData.size is zero, we use a fallback value.
        size_t numElements = 0;
        if (buffer.bufferData.buffer != nullptr && buffer.bufferData.size > 0)
        {
            numElements = buffer.bufferData.size / sizeof(float);
            
        }
        Shape shape{numElements};

        bindings[bindingIndex] = Tensor{
            .data = buffer.bufferData,
            .shape = shape};
    }

    void ComputeShader::dispatch(const std::string &kernelSource, int groupsX, int groupsY, int groupsZ)
    {
        if (!bindings[0].data.buffer || !bindings[1].data.buffer)
        {
            throw std::runtime_error("Input and output buffers must be set before dispatch");
        }

        Bindings<2> gpu_bindings{{bindings[0], bindings[1]}};

        Kernel op = createKernel(
            mgpu.getContext(),
            code,
            gpu_bindings,
            {static_cast<size_t>(groupsX),
             static_cast<size_t>(groupsY),
             static_cast<size_t>(groupsZ)});

        std::promise<void> promise;
        auto future = promise.get_future();

        dispatchKernel(mgpu.getContext(), op, promise);
        wait(mgpu.getContext(), future);
    }

} // namespace mgpu