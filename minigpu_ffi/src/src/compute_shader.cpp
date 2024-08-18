#include "../include/compute_shader.h"

using namespace gpu;

namespace mgpu
{

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
        bindings[bindingIndex] = gpu::Tensor{buffer.bufferData, gpu::Shape{buffer.bufferData.size}};
    }

    void ComputeShader::dispatch(const std::string &kernel, int groupsX, int groupsY, int groupsZ)
    {
        gpu::Kernel op = gpu::createKernel(ctx, code, bindings.data(), bindings.size(), nullptr, {static_cast<size_t>(groupsX), static_cast<size_t>(groupsY), static_cast<size_t>(groupsZ)});
        std::promise<void> promise;
        gpu::dispatchKernel(ctx, op, promise);
        auto future = promise.get_future();
        gpu::wait(ctx, future);
    }
}