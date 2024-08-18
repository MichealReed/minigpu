#include "../include/compute_shader.h"

using namespace gpu;

namespace mgpu
{

    ComputeShader::ComputeShader(MGPU &mgpu) : mgpu(mgpu) {}

    void ComputeShader::loadKernelString(const std::string &kernelString)
    {
        // Find the @workgroup_size attribute in the WGSL string
        std::string workgroupSizeAttribute = "@workgroup_size";
        size_t attributePos = kernelString.find(workgroupSizeAttribute);
        if (attributePos != std::string::npos)
        {
            // Extract the workgroup size values
            size_t startPos = attributePos + workgroupSizeAttribute.length();
            size_t endPos = kernelString.find(")", startPos);
            if (endPos != std::string::npos)
            {
                std::string workgroupSizeStr = kernelString.substr(startPos, endPos - startPos);
                std::istringstream iss(workgroupSizeStr);
                std::string sizeX, sizeY, sizeZ;
                if (std::getline(iss, sizeX, ',') &&
                    std::getline(iss, sizeY, ',') &&
                    std::getline(iss, sizeZ, ')'))
                {
                    // Convert the size values to integers
                    size_t x = std::stoi(sizeX);
                    size_t y = std::stoi(sizeY);
                    size_t z = std::stoi(sizeZ);
                    // Use the extracted workgroup size values
                    code = gpu::KernelCode{kernelString, {x, y, z}};
                    return;
                }
            }
        }
        // If workgroup size is not found or parsing fails, use default values
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
        gpu::Kernel op = gpu::createKernel(this->mgpu.ctx, code, bindings.data(), bindings.size(), nullptr, {static_cast<size_t>(groupsX), static_cast<size_t>(groupsY), static_cast<size_t>(groupsZ)});
        std::promise<void> promise;
        gpu::dispatchKernel(this->mgpu.ctx, op, promise);
        auto future = promise.get_future();
        gpu::wait(this->mgpu.ctx, future);
    }
}