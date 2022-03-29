#include <string>
#include <iostream>
#include <vulkan/vulkan_raii.hpp>
#include <GLFW/glfw3.h>
#include <SPIRV/GlslangToSpv.h>
#include <StandAlone/ResourceLimits.h>

const std::string externalForceShader = R"(
#version 460
layout(local_size_x = 1, local_size_y = 1) in;
layout(binding = 0, rgba8) uniform image2D velocityImage;
layout(binding = 1) uniform UniformBufferObject {
    vec2 mousePosition;
    vec2 mouseMove;
} ubo;
void main()
{
    //vec2 uv = gl_GlobalInvocationID.xy / (vec2(gl_NumWorkGroups) * vec2(gl_WorkGroupSize));
    //imageStore(velocityImage, ivec2(gl_GlobalInvocationID.xy), vec4(uv, 0, 1));
    imageStore(velocityImage, ivec2(gl_GlobalInvocationID.xy), vec4(ubo.mousePosition, 0, 1));
}
)";

const std::string renderShader = R"(
#version 460
layout(local_size_x = 1, local_size_y = 1) in;
layout(binding = 0, rgba8) uniform image2D renderImage;
layout(binding = 1, rgba8) uniform image2D velocityImage;
void main()
{
    vec4 velocity = imageLoad(velocityImage, ivec2(gl_GlobalInvocationID.xy));
    imageStore(renderImage, ivec2(gl_GlobalInvocationID.xy), velocity);
}
)";

VKAPI_ATTR VkBool32 VKAPI_CALL debugUtilsMessengerCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
    VkDebugUtilsMessageTypeFlagsEXT messageTypes,
    VkDebugUtilsMessengerCallbackDataEXT const* pCallbackData,
    void* pUserData)
{
    std::cerr << pCallbackData->pMessage << std::endl;
    return VK_FALSE;
}

std::vector<unsigned int> compileToSPV(const std::string& glslShader)
{
    EShLanguage stage = EShLangCompute;
    glslang::InitializeProcess();

    const char* shaderStrings[1];
    shaderStrings[0] = glslShader.data();

    glslang::TShader shader(stage);
    shader.setStrings(shaderStrings, 1);

    EShMessages messages = (EShMessages)(EShMsgSpvRules | EShMsgVulkanRules);

    if (!shader.parse(&glslang::DefaultTBuiltInResource, 100, false, messages)) {
        throw std::runtime_error(shader.getInfoLog());
    }

    glslang::TProgram program;
    program.addShader(&shader);

    if (!program.link(messages)) {
        throw std::runtime_error(shader.getInfoLog());
    }

    std::vector<unsigned int> spvShader;
    glslang::GlslangToSpv(*program.getIntermediate(stage), spvShader);
    glslang::FinalizeProcess();

    return spvShader;
}

void setImageLayout(const vk::raii::CommandBuffer& commandBuffer,
                    vk::Image image, vk::ImageLayout oldLayout, vk::ImageLayout newLayout)
{
    vk::PipelineStageFlags srcStageMask = vk::PipelineStageFlagBits::eAllCommands;
    vk::PipelineStageFlags dstStageMask = vk::PipelineStageFlagBits::eAllCommands;

    vk::ImageMemoryBarrier barrier{};
    barrier.setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED);
    barrier.setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED);
    barrier.setImage(image);
    barrier.setOldLayout(oldLayout);
    barrier.setNewLayout(newLayout);
    barrier.setSubresourceRange({ vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 });

    switch (oldLayout) {
    case vk::ImageLayout::eTransferSrcOptimal:
        barrier.srcAccessMask = vk::AccessFlagBits::eTransferRead;
        break;
    case vk::ImageLayout::eTransferDstOptimal:
        barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
        break;
    default:
        break;
    }
    switch (newLayout) {
    case vk::ImageLayout::eTransferDstOptimal:
        barrier.dstAccessMask = vk::AccessFlagBits::eTransferWrite;
        break;
    case vk::ImageLayout::eTransferSrcOptimal:
        barrier.dstAccessMask = vk::AccessFlagBits::eTransferRead;
        break;
    default:
        break;
    }
    commandBuffer.pipelineBarrier(srcStageMask, dstStageMask, {}, {}, {}, barrier);
}

struct UniformBufferObject
{
    float mousePosition[2];
    float mouseMove[2];
};

uint32_t findMemoryTypeIndex(const vk::raii::PhysicalDevice& physicalDevice,
                             vk::MemoryRequirements requirements)
{
    uint32_t memoryTypeIndex;
    vk::PhysicalDeviceMemoryProperties memoryProperties = physicalDevice.getMemoryProperties();
    for (uint32_t index = 0; index < memoryProperties.memoryTypeCount; ++index) {
        if (requirements.memoryTypeBits & (1 << index)) {
            memoryTypeIndex = index;
        }
    }
    return memoryTypeIndex;
}

struct Image
{
    Image(const vk::raii::Device& device,
          const vk::raii::PhysicalDevice& physicalDevice,
          const vk::raii::CommandBuffer& commandBuffer,
          const vk::raii::Queue& queue,
          int width, int height)
        : image{ device, makeImageCreateInfo(width, height) }
        , memory{ device, makeMemoryAllocationInfo(device, physicalDevice) }
        , view{ device, makeImageViewCreateInfo() }
    {
        // Set image layout
        commandBuffer.begin(vk::CommandBufferBeginInfo{});
        setImageLayout(commandBuffer, *image, vk::ImageLayout::eUndefined, vk::ImageLayout::eGeneral);
        commandBuffer.end();

        vk::SubmitInfo submitInfo;
        submitInfo.setCommandBuffers(*commandBuffer);
        queue.submit(submitInfo);
        queue.waitIdle();
    }

    vk::ImageCreateInfo makeImageCreateInfo(uint32_t width, uint32_t height)
    {
        vk::ImageCreateInfo imageCreateInfo;
        imageCreateInfo.setImageType(vk::ImageType::e2D);
        imageCreateInfo.setFormat(vk::Format::eB8G8R8A8Unorm);
        imageCreateInfo.setExtent({ width, height, 1 });
        imageCreateInfo.setMipLevels(1);
        imageCreateInfo.setArrayLayers(1);
        imageCreateInfo.setUsage(vk::ImageUsageFlagBits::eStorage |
                                 vk::ImageUsageFlagBits::eTransferSrc |
                                 vk::ImageUsageFlagBits::eTransferDst);
        return imageCreateInfo;
    }

    vk::MemoryAllocateInfo makeMemoryAllocationInfo(const vk::raii::Device& device,
                                                    const vk::raii::PhysicalDevice& physicalDevice)
    {
        vk::MemoryRequirements requirements = image.getMemoryRequirements();
        uint32_t memoryTypeIndex = findMemoryTypeIndex(physicalDevice, requirements);

        vk::MemoryAllocateInfo memoryAllocateInfo;
        memoryAllocateInfo.setAllocationSize(requirements.size);
        memoryAllocateInfo.setMemoryTypeIndex(memoryTypeIndex);
        return memoryAllocateInfo;
    }

    vk::ImageViewCreateInfo makeImageViewCreateInfo()
    {
        image.bindMemory(*memory, 0);

        vk::ImageViewCreateInfo imageViewCreateInfo;
        imageViewCreateInfo.setImage(*image);
        imageViewCreateInfo.setViewType(vk::ImageViewType::e2D);
        imageViewCreateInfo.setFormat(vk::Format::eB8G8R8A8Unorm);
        imageViewCreateInfo.setSubresourceRange({ vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 });
        return imageViewCreateInfo;
    }

    vk::raii::Image image;
    vk::raii::DeviceMemory memory;
    vk::raii::ImageView view;
};

struct Buffer
{
    Buffer(const vk::raii::Device& device,
           const vk::raii::PhysicalDevice& physicalDevice,
           const vk::raii::CommandBuffer& commandBuffer,
           const vk::raii::Queue& queue,
           vk::DeviceSize size)
        : size{ size }
        , buffer{ device, makeBufferCreateInfo(size) }
        , memory{ device, makeMemoryAllocationInfo(device, physicalDevice) }
    {
        buffer.bindMemory(*memory, 0);
    }

    vk::BufferCreateInfo makeBufferCreateInfo(vk::DeviceSize size)
    {
        vk::BufferCreateInfo createInfo;
        createInfo.setSize(size);
        createInfo.setUsage(vk::BufferUsageFlagBits::eUniformBuffer);
        return createInfo;
    }

    vk::MemoryAllocateInfo makeMemoryAllocationInfo(const vk::raii::Device& device,
                                                    const vk::raii::PhysicalDevice& physicalDevice)
    {
        vk::MemoryPropertyFlags properties = vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent;
        vk::MemoryRequirements requirements = buffer.getMemoryRequirements();
        uint32_t memoryTypeIndex = findMemoryTypeIndex(physicalDevice, requirements);

        vk::MemoryAllocateInfo memoryAllocateInfo;
        memoryAllocateInfo.setAllocationSize(requirements.size);
        memoryAllocateInfo.setMemoryTypeIndex(memoryTypeIndex);
        return memoryAllocateInfo;
    }

    void copy(void* data)
    {
        if (!mapped) {
            mapped = memory.mapMemory(0, size);
        }
        memcpy(mapped, data, size);
    }

    vk::DeviceSize size;
    vk::raii::Buffer buffer;
    vk::raii::DeviceMemory memory;
    void* mapped = nullptr;
};

struct ComputeKernel
{
    ComputeKernel(const vk::raii::Device& device,
                  const std::string& code,
                  const std::vector<vk::DescriptorSetLayoutBinding>& bindings,
                  const vk::raii::DescriptorPool& descPool)
        : shaderModule{ device, makeShaderModuleCreateInfo(code) }
        , descSetLayout{ device, makeDescSetLayoutCreateInfo(bindings) }
        , pipelineLayout{ device, makePipelineLayoutCreateInfo(device) }
        , pipeline{ device, nullptr, makeComputePipelineCreateInfo(device) }
        , descSet{ std::move(vk::raii::DescriptorSets{ device, makeDescriptorSetAllocateInfo(descPool)}.front()) }
    {
    }

    vk::ShaderModuleCreateInfo makeShaderModuleCreateInfo(const std::string& code)
    {
        spirvCode = compileToSPV(code);
        vk::ShaderModuleCreateInfo createInfo;
        createInfo.setCode(spirvCode);
        return createInfo;
    }

    vk::DescriptorSetLayoutCreateInfo makeDescSetLayoutCreateInfo(
        const std::vector<vk::DescriptorSetLayoutBinding>& bindings)
    {
        vk::DescriptorSetLayoutCreateInfo createInfo;
        createInfo.setBindings(bindings);
        return createInfo;
    }

    vk::PipelineShaderStageCreateInfo makePipelineShaderStageCreateInfo()
    {
        vk::PipelineShaderStageCreateInfo createInfo;
        createInfo.setStage(vk::ShaderStageFlagBits::eCompute);
        createInfo.setModule(*shaderModule);
        createInfo.setPName("main");
        return createInfo;
    }

    vk::PipelineLayoutCreateInfo makePipelineLayoutCreateInfo(const vk::raii::Device& device)
    {
        vk::PipelineLayoutCreateInfo createInfo;
        createInfo.setSetLayouts(*descSetLayout);
        return createInfo;
    }

    vk::ComputePipelineCreateInfo makeComputePipelineCreateInfo(const vk::raii::Device& device)
    {
        vk::ComputePipelineCreateInfo createInfo;
        createInfo.setStage(makePipelineShaderStageCreateInfo());
        createInfo.setLayout(*pipelineLayout);
        return createInfo;
    }

    vk::DescriptorSetAllocateInfo makeDescriptorSetAllocateInfo(const vk::raii::DescriptorPool& descPool)
    {
        vk::DescriptorSetAllocateInfo allocateInfo;
        allocateInfo.setDescriptorPool(*descPool);
        allocateInfo.setSetLayouts(*descSetLayout);
        return allocateInfo;
    }

    void updateDescriptorSet(const vk::raii::Device& device, uint32_t binding, uint32_t count, const Image& image)
    {
        vk::DescriptorImageInfo descImageInfo;
        descImageInfo.setImageView(*image.view);
        descImageInfo.setImageLayout(vk::ImageLayout::eGeneral);

        vk::WriteDescriptorSet imageWrite;
        imageWrite.setDstSet(*descSet);
        imageWrite.setDescriptorType(vk::DescriptorType::eStorageImage);
        imageWrite.setDescriptorCount(count);
        imageWrite.setDstBinding(binding);
        imageWrite.setImageInfo(descImageInfo);

        device.updateDescriptorSets(imageWrite, nullptr);
    }

    void updateDescriptorSet(const vk::raii::Device& device, uint32_t binding, uint32_t count, const Buffer& buffer)
    {
        vk::DescriptorBufferInfo descBufferInfo;
        descBufferInfo.setBuffer(*buffer.buffer);
        descBufferInfo.setOffset(0);
        descBufferInfo.setRange(buffer.size);

        vk::WriteDescriptorSet bufferWrite;
        bufferWrite.setDstSet(*descSet);
        bufferWrite.setDescriptorType(vk::DescriptorType::eUniformBuffer);
        bufferWrite.setDescriptorCount(count);
        bufferWrite.setDstBinding(binding);
        bufferWrite.setBufferInfo(descBufferInfo);

        device.updateDescriptorSets(bufferWrite, nullptr);
    }

    void run(const vk::raii::CommandBuffer& commandBuffer, uint32_t groupCountX, uint32_t groupCountY)
    {
        commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, *pipeline);
        commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, *pipelineLayout, 0, *descSet, nullptr);
        commandBuffer.dispatch(groupCountX, groupCountY, 1);
    }

    std::vector<unsigned int> spirvCode;
    vk::raii::ShaderModule shaderModule;
    vk::raii::DescriptorSetLayout descSetLayout;
    vk::raii::PipelineLayout pipelineLayout;
    vk::raii::Pipeline pipeline;
    vk::raii::DescriptorSet descSet;
};

int main()
{
    try {
        // Create window
        const int width = 1280;
        const int height = 720;
        glfwInit();
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
        GLFWwindow* window = glfwCreateWindow(width, height, "Vulkan", nullptr, nullptr);

        // Create context
        vk::raii::Context context;

        // Gather extensions
        uint32_t glfwExtensionCount = 0;
        const char** glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
        std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);
        extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);

        // Gather layers
        std::vector<const char*> layers{ "VK_LAYER_KHRONOS_validation" };

        // Create instance
        vk::ApplicationInfo appInfo;
        appInfo.apiVersion = VK_API_VERSION_1_2;
        vk::InstanceCreateInfo instanceCreateInfo;
        instanceCreateInfo.setPApplicationInfo(&appInfo);
        instanceCreateInfo.setPEnabledExtensionNames(extensions);
        instanceCreateInfo.setPEnabledLayerNames(layers);
        vk::raii::Instance instance{ context, instanceCreateInfo };

        // Create debug messenger
        vk::DebugUtilsMessengerCreateInfoEXT debugMessengerCreateInfo;
        debugMessengerCreateInfo.setMessageSeverity(vk::DebugUtilsMessageSeverityFlagBitsEXT::eError);
        debugMessengerCreateInfo.setMessageType(vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation);
        debugMessengerCreateInfo.setPfnUserCallback(&debugUtilsMessengerCallback);
        vk::raii::DebugUtilsMessengerEXT debugUtilsMessenger{ instance, debugMessengerCreateInfo };

        // Pick first physical device
        vk::raii::PhysicalDevice physicalDevice = vk::raii::PhysicalDevices{ instance }.front();

        // Create surface
        VkSurfaceKHR _surface;
        if (glfwCreateWindowSurface(VkInstance(*instance), window, nullptr, &_surface) != VK_SUCCESS) {
            throw std::runtime_error("failed to create window surface!");
        }
        vk::raii::SurfaceKHR surface{ instance, _surface };

        // Find queue families
        uint32_t computeFamily;
        uint32_t presentFamily;
        std::vector queueFamilyProperties = physicalDevice.getQueueFamilyProperties();
        for (uint32_t index = 0; index < queueFamilyProperties.size(); index++) {
            if (queueFamilyProperties[index].queueFlags & vk::QueueFlagBits::eCompute) {
                computeFamily = index;
            }
            if (physicalDevice.getSurfaceSupportKHR(index, *surface)) {
                presentFamily = index;
            }
        }

        // Create device
        const std::vector requiredExtensions{
            VK_KHR_SWAPCHAIN_EXTENSION_NAME,
        };

        float queuePriority = 0.0f;
        vk::DeviceQueueCreateInfo computeQueueCreateInfo;
        computeQueueCreateInfo.setQueueFamilyIndex(computeFamily);
        computeQueueCreateInfo.setQueueCount(1);
        computeQueueCreateInfo.setPQueuePriorities(&queuePriority);

        vk::DeviceQueueCreateInfo presentQueueCreateInfo;
        presentQueueCreateInfo.setQueueFamilyIndex(presentFamily);
        presentQueueCreateInfo.setQueueCount(1);
        presentQueueCreateInfo.setPQueuePriorities(&queuePriority);

        std::vector deviceQueueCreateInfos{
            computeQueueCreateInfo,
            presentQueueCreateInfo
        };

        vk::DeviceCreateInfo deviceCreateInfo;
        deviceCreateInfo.setQueueCreateInfos(computeQueueCreateInfo);
        deviceCreateInfo.setPEnabledLayerNames(layers);
        deviceCreateInfo.setPEnabledExtensionNames(requiredExtensions);
        vk::raii::Device device{ physicalDevice, deviceCreateInfo };

        // Get queue
        vk::raii::Queue computeQueue{ device, computeFamily, 0 };
        vk::raii::Queue presentQueue{ device, presentFamily, 0 };

        // Create command pool
        vk::CommandPoolCreateInfo commandPoolCreateInfo;
        commandPoolCreateInfo.setFlags(vk::CommandPoolCreateFlagBits::eResetCommandBuffer);
        commandPoolCreateInfo.setQueueFamilyIndex(computeFamily);
        vk::raii::CommandPool commandPool{ device, commandPoolCreateInfo };

        // Create command buffer
        vk::CommandBufferAllocateInfo commandBufferAllocateInfo{ *commandPool, vk::CommandBufferLevel::ePrimary, 1 };
        vk::raii::CommandBuffers commandBuffers{ device, commandBufferAllocateInfo };
        vk::raii::CommandBuffer commandBuffer = std::move(commandBuffers.front());

        // Create swapchain
        vk::SwapchainCreateInfoKHR swapchainCreateInfo;
        swapchainCreateInfo.setSurface(*surface);
        swapchainCreateInfo.setMinImageCount(3);
        swapchainCreateInfo.setImageFormat(vk::Format::eB8G8R8A8Unorm);
        swapchainCreateInfo.setImageColorSpace(vk::ColorSpaceKHR::eSrgbNonlinear);
        swapchainCreateInfo.setImageExtent({ width, height });
        swapchainCreateInfo.setImageArrayLayers(1);
        swapchainCreateInfo.setImageUsage(vk::ImageUsageFlagBits::eTransferDst);
        swapchainCreateInfo.setPreTransform(vk::SurfaceTransformFlagBitsKHR::eIdentity);
        swapchainCreateInfo.setPresentMode(vk::PresentModeKHR::eFifo);
        swapchainCreateInfo.setClipped(true);
        vk::raii::SwapchainKHR swapchain{ device, swapchainCreateInfo };
        std::vector swapChainImages = swapchain.getImages();

        // Create resources
        Image renderImage{ device, physicalDevice, commandBuffer, computeQueue, width, height };
        Image velocityImage{ device, physicalDevice, commandBuffer, computeQueue, width, height };
        Buffer uniformBuffer{ device, physicalDevice, commandBuffer, computeQueue, sizeof(UniformBufferObject) };

        UniformBufferObject ubo;
        ubo.mousePosition[0] = 0.0f;
        ubo.mousePosition[1] = 1.0f;
        uniformBuffer.copy(&ubo);

        // Create descriptor pool
        std::vector poolSizes{
            vk::DescriptorPoolSize{vk::DescriptorType::eStorageImage, 10},
            vk::DescriptorPoolSize{vk::DescriptorType::eUniformBuffer, 10}
        };
        vk::DescriptorPoolCreateInfo descPoolCreateInfo;
        descPoolCreateInfo.setPoolSizes(poolSizes);
        descPoolCreateInfo.setMaxSets(10);
        descPoolCreateInfo.setFlags(vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet);
        vk::raii::DescriptorPool descPool{ device, descPoolCreateInfo };

        // Create compute pipeline
        std::vector<vk::DescriptorSetLayoutBinding> renderKernelBindings{
            {0, vk::DescriptorType::eStorageImage, 1, vk::ShaderStageFlagBits::eCompute},
            {1, vk::DescriptorType::eStorageImage, 1, vk::ShaderStageFlagBits::eCompute}
        };
        std::vector<vk::DescriptorSetLayoutBinding> externalForceKernelBindings{
            {0, vk::DescriptorType::eStorageImage, 1, vk::ShaderStageFlagBits::eCompute},
            {1, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eCompute},
        };

        // Create kernels
        ComputeKernel renderKernel{ device, renderShader, renderKernelBindings, descPool };
        ComputeKernel externalForceKernel{ device, externalForceShader, externalForceKernelBindings, descPool };
        renderKernel.updateDescriptorSet(device, 0, 1, renderImage);
        renderKernel.updateDescriptorSet(device, 1, 1, velocityImage);
        externalForceKernel.updateDescriptorSet(device, 0, 1, velocityImage);
        externalForceKernel.updateDescriptorSet(device, 1, 1, uniformBuffer);

        // Main loop
        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();

            // Acquire next image
            vk::raii::Semaphore semaphore{ device, vk::SemaphoreCreateInfo {} };
            auto [result, imageIndex] = swapchain.acquireNextImage(UINT64_MAX, *semaphore);
            auto swapChainImage = swapChainImages[imageIndex];

            // Dispatch compute shader
            commandBuffer.reset();
            commandBuffer.begin(vk::CommandBufferBeginInfo{});
            externalForceKernel.run(commandBuffer, width, height);
            renderKernel.run(commandBuffer, width, height);

            // Copy render image
            setImageLayout(commandBuffer, *renderImage.image, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferSrcOptimal);
            setImageLayout(commandBuffer, swapChainImage, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal);

            vk::ImageCopy copyRegion;
            copyRegion.setSrcSubresource({ vk::ImageAspectFlagBits::eColor, 0, 0, 1 });
            copyRegion.setDstSubresource({ vk::ImageAspectFlagBits::eColor, 0, 0, 1 });
            copyRegion.setExtent({ width, height, 1 });
            commandBuffer.copyImage(*renderImage.image, vk::ImageLayout::eTransferSrcOptimal,
                                    swapChainImage, vk::ImageLayout::eTransferDstOptimal, copyRegion);

            setImageLayout(commandBuffer, *renderImage.image, vk::ImageLayout::eTransferSrcOptimal, vk::ImageLayout::eGeneral);
            setImageLayout(commandBuffer, swapChainImage, vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::ePresentSrcKHR);

            commandBuffer.end();

            // Submit command buffer
            vk::SubmitInfo submitInfo;
            submitInfo.setCommandBuffers(*commandBuffer);
            computeQueue.submit(submitInfo);
            computeQueue.waitIdle();

            // Present image
            vk::PresentInfoKHR presentInfo;
            presentInfo.setSwapchains(*swapchain);
            presentInfo.setImageIndices(imageIndex);
            presentQueue.presentKHR(presentInfo);
            presentQueue.waitIdle();
        }
        glfwDestroyWindow(window);
        glfwTerminate();
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
    }
}
