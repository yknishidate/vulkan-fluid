#include <string>
#include <iostream>
#include <vulkan/vulkan_raii.hpp>
#include <GLFW/glfw3.h>
#include <SPIRV/GlslangToSpv.h>
#include <StandAlone/ResourceLimits.h>

const std::string computeShader = R"(
#version 460
layout(local_size_x = 1, local_size_y = 1) in;
layout(binding = 0, set = 0, rgba8) uniform image2D renderImage;
void main()
{
    vec2 uv = gl_GlobalInvocationID.xy / (vec2(gl_NumWorkGroups) * vec2(gl_WorkGroupSize));
    imageStore(renderImage, ivec2(gl_GlobalInvocationID.xy), vec4(uv, 0, 1));
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

        // Create Instance
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

        // Create image
        vk::ImageCreateInfo imageCreateInfo;
        imageCreateInfo.setImageType(vk::ImageType::e2D);
        imageCreateInfo.setFormat(vk::Format::eB8G8R8A8Unorm);
        imageCreateInfo.setExtent({ width, height, 1 });
        imageCreateInfo.setMipLevels(1);
        imageCreateInfo.setArrayLayers(1);
        imageCreateInfo.setUsage(vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eTransferSrc);
        vk::raii::Image renderImage{ device, imageCreateInfo };

        // Get memory type index for image
        uint32_t memoryTypeIndex;
        vk::MemoryRequirements requirements = renderImage.getMemoryRequirements();
        vk::PhysicalDeviceMemoryProperties memoryProperties = physicalDevice.getMemoryProperties();
        for (uint32_t index = 0; index < memoryProperties.memoryTypeCount; ++index) {
            if (requirements.memoryTypeBits & (1 << index)) {
                memoryTypeIndex = index;
            }
        }

        // Allocate memory
        vk::MemoryAllocateInfo memoryAllocateInfo;
        memoryAllocateInfo.setAllocationSize(requirements.size);
        memoryAllocateInfo.setMemoryTypeIndex(memoryTypeIndex);
        vk::raii::DeviceMemory imageMemory{ device, memoryAllocateInfo };

        // Bind memory
        renderImage.bindMemory(*imageMemory, 0);

        // Create image view
        vk::ImageViewCreateInfo imageViewCreateInfo;
        imageViewCreateInfo.setImage(*renderImage);
        imageViewCreateInfo.setViewType(vk::ImageViewType::e2D);
        imageViewCreateInfo.setFormat(vk::Format::eB8G8R8A8Unorm);
        imageViewCreateInfo.setSubresourceRange({ vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 });
        vk::raii::ImageView imageView{ device, imageViewCreateInfo };

        // Set image layout
        commandBuffer.begin(vk::CommandBufferBeginInfo{});
        setImageLayout(commandBuffer, *renderImage, vk::ImageLayout::eUndefined, vk::ImageLayout::eGeneral);
        commandBuffer.end();

        vk::SubmitInfo submitInfo;
        submitInfo.setCommandBuffers(*commandBuffer);
        computeQueue.submit(submitInfo);
        computeQueue.waitIdle();

        // Compile shader
        std::vector spvShader = compileToSPV(computeShader);

        // Create shader module
        vk::ShaderModuleCreateInfo shaderModuleCreateInfo;
        shaderModuleCreateInfo.setCode(spvShader);
        vk::raii::ShaderModule shaderModule{ device, shaderModuleCreateInfo };

        // Create compute pipeline
        vk::DescriptorSetLayoutBinding binding;
        binding.setBinding(0);
        binding.setDescriptorType(vk::DescriptorType::eStorageImage);
        binding.setDescriptorCount(1);
        binding.setStageFlags(vk::ShaderStageFlagBits::eCompute);

        vk::DescriptorSetLayoutCreateInfo descSetLayoutCreateInfo;
        descSetLayoutCreateInfo.setBindings(binding);
        vk::raii::DescriptorSetLayout descSetLayout{ device, descSetLayoutCreateInfo };

        vk::PipelineLayoutCreateInfo pipelineLayoutCreateInfo;
        pipelineLayoutCreateInfo.setSetLayouts(*descSetLayout);
        vk::raii::PipelineLayout pipelineLayout{ device, pipelineLayoutCreateInfo };

        vk::PipelineShaderStageCreateInfo shaderStageCreateInfo;
        shaderStageCreateInfo.setStage(vk::ShaderStageFlagBits::eCompute);
        shaderStageCreateInfo.setModule(*shaderModule);
        shaderStageCreateInfo.setPName("main");

        vk::ComputePipelineCreateInfo pipelineCreateInfo;
        pipelineCreateInfo.setStage(shaderStageCreateInfo);
        pipelineCreateInfo.setLayout(*pipelineLayout);
        vk::raii::Pipeline pipeline{ device, nullptr, pipelineCreateInfo };

        // Create descriptor set
        std::vector poolSizes{
            vk::DescriptorPoolSize{vk::DescriptorType::eStorageImage, 1}
        };
        vk::DescriptorPoolCreateInfo descPoolCreateInfo;
        descPoolCreateInfo.setPoolSizes(poolSizes);
        descPoolCreateInfo.setMaxSets(1);
        descPoolCreateInfo.setFlags(vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet);
        vk::raii::DescriptorPool descPool{ device, descPoolCreateInfo };

        vk::DescriptorSetAllocateInfo descSetAllocationInfo;
        descSetAllocationInfo.setDescriptorPool(*descPool);
        descSetAllocationInfo.setSetLayouts(*descSetLayout);

        vk::raii::DescriptorSets descSets{ device, descSetAllocationInfo };
        vk::raii::DescriptorSet descSet = std::move(descSets.front());

        // Update descriptor set
        vk::DescriptorImageInfo descImageInfo;
        descImageInfo.setImageView(*imageView);
        descImageInfo.setImageLayout(vk::ImageLayout::eGeneral);

        vk::WriteDescriptorSet imageWrite;
        imageWrite.setDstSet(*descSet);
        imageWrite.setDescriptorType(vk::DescriptorType::eStorageImage);
        imageWrite.setDescriptorCount(1);
        imageWrite.setDstBinding(0);
        imageWrite.setImageInfo(descImageInfo);

        device.updateDescriptorSets(imageWrite, nullptr);

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
            commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, *pipeline);
            commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, *pipelineLayout, 0, *descSet, nullptr);
            commandBuffer.dispatch(width, height, 1);

            // Copy render image
            setImageLayout(commandBuffer, *renderImage, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferSrcOptimal);
            setImageLayout(commandBuffer, swapChainImage, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal);

            vk::ImageCopy copyRegion;
            copyRegion.setSrcSubresource({ vk::ImageAspectFlagBits::eColor, 0, 0, 1 });
            copyRegion.setDstSubresource({ vk::ImageAspectFlagBits::eColor, 0, 0, 1 });
            copyRegion.setExtent({ width, height, 1 });
            commandBuffer.copyImage(*renderImage, vk::ImageLayout::eTransferSrcOptimal,
                                    swapChainImage, vk::ImageLayout::eTransferDstOptimal, copyRegion);

            setImageLayout(commandBuffer, *renderImage, vk::ImageLayout::eTransferSrcOptimal, vk::ImageLayout::eGeneral);
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
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
    }
}
