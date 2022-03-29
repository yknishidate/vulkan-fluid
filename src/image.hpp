#pragma once
#include <vulkan/vulkan_raii.hpp>

void setImageLayout(const vk::raii::CommandBuffer& commandBuffer,
                    vk::Image image, vk::ImageLayout oldLayout, vk::ImageLayout newLayout)
{
    vk::PipelineStageFlags srcStageMask = vk::PipelineStageFlagBits::eAllCommands;
    vk::PipelineStageFlags dstStageMask = vk::PipelineStageFlagBits::eAllCommands;

    vk::ImageMemoryBarrier barrier;
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

struct Image
{
    Image(const vk::raii::Device& device,
          const vk::raii::PhysicalDevice& physicalDevice,
          const vk::raii::CommandBuffer& commandBuffer,
          const vk::raii::Queue& queue,
          int width, int height,
          vk::Format format = vk::Format::eR32G32B32A32Sfloat)
        : image{ device, makeImageCreateInfo(width, height, format) }
        , memory{ device, makeMemoryAllocationInfo(device, physicalDevice) }
        , view{ device, makeImageViewCreateInfo(format) }
        , sampler{ device, makeSamplerCreateInfo() }
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

    vk::ImageCreateInfo makeImageCreateInfo(uint32_t width, uint32_t height, vk::Format format)
    {
        vk::ImageCreateInfo imageCreateInfo;
        imageCreateInfo.setImageType(vk::ImageType::e2D);
        imageCreateInfo.setFormat(format);
        imageCreateInfo.setExtent({ width, height, 1 });
        imageCreateInfo.setMipLevels(1);
        imageCreateInfo.setArrayLayers(1);
        imageCreateInfo.setUsage(vk::ImageUsageFlagBits::eStorage |
                                 vk::ImageUsageFlagBits::eTransferSrc |
                                 vk::ImageUsageFlagBits::eTransferDst |
                                 vk::ImageUsageFlagBits::eSampled);
        return imageCreateInfo;
    }

    vk::MemoryAllocateInfo makeMemoryAllocationInfo(const vk::raii::Device& device,
                                                    const vk::raii::PhysicalDevice& physicalDevice)
    {
        vk::MemoryRequirements requirements = image.getMemoryRequirements();
        uint32_t memoryTypeIndex;
        vk::PhysicalDeviceMemoryProperties memoryProperties = physicalDevice.getMemoryProperties();
        for (uint32_t index = 0; index < memoryProperties.memoryTypeCount; ++index) {
            if (requirements.memoryTypeBits & (1 << index)) {
                memoryTypeIndex = index;
            }
        }

        vk::MemoryAllocateInfo memoryAllocateInfo;
        memoryAllocateInfo.setAllocationSize(requirements.size);
        memoryAllocateInfo.setMemoryTypeIndex(memoryTypeIndex);
        return memoryAllocateInfo;
    }

    vk::ImageViewCreateInfo makeImageViewCreateInfo(vk::Format format)
    {
        image.bindMemory(*memory, 0);

        vk::ImageViewCreateInfo imageViewCreateInfo;
        imageViewCreateInfo.setImage(*image);
        imageViewCreateInfo.setViewType(vk::ImageViewType::e2D);
        imageViewCreateInfo.setFormat(format);
        imageViewCreateInfo.setSubresourceRange({ vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 });
        return imageViewCreateInfo;
    }

    vk::SamplerCreateInfo makeSamplerCreateInfo()
    {
        vk::SamplerCreateInfo createInfo;
        createInfo.setMagFilter(vk::Filter::eLinear);
        createInfo.setMinFilter(vk::Filter::eLinear);
        createInfo.setAnisotropyEnable(VK_FALSE);
        createInfo.setMaxLod(0.0f);
        createInfo.setMinLod(0.0f);
        createInfo.setMipmapMode(vk::SamplerMipmapMode::eLinear);
        return createInfo;
    }

    vk::raii::Image image;
    vk::raii::DeviceMemory memory;
    vk::raii::ImageView view;
    vk::raii::Sampler sampler;
};
