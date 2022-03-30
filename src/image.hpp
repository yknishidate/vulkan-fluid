#pragma once
#include "common.hpp"

void setImageLayout(vk::CommandBuffer commandBuffer, vk::Image image,
                    vk::ImageLayout oldLayout, vk::ImageLayout newLayout)
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
    Image(vk::Device device,
          vk::PhysicalDevice physicalDevice,
          vk::CommandBuffer commandBuffer,
          vk::Queue queue,
          int width, int height,
          vk::Format format = vk::Format::eR32G32B32A32Sfloat)
        : device{ device }
    {
        createImage(width, height, format);
        allocateMemory(physicalDevice);
        bindMemory();
        createImageView(format);
        createSampler();
        transImageLayout(commandBuffer, queue);
    }

    void createImage(uint32_t width, uint32_t height, vk::Format format)
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
        image = device.createImageUnique(imageCreateInfo);
    }

    void allocateMemory(vk::PhysicalDevice physicalDevice)
    {
        vk::MemoryRequirements requirements = device.getImageMemoryRequirements(*image);
        uint32_t memoryTypeIndex = findMemoryTypeIndex(physicalDevice, requirements,
                                                       vk::MemoryPropertyFlagBits::eDeviceLocal);
        vk::MemoryAllocateInfo memoryAllocateInfo;
        memoryAllocateInfo.setAllocationSize(requirements.size);
        memoryAllocateInfo.setMemoryTypeIndex(memoryTypeIndex);
        memory = device.allocateMemoryUnique(memoryAllocateInfo);
    }

    void bindMemory()
    {
        device.bindImageMemory(*image, *memory, 0);
    }

    void createImageView(vk::Format format)
    {
        vk::ImageViewCreateInfo imageViewCreateInfo;
        imageViewCreateInfo.setImage(*image);
        imageViewCreateInfo.setViewType(vk::ImageViewType::e2D);
        imageViewCreateInfo.setFormat(format);
        imageViewCreateInfo.setSubresourceRange({ vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 });
        view = device.createImageViewUnique(imageViewCreateInfo);
    }

    void createSampler()
    {
        vk::SamplerCreateInfo createInfo;
        createInfo.setMagFilter(vk::Filter::eLinear);
        createInfo.setMinFilter(vk::Filter::eLinear);
        createInfo.setAnisotropyEnable(VK_FALSE);
        createInfo.setMaxLod(0.0f);
        createInfo.setMinLod(0.0f);
        createInfo.setMipmapMode(vk::SamplerMipmapMode::eLinear);
        createInfo.setAddressModeU(vk::SamplerAddressMode::eClampToBorder);
        createInfo.setAddressModeV(vk::SamplerAddressMode::eClampToBorder);
        createInfo.setAddressModeW(vk::SamplerAddressMode::eClampToBorder);
        sampler = device.createSamplerUnique(createInfo);
    }

    void transImageLayout(vk::CommandBuffer commandBuffer, vk::Queue queue)
    {
        commandBuffer.begin(vk::CommandBufferBeginInfo{});
        setImageLayout(commandBuffer, *image, vk::ImageLayout::eUndefined, vk::ImageLayout::eGeneral);
        commandBuffer.end();

        vk::SubmitInfo submitInfo;
        submitInfo.setCommandBuffers(commandBuffer);
        queue.submit(submitInfo);
        queue.waitIdle();
    }

    vk::Device device;
    vk::UniqueImage image;
    vk::UniqueDeviceMemory memory;
    vk::UniqueImageView view;
    vk::UniqueSampler sampler;
};
