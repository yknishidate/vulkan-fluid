#pragma once
#include <vulkan/vulkan_raii.hpp>

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
