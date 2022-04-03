#pragma once
#include <fstream>
#include <sstream>
#include <iostream>
#include <vulkan/vulkan.hpp>
#include <SPIRV/GlslangToSpv.h>
#include <StandAlone/ResourceLimits.h>

std::string readFile(const std::string& path)
{
    std::ifstream input_file(path);
    if (!input_file.is_open()) {
        throw std::runtime_error("Failed to open file: " + path);
    }
    return std::string((std::istreambuf_iterator<char>(input_file)), std::istreambuf_iterator<char>());
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
        throw std::runtime_error(glslShader + "\n" + shader.getInfoLog());
    }

    glslang::TProgram program;
    program.addShader(&shader);

    if (!program.link(messages)) {
        throw std::runtime_error(glslShader + "\n" + shader.getInfoLog());
    }

    std::vector<unsigned int> spvShader;
    glslang::GlslangToSpv(*program.getIntermediate(stage), spvShader);
    glslang::FinalizeProcess();

    return spvShader;
}

struct ComputeKernel
{
    ComputeKernel(vk::Device device,
                  const std::string& path,
                  const std::vector<vk::DescriptorSetLayoutBinding>& bindings,
                  vk::DescriptorPool descPool,
                  size_t pushSize = 0)
        : device{ device }
        , pushSize{ pushSize }
    {
        createShaderModule(path);
        createDescSetLayout(bindings);
        createPipelineLayout();
        createComputePipeline();
        allocateDescriptorSet(descPool);
    }

    void createShaderModule(const std::string& path)
    {
        std::string code = readFile(path);
        spirvCode = compileToSPV(code);
        vk::ShaderModuleCreateInfo createInfo;
        createInfo.setCode(spirvCode);
        shaderModule = device.createShaderModuleUnique(createInfo);
    }

    void createDescSetLayout(const std::vector<vk::DescriptorSetLayoutBinding>& bindings)
    {
        vk::DescriptorSetLayoutCreateInfo createInfo;
        createInfo.setBindings(bindings);
        descSetLayout = device.createDescriptorSetLayoutUnique(createInfo);
    }

    void createPipelineLayout()
    {
        vk::PipelineLayoutCreateInfo createInfo;
        createInfo.setSetLayouts(*descSetLayout);
        if (pushSize) {
            vk::PushConstantRange pushRange;
            pushRange.setOffset(0);
            pushRange.setSize(pushSize);
            pushRange.setStageFlags(vk::ShaderStageFlagBits::eCompute);
            createInfo.setPushConstantRanges(pushRange);
        }
        pipelineLayout = device.createPipelineLayoutUnique(createInfo);
    }

    void createComputePipeline()
    {
        vk::PipelineShaderStageCreateInfo stage;
        stage.setStage(vk::ShaderStageFlagBits::eCompute);
        stage.setModule(*shaderModule);
        stage.setPName("main");

        vk::ComputePipelineCreateInfo createInfo;
        createInfo.setStage(stage);
        createInfo.setLayout(*pipelineLayout);
        auto res = device.createComputePipelinesUnique({}, createInfo);
        if (res.result != vk::Result::eSuccess) {
            throw std::runtime_error("failed to create ray tracing pipeline.");
        }
        pipeline = std::move(res.value.front());
    }

    void allocateDescriptorSet(vk::DescriptorPool descPool)
    {
        vk::DescriptorSetAllocateInfo allocateInfo;
        allocateInfo.setDescriptorPool(descPool);
        allocateInfo.setSetLayouts(*descSetLayout);
        std::vector descSets = device.allocateDescriptorSetsUnique(allocateInfo);
        descSet = std::move(descSets.front());
    }

    void updateDescriptorSet(uint32_t binding, uint32_t count, const Image& image,
                             vk::DescriptorType descType = vk::DescriptorType::eStorageImage)
    {
        vk::DescriptorImageInfo descImageInfo;
        descImageInfo.setImageView(*image.view);
        descImageInfo.setImageLayout(vk::ImageLayout::eGeneral);
        descImageInfo.setSampler(*image.sampler);

        vk::WriteDescriptorSet imageWrite;
        imageWrite.setDstSet(*descSet);
        imageWrite.setDescriptorType(descType);
        imageWrite.setDescriptorCount(count);
        imageWrite.setDstBinding(binding);
        imageWrite.setImageInfo(descImageInfo);

        device.updateDescriptorSets(imageWrite, nullptr);
    }

    void run(vk::CommandBuffer commandBuffer, uint32_t groupCountX, uint32_t groupCountY, void* pushData = nullptr)
    {
        commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, *pipeline);
        commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, *pipelineLayout, 0, *descSet, nullptr);
        if (pushData) {
            commandBuffer.pushConstants(*pipelineLayout, vk::ShaderStageFlagBits::eCompute, 0, pushSize, pushData);
        }
        commandBuffer.dispatch(groupCountX, groupCountY, 1);
    }

    vk::Device device;
    std::vector<unsigned int> spirvCode;
    vk::UniqueShaderModule shaderModule;
    vk::UniqueDescriptorSetLayout descSetLayout;
    vk::UniquePipelineLayout pipelineLayout;
    vk::UniquePipeline pipeline;
    vk::UniqueDescriptorSet descSet;
    size_t pushSize;
};
