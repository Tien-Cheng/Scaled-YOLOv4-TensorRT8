#include <cmath>
#include <stdio.h>
#include <cassert>
#include <iostream>
#include "mish.h"
#include "macros.h"

namespace nvinfer1
{
    MishPlugin::MishPlugin() TRT_NOEXCEPT
    {
    }

    MishPlugin::~MishPlugin() TRT_NOEXCEPT
    {
    }

    // create the plugin at runtime from a byte stream
    MishPlugin::MishPlugin(const void* data, size_t length) TRT_NOEXCEPT
    {
        assert(length == sizeof(input_size_));
        input_size_ = *reinterpret_cast<const int*>(data);
    }

    void MishPlugin::serialize(void* buffer) const TRT_NOEXCEPT
    {
        *reinterpret_cast<int*>(buffer) = input_size_;
    }

    size_t MishPlugin::getSerializationSize() const TRT_NOEXCEPT
    {  
        return sizeof(input_size_);
    }

    int MishPlugin::initialize() TRT_NOEXCEPT
    { 
        return 0;
    }

    Dims MishPlugin::getOutputDimensions(int index, const Dims* inputs, int nbInputDims) TRT_NOEXCEPT
    {
        assert(nbInputDims == 1);
        assert(index == 0);
        input_size_ = inputs[0].d[0] * inputs[0].d[1] * inputs[0].d[2];
        // Output dimensions
        return Dims3(inputs[0].d[0], inputs[0].d[1], inputs[0].d[2]);
    }

    // Set plugin namespace
    void MishPlugin::setPluginNamespace(const char* pluginNamespace) TRT_NOEXCEPT
    {
        mPluginNamespace = pluginNamespace;
    }

    const char* MishPlugin::getPluginNamespace() const TRT_NOEXCEPT
    {
        return mPluginNamespace;
    }

    // Return the DataType of the plugin output at the requested index
    DataType MishPlugin::getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const TRT_NOEXCEPT
    {
        return DataType::kFLOAT;
    }

    // Return true if output tensor is broadcast across a batch.
    bool MishPlugin::isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const TRT_NOEXCEPT
    {
        return false;
    }

    // Return true if plugin can use input that is broadcast across batch without replication.
    bool MishPlugin::canBroadcastInputAcrossBatch(int inputIndex) const TRT_NOEXCEPT
    {
        return false;
    }

    void MishPlugin::configurePlugin(const PluginTensorDesc* in, int nbInput, const PluginTensorDesc* out, int nbOutput) TRT_NOEXCEPT
    {
    }

    // Attach the plugin object to an execution context and grant the plugin the access to some context resource.
    void MishPlugin::attachToContext(cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator) TRT_NOEXCEPT
    {
    }

    // Detach the plugin object from its execution context.
    void MishPlugin::detachFromContext() TRT_NOEXCEPT {}

    const char* MishPlugin::getPluginType() const TRT_NOEXCEPT
    {
        return "Mish_TRT";
    }

    const char* MishPlugin::getPluginVersion() const TRT_NOEXCEPT
    {
        return "1";
    }

    void MishPlugin::destroy() TRT_NOEXCEPT
    {
        delete this;
    }

    // Clone the plugin
    IPluginV2IOExt* MishPlugin::clone() const TRT_NOEXCEPT
    {
        MishPlugin *p = new MishPlugin();
        p->input_size_ = input_size_;
        p->setPluginNamespace(mPluginNamespace);
        return p;
    }

    __device__ float tanh_activate_kernel(float x){return (2/(1 + expf(-2*x)) - 1);}

    __device__ float softplus_kernel(float x, float threshold = 20) {
        if (x > threshold) return x;                // too large
        else if (x < -threshold) return expf(x);    // too small
        return logf(expf(x) + 1);
    }

    __global__ void mish_kernel(const float *input, float *output, int num_elem) {

        int idx = threadIdx.x + blockDim.x * blockIdx.x;
        if (idx >= num_elem) return;

        //float t = exp(input[idx]);
        //if (input[idx] > 20.0) {
        //    t *= t;
        //    output[idx] = (t - 1.0) / (t + 1.0);
        //} else {
        //    float tt = t * t;
        //    output[idx] = (tt + 2.0 * t) / (tt + 2.0 * t + 2.0);
        //}
        //output[idx] *= input[idx];
        output[idx] = input[idx] * tanh_activate_kernel(softplus_kernel(input[idx]));
    }

    void MishPlugin::forwardGpu(const float *const * inputs, float* output, cudaStream_t stream, int batchSize) TRT_NOEXCEPT {
        int block_size = thread_count_;
        int grid_size = (input_size_ * batchSize + block_size - 1) / block_size;
        mish_kernel<<<grid_size, block_size>>>(inputs[0], output, input_size_ * batchSize);
    }

    int MishPlugin::enqueue(int batchSize, const void*const * inputs, void*TRT_CONST_ENQUEUE* outputs, void* workspace, cudaStream_t stream) TRT_NOEXCEPT
    {
        //assert(batchSize == 1);
        //GPU
        //CUDA_CHECK(cudaStreamSynchronize(stream));
        forwardGpu((const float *const *)inputs, (float*)outputs[0], stream, batchSize);
        return 0;
    }

    PluginFieldCollection MishPluginCreator::mFC{};
    std::vector<PluginField> MishPluginCreator::mPluginAttributes;

    MishPluginCreator::MishPluginCreator() TRT_NOEXCEPT
    {
        mPluginAttributes.clear();

        mFC.nbFields = mPluginAttributes.size();
        mFC.fields = mPluginAttributes.data();
    }

    const char* MishPluginCreator::getPluginName() const TRT_NOEXCEPT
    {
            return "Mish_TRT";
    }

    const char* MishPluginCreator::getPluginVersion() const TRT_NOEXCEPT
    {
            return "1";
    }

    const PluginFieldCollection* MishPluginCreator::getFieldNames() TRT_NOEXCEPT
    {
            return &mFC;
    }

    IPluginV2IOExt* MishPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) TRT_NOEXCEPT
    {
        MishPlugin* obj = new MishPlugin();
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }

    IPluginV2IOExt* MishPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength) TRT_NOEXCEPT
    {
        // This object will be deleted when the network is destroyed, which will
        // call MishPlugin::destroy()
        MishPlugin* obj = new MishPlugin(serialData, serialLength);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }

}

