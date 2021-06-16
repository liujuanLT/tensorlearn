
#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <memory>
#include <sstream>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <vector>

#include "NvInfer.h"
#include "NvInferPlugin.h"

#include "buffers.h"
#include "common.h"
#include "logger.h"
#include "sampleDevice.h"
#include "sampleEngines.h"
#include "sampleInference.h"
#include "sampleOptions.h"
#include "sampleReporting.h"

using namespace nvinfer1;
using namespace sample;

using time_point = std::chrono::time_point<std::chrono::high_resolution_clock>;
using duration = std::chrono::duration<float>;

ICudaEngine* loadEngine(const std::string& engine, int DLACore)
{
    std::ifstream engineFile(engine, std::ios::binary);
    if (!engineFile)
    {
        std::cout << "Error opening engine file: " << engine << std::endl;
        return nullptr;
    }

    engineFile.seekg(0, engineFile.end);
    long int fsize = engineFile.tellg();
    engineFile.seekg(0, engineFile.beg);

    std::vector<char> engineData(fsize);
    engineFile.read(engineData.data(), fsize);
    if (!engineFile)
    {
        std::cout << "Error loading engine file: " << engine << std::endl;
        return nullptr;
    }

    TrtUniquePtr<IRuntime> runtime{createInferRuntime(sample::gLogger.getTRTLogger())};
    if (DLACore != -1)
    {
        runtime->setDLACore(DLACore);
    }

    return runtime->deserializeCudaEngine(engineData.data(), fsize, nullptr);
}

int main(int argc, char** argv)
{
    cudaSetDevice(0);

    initLibNvInferPlugins(&sample::gLogger.getTRTLogger(), "");

    // DeserializeEngine("/home/jliu/data/models/ssd300_coco_640x427inshape_reshape_fp32.trt");
    nvinfer1::ICudaEngine *mEngine = loadEngine(std::string("/home/jliu/data/models/ssd300_coco_640x427inshape_reshape_fp32.trt"), -1);

    std::cout<<"max batch size of deserialized engine: " << mEngine->getMaxBatchSize() << std::endl;

    return 0;  
}