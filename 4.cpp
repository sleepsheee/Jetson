#include <iostream>
#include <fstream>
#include <NvInfer.h>
#include <memory>
#include <NvOnnxParser.h>
#include <vector>
#include <cuda_runtime_api.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/core.hpp>
#include <opencv2/cudaarithm.hpp>
#include <algorithm>
#include <numeric>
#include <ctime>
#include <sys/stat.h>
#include <fstream>
#include <half.hpp>
#include <cuda_fp16.h>

// utilities ----------------------------------------------------------------------------------------------------------
// class to log errors, warnings, and other information during the build and inference phases
class Logger : public nvinfer1::ILogger
{
public:
    void log(Severity severity, const char* msg) noexcept override {
        // remove this 'if' if you need more logged info
        if ((severity == Severity::kERROR) || (severity == Severity::kINTERNAL_ERROR)) {
            std::cout << msg << "\n";
        }
    }
} gLogger;


class MyLogger : public nvinfer1::ILogger {
 public:
  explicit MyLogger(nvinfer1::ILogger::Severity severity =
                        nvinfer1::ILogger::Severity::kWARNING)
      : severity_(severity) {}

  void log(nvinfer1::ILogger::Severity severity,
           const char *msg) noexcept override {
    if (severity <= severity_) {
      std::cerr << msg << std::endl;
    }
  }
  nvinfer1::ILogger::Severity severity_;
};


// destroy TensorRT objects if something goes wrong
struct TRTDestroy
{
    template <class T>
    void operator()(T* obj) const
    {
        if (obj)
        {
            obj->destroy();
        }
    }
};

template <class T>
using TRTUniquePtr = std::unique_ptr<T, TRTDestroy>;

// calculate size of tensor
size_t getSizeByDim(const nvinfer1::Dims& dims)
{
    size_t size = 1;
    for (size_t i = 0; i < dims.nbDims; ++i)
    {
        size *= dims.d[i];
    }
    return size;
}

// get classes names
std::vector<std::string> getClassNames(const std::string& imagenet_classes)
{
    std::ifstream classes_file(imagenet_classes);
    std::vector<std::string> classes;
    if (!classes_file.good())
    {
        std::cerr << "ERROR: can't read file with classes names.\n";
        return classes;
    }
    std::string class_name;
    while (std::getline(classes_file, class_name))
    {
        classes.push_back(class_name);
    }
    return classes;
}



void convert_fp16(cv::Mat& img, half_float::half * data) {
    for (int c = 0; c < 3; ++c) {
        for (int i = 0; i < img.rows; ++i) { 
            cv::Vec3f *p1 = img.ptr<cv::Vec3f>(i);
            for (int j = 0; j < img.cols; ++j)  {
                data[c * img.cols * img.rows + i * img.cols + j] = half_float::half(p1[j][c]);  
            }
        }   
    }

  
}



// preprocessing stage ------------------------------------------------------------------------------------------------
void preprocessImage(const std::string& image_path, half_float::half* gpu_input, const nvinfer1::Dims& dims)
{
    // read input image
    std::cout<<"start preprocess"<<std::endl;
    cv::Mat frame = cv::imread(image_path);


    //convert to fp16
    half_float::half* data = new half_float::half[640 * 480 * 3];
    convert_fp16(frame, data);
  

    if (frame.empty())
    {
        std::cerr << "Input image " << image_path << " load failed\n";
        return;
    }
  //  cv::cuda::GpuMat gpu_frame;
    // upload image to GPU

    auto input_width = dims.d[2];
    auto input_height = dims.d[1];
    auto channels = dims.d[0];
    /*half_float::half* hostDataBuffer = static_cast<half_float::half*>(buffers->getHostBuffer('input');

    auto input_size = cv::Size(input_width, input_height);
    
	for (int i = 0; i < input_height * input_width * channels; i++) {
	    hostDataBuffer[i] = half_float::half(data[i]);
	}
    buffers->copyInputToDevice();*/

    cudaMemcpy(gpu_input, data, 640 * 480 * 3* sizeof(half_float::half), cudaMemcpyHostToDevice);

    double begin_time = clock();
  
    // to tensor
    /*std::vector<cv::cuda::GpuMat> chw;
    for (size_t i = 0; i < channels; ++i)
    {
        chw.emplace_back(cv::cuda::GpuMat(input_size, CV_32FC1, gpu_input + i * input_width * input_height));
    }
    cv::cuda::split(gpu_frame, chw);*/




    double end_time = clock();
    double process_time = (double)(end_time-begin_time)/CLOCKS_PER_SEC;
    std::cout<<"preprocess time:"<<process_time<<std::endl;
    std::cout<<"finish preprocess"<<std::endl;
    
}


// post-processing stage ----------------------------------------------------------------------------------------------
void postprocessResults(std::vector<void *> buffers, std::vector<nvinfer1::Dims> &dims, int batch_size)
{
    // get class names
    //auto classes = getClassNames("imagenet_classes.txt");

    // copy results from GPU to CPU
  
    std::vector<std::vector<half_float::half> > cpu_outputs;
    for (size_t i = 0; i < dims.size(); i++) 
    {
        cpu_outputs.push_back(std::vector<half_float::half>(getSizeByDim(dims[i]) * batch_size));
        cudaMemcpy(cpu_outputs[i].data(), ( half_float::half *)buffers[1 + i], cpu_outputs[i].size() * sizeof(half_float::half), cudaMemcpyDeviceToHost);
     
    }
    std::cout<<"cpu output size:"<<cpu_outputs.size()<<std::endl;
    //std::cout<<"output1 type "<<typeid(cpu_outputs[0]).name()<<std::endl;
    //std::cout<<"output1 "<<cpu_outputs[0].data()<<std::endl;
    //std::cout<<"output1: ";
    int count1{0};

     // using begin() to print vector
    /*for (auto it = cpu_outputs[0].begin();
         it != cpu_outputs[0].end(); ++it)
        std::cout << ' ' << *it<<std::endl;*/
    
    
    std::ofstream fout;
    fout.open("output1_fp16.txt");

    for (int l = 0; l < cpu_outputs[0].size(); l++)
         fout  << cpu_outputs[0].at(l) << std::endl;

    fout.close();

    
    std::cout<<"output2: ";

    // using begin() to print vector
    for (auto it = cpu_outputs[1].begin();
         it != cpu_outputs[1].end(); ++it)
        std::cout << ' ' << *it<<std::endl;
    std::cout<<"output3: ";

    // using begin() to print vector
    for (auto it = cpu_outputs[2].begin();
         it != cpu_outputs[2].end(); ++it)
        std::cout << ' ' << *it<<std::endl;

    std::cout<<"output1 size "<<cpu_outputs[0].size()<<std::endl;
    std::cout<<"output2 size "<<cpu_outputs[1].size()<<std::endl;
    std::cout<<"output3 size "<<cpu_outputs[2].size()<<std::endl;

    std::cout<<"output4 size "<<cpu_outputs[3].size()<<std::endl;
    std::cout<<"output5 size "<<cpu_outputs[4].size()<<std::endl;
    std::cout<<"output6 size "<<cpu_outputs[5].size()<<std::endl;

    
}

// initialize TensorRT engine and parse ONNX model --------------------------------------------------------------------
void parseOnnxModel(const std::string& model_path, TRTUniquePtr<nvinfer1::ICudaEngine>& engine,
                    TRTUniquePtr<nvinfer1::IExecutionContext>& context)
{


    //create IBuilder instance
    TRTUniquePtr<nvinfer1::IBuilder> builder{nvinfer1::createInferBuilder(gLogger)};
    //create network structure
    TRTUniquePtr<nvinfer1::INetworkDefinition> network{builder->createNetworkV2(1U << (uint32_t)nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH)};
    //
    TRTUniquePtr<nvonnxparser::IParser> parser{nvonnxparser::createParser(*network, gLogger)};
    TRTUniquePtr<nvinfer1::IBuilderConfig> config{builder->createBuilderConfig()};
    // parse ONNX
    if (!parser->parseFromFile(model_path.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kINFO)))
    {
        std::cerr << "ERROR: could not parse the model.\n";
        return;
    }
    // allow TensorRT to use up to 1GB of GPU memory for tactic selection.
    config->setMaxWorkspaceSize(1ULL << 30);
    // use FP16 mode if possible
    if (builder->platformHasFastFp16())
    {
        std::cout<<"support fp16"<<std::endl;
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
    }
    // we have only one image in batch
    builder->setMaxBatchSize(1);
    // generate TensorRT engine optimized for the target platform



    //set input0/output1,2,3 to fp16
    network->getInput(0)->setType(nvinfer1::DataType::kHALF);
    network->getOutput(0)->setType(nvinfer1::DataType::kHALF);
    network->getOutput(1)->setType(nvinfer1::DataType::kHALF);
    network->getOutput(2)->setType(nvinfer1::DataType::kHALF);
    network->getOutput(3)->setType(nvinfer1::DataType::kHALF);
    network->getOutput(4)->setType(nvinfer1::DataType::kHALF);
    network->getOutput(5)->setType(nvinfer1::DataType::kHALF);



    engine.reset(builder->buildEngineWithConfig(*network, *config));
    context.reset(engine->createExecutionContext());


    /*--------------------------------------------------*/
    nvinfer1::IHostMemory *serialized_model = builder->buildSerializedNetwork(*network, *config);
    // 将模型序列化到engine文件中
    std::stringstream engine_file_stream;
    engine_file_stream.seekg(0, engine_file_stream.beg);
    engine_file_stream.write(static_cast<const char *>(serialized_model->data()), serialized_model->size());
    const std::string engine_file_path = "3_fp16.engine";
    std::ofstream out_file(engine_file_path);
    assert(out_file.is_open());
    out_file << engine_file_stream.rdbuf();
    out_file.close();   
}



void readEngine(TRTUniquePtr<nvinfer1::ICudaEngine> &engine,
                    TRTUniquePtr<nvinfer1::IExecutionContext> &context)
{
       
    auto cache_path = "3_fp16.engine";
    char *engineStream = NULL;
    size_t engineSize = 0;
    // determine the file size of the engine
    struct stat filestat;
    stat(cache_path, &filestat);
    engineSize = filestat.st_size;
    // allocate memory to hold the engine
    engineStream = (char *)malloc(engineSize);
    // open the engine cache file from disk
    FILE *cacheFile = NULL;
    cacheFile = fopen(cache_path, "rb");
    // read the serialized engine into memory
    const size_t bytesRead = fread(engineStream, 1, engineSize, cacheFile);
    if (bytesRead != engineSize) // Problem while deserializing.
    {
      std::cerr << "Error reading serialized engine into memory." << std::endl;
     
    }
    // close the plan cache
    fclose(cacheFile);
    // Recreate the inference runtime
    TRTUniquePtr<nvinfer1::IRuntime> infer{nvinfer1::createInferRuntime(gLogger)};
    engine.reset(infer->deserializeCudaEngine(engineStream, engineSize, NULL));
    context.reset(engine->createExecutionContext());;
}



// main pipeline ------------------------------------------------------------------------------------------------------
int main(int argc, char* argv[])
{
    
    if (argc < 3)
    {
        std::cerr << "usage: " << argv[0] << " model.onnx image.jpg\n";
        return -1;
    }
    std::string model_path(argv[1]);
    std::string image_path(argv[2]);
    int batch_size = 1;

    // initialize TensorRT engine and parse ONNX model
    TRTUniquePtr<nvinfer1::ICudaEngine> engine{nullptr};
    TRTUniquePtr<nvinfer1::IExecutionContext> context{nullptr};

    
  
    /*std::cout<<"start parse"<<std::endl;
    parseOnnxModel(model_path, engine, context);
    std::cout<<"finish parse"<<std::endl;*/
    std::cout<<"start reading engine"<<std::endl;
    readEngine(engine, context);
    std::cout<<"finish reading engine"<<std::endl;

   

    // get sizes of input and output and allocate memory required for input data and for output data
    std::vector<nvinfer1::Dims> input_dims; // we expect only one input
    std::vector<nvinfer1::Dims> output_dims; 
   
    // std::vector<std::vector<nvinfer1::Dims> output_dims;
    //std::cout<<"output vector len"<<output_dims.size();
   
    

    //get binding nums
    int nbBindings = engine->getNbBindings();
    std::cout<<"input + output nums = "<<nbBindings<<std::endl;
    std::vector<void*> buffers(engine->getNbBindings()); // buffers for input and output data



    for (size_t i = 0; i < engine->getNbBindings(); ++i)
    {
        //get input and output dimension
        nvinfer1::Dims dims = engine->getBindingDimensions(i);
        //std::cout<<dims<<std::endl;
        auto binding_size = getSizeByDim(engine->getBindingDimensions(i)) * batch_size * sizeof(half_float::half);

        //output data type = 0 ,means kFloat
        nvinfer1::DataType type= engine->getBindingDataType(i);
	//std::cout<<"output data type:"<<int(type)<<std::endl;
        if(type == nvinfer1::DataType::kFLOAT){std::cout<<"kfloat"<<std::endl;}
        if(type == nvinfer1::DataType::kHALF){std::cout<<"khalf"<<std::endl;}
      

        cudaMalloc(&buffers[i], binding_size);
        if (engine->bindingIsInput(i))
        {
            input_dims.emplace_back(engine->getBindingDimensions(i));
            std::cout<<"input"<<i<<std::endl;
        }
        else
        {
            output_dims.emplace_back(engine->getBindingDimensions(i));
            auto size = engine->getBindingDimensions(i);
	    //std::cout<<"half_float::half size"<<sizeof(half_float::half)<<std::endl;
            std::cout<<"output"<<i<<std::endl;
            
        }
        
    }
    if (input_dims.empty() || output_dims.empty())
    {
        std::cerr << "Expect at least one input and one output for network\n";
        return -1;
    }


   
    std::cout<<"warmup"<<std::endl;
    double infer_time{0};
    double infer_mean{0};
    double  infer_variance{0};
    double infer_res{0};
    int infer_count{0};
    for (size_t idx = 0; idx != 1; ++idx){

	    // preprocess input data
	    preprocessImage(image_path, (half_float::half *) buffers[0], input_dims[0]);
	    // inference
	    double infer_begin_time = clock();
	    context->enqueue(batch_size, buffers.data(), 0, nullptr);
	    double infer_end_time = clock();
	    infer_count++;
            double infer{0};
            infer = (double)(infer_end_time-infer_begin_time)/CLOCKS_PER_SEC;
            std::cout<<"infer for this "<<infer<<std::endl;
	    infer_time = infer_time+infer;
	    std::cout<<"infer time accumulate"<<infer_time<<std::endl;
	    
	    // postprocess results
	    postprocessResults(buffers, output_dims, batch_size);
     }

    infer_mean = infer_time/ infer_count;
    std::cout<<"infer mean"<<infer_mean<<std::endl;


   // infer_variance = sum([((x - infer_mean) ** 2) for x in infer_time]) / count;
  //  infer_res = infer_variance ** 0.5


    std::cout<<"finished"<<std::endl;

    for (void* buf : buffers)
    {
        cudaFree(buf);
    }
    return 0;
}
