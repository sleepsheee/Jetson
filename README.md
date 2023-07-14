# Jetson

## Useful Links

- Jetson Inference 
https://github.com/dusty-nv/jetson-inference
- sc2 code repository
https://github.com/yoshitomo-matsubara/sc2-benchmark
- ONNX
https://blog.roboflow.com/what-is-onnx/
- TensorRT
https://medium.com/@abhaychaturvedi_72055/understanding-nvidias-tensorrt-for-deep-learning-model-optimization-dad3eb6b26d9
- Semantic Segmentation
https://www.youtube.com/watch?v=AQhkMLaB_fY&list=PL5B692fm6--uQRRDTPsJDp4o0xbzkoyf8&index=15
- sc2 paper
https://arxiv.org/abs/2203.08875



## Split Computing
- Basic Idea : Split the original model into head (executed on mobile device) and tail (executed on edge server) models.

- Bottleneck : The output tensor of head model can be designed to be smaller than the input by introducing the bottleneck layer to the early layers of original model. Thus, a smaller data is transmitted over the channel to edge server instead of input data. This compressed data is then be used as the input of tail model to produce the final output, which is then sent back to the mobile device.

- Encoder and Decoder Architecture: The part of the model executed on the mobile device is called encoder because it generates smaller tensor compared to the input.
Decoder contains part of tail models, it decompress the encoded part.

- Knowledge Distillation : Sc2 uses the orignal pretrained model as teacher and train bottleneck injected model (student). This approach helps with increasing accuracy of the student model without increasing model complexity.


## Run sc2 on Jetson
1. Download the chekpoints :
   https://github.com/yoshitomo-matsubara/sc2-benchmark/releases/tag/v0.0.3
   
2. Downlod file :
   https://github.com/MatteoMendula/splittableFasterCRRNN_BQ/blob/main/script/split_alternate/dummy_client_v4.py

3. Save the encoder as ONNX file
   ```sh
   def parse_to_onnx(self):
        input = [torch.randn((1,3,300,300)).to("cuda")]
        if self.precision == 'fp16':
            input = [torch.randn((1,3,300,300)).to("cuda").half()]
        model = self.ssd300.eval().to("cuda")
        traced_model = torch.jit.trace(model, input)    
        torch.onnx.export(traced_model,  # model being run
                            input,  # model input (or a tuple for multiple inputs)
                            "./models/ssd_{}.onnx".format(self.precision),  # where to save the model (can be a file or file-like object)
                            export_params=True,  # store the trained parameter weights inside the model file
                            opset_version=13,  # the ONNX version to export the model to
                            do_constant_folding=True,  # whether to execute constant folding for optimization
                            input_names=['input'],  # the model's input names
                            output_names=['output'])

    ```
   
4. Install TensorTRT on your machine:
   https://github.com/NVIDIA/TensorRT

   - An example : Linux (aarch64) build with default cuda-12.0
      ```sh
       cd $TRT_OSSPATH
       mkdir -p build && cd build
       cmake .. -DTRT_LIB_DIR=$TRT_LIBPATH -DTRT_OUT_DIR=`pwd`/out -DCMAKE_TOOLCHAIN_FILE=$TRT_OSSPATH/cmake/toolchains/cmake_aarch64-native.toolchain
       make -j$(nproc)
      ```
5. Convert ONNX to TensorRT engine
   - The fisrt part of this command is the installation path of TensorRT package
   ```sh
   '/home/matteo/TensorRT-8.6.1.6/bin/trtexec --onnx=./models/ssd_{}.onnx --saveEngine=./models/ssd_{}.trt'.format(self.precision, self.precision)
        output = subprocess.check_call(cmd.split(' '))
   ```
   You should get a ___.trt file
   
6. Check if TRT inference is the file to run a TRT engine
   https://github.com/MatteoMendula/QuantizingPytorch/blob/main/object_detection/ssd/checkTRT_inference.py
   
