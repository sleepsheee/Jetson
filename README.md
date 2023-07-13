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

