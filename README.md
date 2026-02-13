# A-Demo-for-MLOps
This Project is using for exercising which also recording my learning process for MLOps and ONNX. Welcome everyone to come and exchange ideas.

## The optimization of the resnet50-v1-12 onnx model
Here, I using the onnx model from [ONNX MODEL ZOO](https://github.com/onnx/models/tree/main#onnx-model-zoo). And the ONNX model is [ResNet](https://github.com/onnx/models/blob/main/validated/vision/classification/resnet/model/resnet50-v1-12.tar.gz). 

Up til now, my primary work involves fusing Conv and BatchNormalization layers in the ONNX model to optimize inference efficiency, and applying SVD-based low-rank decomposition to selected Conv layers for model compression.

### Project Workflow So Far

1. Preprocess the ONNX file and identify candidate nodes for optimization.
   - 
