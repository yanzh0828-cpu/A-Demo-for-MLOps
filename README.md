# A-Demo-for-MLOps

This Project is using for exercising which also recording my learning process for MLOps and ONNX. Welcome everyone to come and exchange ideas.

## The optimization of the resnet50-v1-12 onnx model

Here, I using the onnx model from [ONNX MODEL ZOO](https://github.com/onnx/models/tree/main#onnx-model-zoo). And the ONNX model is [ResNet](https://github.com/onnx/models/blob/main/validated/vision/classification/resnet/model/resnet50-v1-12.tar.gz).

Up til now, my primary work involves fusing Conv and BatchNormalization layers in the ONNX model to optimize inference efficiency, and applying SVD-based low-rank decomposition to selected Conv layers for model compression.

### Project Workflow So Far

- Preprocess the ONNX file and identify candidate nodes for optimization

  - [Insert](/opt-for-onnx/opt/insert.py) file is used to insert intermediate node, cause we need view the intermediate feature map and then plot the energy rank figure [plot](/opt-for-onnx/opt/SVD_plot.py).
  - By fusing Conv and BatchNormalization layers, the model size and computational cost are reduced, and the error introduced by low-rank decomposition is prevented from being amplified by BN [fusing](/opt-for-onnx/opt/fused.py)
- Optimization
  - By iterating over rank values for the target nodes, plot the relationship between rank and various errors (L2 norm, clipped relative error, mean error, max error), and select the final rank based on these results. This is primarily done to reduce the model's floating-point operations (FLOPs). [rank](/opt-for-onnx/opt/rank_err.py).
- Test
  - By testing with the [ImageNet ILSVRC2012](https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar), I derived the inference latency. [test](/opt-for-onnx/test/)
- Counterintuitive
  - The latency of optimal onnx higher than original onnx. This is because, on CUDA, a reduction in FLOPs can lead to decreased parallel computing efficiency. After optimization, the GPU may spend more time on kernel scheduling rather than actual parallel computation. On the CPU, performing SVD requires decomposing the original node into two nodes, which increases memory access, scheduling, and cache hits, so the time overhead can outweigh the time saved by reduced FLOPs.

Therefore, the computational savings (FLOPs) from this method are more significant on hardware-constrained devices (such as mobile, embedded systems, or FPGA), or when using large batch sizes on CPU/GPU, where theoretical FLOPs reductions can have a greater impact.

However, in practical engineering applications, a more holistic analysis is needed. A small reduction in FLOPs does not necessarily translate to faster inference, but it provides potential performance headroom for subsequent MLOps optimizations.

### ONNX Model Deployment

The server is implemented in C++, with the main server program decoupled from the inference module. This design ensures that updating or replacing the inference model does not affect the normal operation of the main service.

Unix Domain Sockets (UDS) are used to guarantee efficient local data transmission between processes. Additionally, the batch size is carefully tuned to maximize GPU utilization and overall performance.