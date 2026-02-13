import onnx 
import numpy as np

def cop_attr(node, name, default=None):
    for a in node.attribute:
        if a.name == name:
            return onnx.helper.get_attribute_value(a)
    return default

model_path = "../model/resnet50_fused.onnx"
impove_model_path = "../model/resnet50_fused_opt.onnx"
output_name = "resnetv17_stage4_batchnorm1_fwd"

model = onnx.load(model_path)

producer_node = None
for node in model.graph.node:
    if output_name in node.output and node.op_type == "Conv":
        producer_node = node
        break

if producer_node == None:
    print(f"Cannot found the producer of {output_name}")
    exit(0)

print(f"The producer of {output_name} is {producer_node.name}")

# 取权重
weight_name = producer_node.input[1]
print(f"weight_name is {weight_name}")
W_init = None

for init in model.graph.initializer:
    if init.name == weight_name:
        W_init = init
        break

if W_init == None:
    print(f"{producer_node} weight tensor not found.")
    exit(0)

W = onnx.numpy_helper.to_array(W_init)
Cout, Cin, kH, kW, = W.shape
print(f"Original Conv weight shape: {W.shape}")

# SVD 分解
Rank = 40
W_mat = W.reshape(Cout, Cin * kH * kW)

U, S, Vt = np.linalg.svd(W_mat, full_matrices = False)

Ur = U[:, :Rank]
Sr = S[:Rank]
Vr = Vt[:Rank, :]

# 构造两个 Conv 的权重
Wp = Vr.reshape(Rank, Cin, kH, kW)

Wh = (Ur * Sr).reshape(Cout, Rank, 1, 1)

# 创建新的 initializer
Wp_name = weight_name + "_proj"
Wh_name = weight_name + "_head"

Wp_init = onnx.numpy_helper.from_array(Wp.astype(np.float32), Wp_name)
Wh_init = onnx.numpy_helper.from_array(Wh.astype(np.float32), Wh_name)

model.graph.initializer.extend([Wp_init, Wh_init])

attrs = {
    attr.name: onnx.helper.get_attribute_value(attr)
    for attr in producer_node.attribute
}

# 创建新 Conv 节点
proj_conv = onnx.helper.make_node(
    "Conv",
    inputs = [producer_node.input[0], Wp_name],
    outputs = [producer_node.name + "_proj_out"],
    name = producer_node.name + "_proj",
    **attrs
)

head_conv = onnx.helper.make_node(
    "Conv",
    inputs = [proj_conv.output[0], Wh_name],
    outputs = producer_node.output,
    name = producer_node.name + "_head",
    kernel_shape = [1, 1],
    pads = [0, 0, 0, 0],
    strides = [1, 1]
)

# 替换 Graph 中的节点
nodes = []
for node in model.graph.node:
    if node != producer_node:
        nodes.append(node)
    else:
        nodes.append(proj_conv)
        nodes.append(head_conv)

model.graph.node.clear()
model.graph.node.extend(nodes)

# 删除旧权重
model.graph.initializer.remove(W_init)

inputs_to_keep = []
for inp in model.graph.input:
    if inp.name != weight_name:
        inputs_to_keep.append(inp)

model.graph.input.clear()
model.graph.input.extend(inputs_to_keep)

onnx.checker.check_model(model)
onnx.save(model, impove_model_path)

print("SVD-pruned ONNX saved to: ", impove_model_path)