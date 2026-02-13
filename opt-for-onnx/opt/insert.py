import onnx

model_path = "../model/resnet50_fused.onnx"
output_name = "resnetv17_stage4_batchnorm1_fwd"
new_model_path = "../model/resnet50_fused_inserted.onnx"

model = onnx.load(model_path)

for o in model.graph.output:
    if o.name == output_name:
        print(f"{output_name} already in graph output")
        onnx.save(model, new_model_path)
        exit(0)

producer_node = None
for node in model.graph.node:
    if output_name in node.output:
        producer_node = node
        break

if producer_node is None:
    raise RuntimeError(f"Cannot find node producing {output_name}")

print(f"Found producer node: {producer_node.name} ({producer_node.op_type})")

model = onnx.shape_inference.infer_shapes(model)

value_info = None
for v in model.graph.value_info:
    if v.name == output_name:
        value_info = v
        break

if value_info == None:
    print("Shape not inferred, create VlaueInfo manually")
    value_info = onnx.helper.make_tensor_value_info(
        output_name,
        onnx.TensorProto.FLOAT,
        None
    )

model.graph.output.append(value_info)

onnx.save(model, new_model_path)
print(f"Saved new model to {new_model_path}")