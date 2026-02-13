import onnx
import numpy as np

def fuse_conv_bn(conv_node, bn_node, initializer_map):
    W = initializer_map[conv_node.input[1]]
    if len(conv_node.input) == 3:
        b = initializer_map[conv_node.input[2]]
    else:
        b = np.zeros(W.shape[0], dtype = W.dtype)
    

    gamma = initializer_map[bn_node.input[1]]
    beta = initializer_map[bn_node.input[2]]
    mean = initializer_map[bn_node.input[3]]
    var = initializer_map[bn_node.input[4]]

    eps = next(
        (attr.f for attr in bn_node.attribute if attr.name == "epsilon"),
        1e-5
    )

    std = np.sqrt(var + eps)
    scale = gamma / std

    W_fused = W * scale[:, None, None, None]
    b_fused = scale * (b - mean) +beta

    return W_fused, b_fused

def fuse_conv_bn_in_graph(model):
    graph = model.graph

    # ---- build initializer map ----
    initializer_map = {
        init.name: onnx.numpy_helper.to_array(init)
        for init in graph.initializer
    }

    new_nodes = []
    removed_nodes = set()
    used_inits = set()

    i = 0
    while i < len(graph.node):
        node = graph.node[i]

        # 匹配 Conv -> BN
        if (
            # node.name == "resnetv17_stage3_conv0_fwd"
            # and graph.node[i + 1].name == "resnetv17_stage3_batchnorm0_fwd"
            node.op_type == "Conv"
            and i + 1 < len(graph.node)
            and graph.node[i + 1].op_type == "BatchNormalization"
            and graph.node[i + 1].input[0] == node.output[0]
        ):
            conv = node
            bn = graph.node[i + 1]

            print(f"Fusing {conv.name} + {bn.name}")

            W_fused, b_fused = fuse_conv_bn(conv, bn, initializer_map)

            # ---- new weight names ----
            W_name = conv.name + "_W_fused"
            b_name = conv.name + "_b_fused"

            graph.initializer.extend([
                onnx.numpy_helper.from_array(W_fused, W_name),
                onnx.numpy_helper.from_array(b_fused, b_name),
            ])
            graph.input.extend([

            ])

            # ---- copy Conv attributes ----
            attrs = {
                attr.name: onnx.helper.get_attribute_value(attr)
                for attr in conv.attribute
            }

            fused_conv = onnx.helper.make_node(
                "Conv",
                inputs=[conv.input[0], W_name, b_name],
                outputs=bn.output,   # 直接接 BN 的输出
                name=conv.name + "_fused",
                **attrs
            )

            new_nodes.append(fused_conv)

            i += 2
        else:
            new_nodes.append(node)
            i += 1

    # ---- rebuild graph ----
    graph.ClearField("node")
    graph.node.extend(new_nodes)

    for node in graph.node:
        for inp in node.input:
            used_inits.add(inp)

    new_initializers = [
        init for init in graph.initializer
        if init.name in used_inits
    ]

    graph.ClearField("initializer")
    graph.initializer.extend(new_initializers)

    new_value_info = [
        vi for vi in graph.value_info
        if vi.name in used_inits
    ]
    graph.ClearField("value_info")
    graph.value_info.extend(new_value_info)

    new_inputs = [
        inpu for inpu in graph.input
        if inpu.name in used_inits
    ]
    graph.ClearField("input")
    graph.input.extend(new_inputs)

    return model


model = onnx.load("../model/rank_onnx_R/120.onnx")
model = fuse_conv_bn_in_graph(model)
onnx.checker.check_model(model)
onnx.save(model, "../model/120_allfused.onnx")