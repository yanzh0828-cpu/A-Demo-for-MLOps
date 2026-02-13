import onnx
import onnxruntime as ort
import numpy as np
import os
import matplotlib.pyplot as plt
import sys

def Rank_ONNX(rank, aim_output_name, base_onnx_pth, out_onnx_pth):
    model = onnx.load(base_onnx_pth)
    aim_node = None
    for node in model.graph.node:
        if aim_output_name in node.output and node.op_type == "Conv":
            aim_node = node
            break

    if aim_node == None:
        print(f"The target node {aim_node.name} cannot found, check first!")
        return False
    
    weight_name = aim_node.input[1]
    W_init = None
    for init in model.graph.initializer:
        if init.name == weight_name:
            W_init = init
            break

    if W_init == None:
        print(f"The target node {aim_node}\'s initializer cannot found")
        return False
    
    has_bias = len(aim_node.input) == 3
    if has_bias:
        bias_name = aim_node.input[2]

    W = onnx.numpy_helper.to_array(W_init)
    Cout, Cin, kH, kW = W.shape
    W_mat = W.reshape(Cout, Cin * kH * kW)

    U, S, Vt = np.linalg.svd(W_mat, full_matrices = False)

    Ur = U[:, :rank]
    Sr = S[:rank]
    Vr = Vt[:rank, :]

    Wp = Vr.reshape(rank, Cin, kH, kW)
    Wh = (Ur * Sr).reshape(Cout, rank, 1, 1)

    Wp_name = weight_name + "_proj"
    Wh_name = weight_name + "_head"

    Wp_init = onnx.numpy_helper.from_array(Wp.astype(np.float32), Wp_name)
    Wh_init = onnx.numpy_helper.from_array(Wh.astype(np.float32), Wh_name)

    model.graph.initializer.extend([Wp_init, Wh_init])

    attrs = {
        attr.name: onnx.helper.get_attribute_value(attr)
        for attr in aim_node.attribute
    }

    proj_conv = onnx.helper.make_node(
        "Conv",
        inputs = [aim_node.input[0], Wp_name],
        outputs = [aim_node.name + "_proj_out"],
        name = aim_node.name + "_porj",
        **attrs
    )

    if has_bias:
        head_inputs = [proj_conv.output[0], Wh_name, bias_name]
    else:
        head_inputs = [proj_conv.output[0], Wh_name]

    head_conv = onnx.helper.make_node(
        "Conv",
        inputs = head_inputs,
        outputs = aim_node.output,
        name = aim_node.name + "_head",
        kernel_shape = [1, 1],
        pads = [0, 0, 0, 0],
        strides = [1, 1]
    )

    nodes = []
    for node in model.graph.node:
        if node != aim_node:
            nodes.append(node)
        else:
            nodes.append(proj_conv)
            nodes.append(head_conv)
    
    model.graph.node.clear()
    model.graph.node.extend(nodes)

    model.graph.initializer.remove(W_init)

    inputs_to_keep = []
    for inp in model.graph.input:
        if inp != weight_name:
            inputs_to_keep.append(inp)

    model.graph.input.clear()
    model.graph.input.extend(inputs_to_keep)

    onnx.checker.check_model(model)
    onnx.save(model, out_onnx_pth)

    print(f"New model has been saved at {out_onnx_pth}.")

    return True

def Max_absolute_err(rank_start, rank_end, improve_dic, org_tensor):
    max_abs_err_list = []
    rank = rank_start
    while rank < rank_end + 1:
        rank_str = str(rank)
        opt_tensor = improve_dic[rank_str]
        diff = opt_tensor - org_tensor
        max_abs_err = np.max(np.abs(diff))
        max_abs_err_list.append(max_abs_err)
        rank += 1
    
    if len(max_abs_err_list) == rank_end - rank_start + 1:
        return max_abs_err_list
    else:
        return None

def Mean_absolute_err(rank_start, rank_end, improve_dic, org_tensor):
    mean_abs_err_list = []
    rank = rank_start
    while rank < rank_end + 1:
        rank_str = str(rank)
        opt_tensor = improve_dic[rank_str]
        diff = opt_tensor - org_tensor
        mean_abs_err = np.mean(np.abs(diff))
        mean_abs_err_list.append(mean_abs_err)
        rank += 1
    
    if len(mean_abs_err_list) == rank_end - rank_start + 1:
        return mean_abs_err_list
    else:
        return None
    
def Clipped_relative_err(rank_start, rank_end, improve_dic, org_tensor):
    err_list = []
    rank = rank_start
    alpha = 0.005
    threshold = alpha * np.max(np.abs(org_tensor))
    print("Clipped_relative_err_threshold is: ", threshold)
    while rank < rank_end + 1:
        rank_str = str(rank)
        opt_tensor = improve_dic[rank_str]
        diff = opt_tensor - org_tensor
        mask = np.abs(org_tensor) > threshold
        if np.any(mask):
            clipped_rel_error = np.mean(
                np.abs(diff[mask]) / np.abs(org_tensor[mask])
            )
        else:
            clipped_rel_error = 0.0
        # clipped_rel_error = np.linalg.norm(diff) / np.linalg.norm(org_tensor)
        err_list.append(clipped_rel_error)
        rank += 1

    if len(err_list) == rank_end - rank_start + 1:
        return err_list
    else:
        return None 
    
def L2_norm_err(rank_start, rank_end, improve_dic, org_tensor):
    err_list = []
    rank = rank_start
    while rank < rank_end + 1:
        rank_str = str(rank)
        opt_tensor = improve_dic[rank_str]
        diff = opt_tensor - org_tensor
        clipped_rel_error = np.linalg.norm(diff) / np.linalg.norm(org_tensor)
        err_list.append(clipped_rel_error)
        rank += 1

    if len(err_list) == rank_end - rank_start + 1:
        return err_list
    else:
        return None 

def Val_list(rank_start, rank_end, base_onnx_pth, test_tensor_pth):
    rank = rank_start
    base_output_pth = "../model/rank_onnx"
    opt_val_dic = {}

    input_tensor = onnx.numpy_helper.to_array(onnx.load_tensor(test_tensor_pth + "input_0.pb"))
    # input_tensor = np.random.randn(1, 3, 416, 416).astype(np.float32)

    while rank < rank_end + 1:
        out_onnx_pth = base_output_pth + "/" + str(rank) + ".onnx"
        if Rank_ONNX(rank, "resnetv17_stage4_batchnorm3_fwd", base_onnx_pth, out_onnx_pth) == False:
            print(f"When Rank = {rank}, the function Rank_ONNX is error.")
        
        sess_opt = ort.InferenceSession(out_onnx_pth, providers = ["CUDAExecutionProvider"])

        output_opt_tensor = sess_opt.run(None, {sess_opt.get_inputs()[0].name: input_tensor})[0]

        rank_str = str(rank)
        opt_val_dic[rank_str] = output_opt_tensor
        rank += 1

    sess_org = ort.InferenceSession("../model/resnet50-v1-12.onnx", providers = ["CUDAExecutionProvider"])
    output_org_tensor = sess_org.run(None, {sess_org.get_inputs()[0].name: input_tensor})[0]
    
    return opt_val_dic, output_org_tensor

def Plot(rank_start, rank_end, max_abs_list, mean_abs_list, alip_list, l2_norm_list):
    x = np.arange(rank_start, rank_end + 1)

    print(f"The maximum of Max_abs is {np.max(max_abs_list)} when rank = {rank_start + max_abs_list.index(np.max(max_abs_list))}")
    print(f"The minimum of Max_abs is {np.min(max_abs_list)} when rank = {rank_start + max_abs_list.index(np.min(max_abs_list))}")
    
    print(f"The maximum Mean_abs is {np.max(mean_abs_list)} when rank = {rank_start + mean_abs_list.index(np.max(mean_abs_list))}")
    print(f"The minimum Mean_abs is {np.min(mean_abs_list)} when rank = {rank_start + mean_abs_list.index(np.min(mean_abs_list))}")

    print(f"The minimum alip_abs is {np.min(alip_list)} when rank = {rank_start + alip_list.index(np.min(alip_list))}")
    plt.figure(figsize=(10, 6))
    plt.plot(x, max_abs_list, marker='o', label='max_abs = f(rank)', linewidth=2)
    plt.plot(x, mean_abs_list, marker='s', label='mean_abs = g(rank)', linewidth=2)
    plt.plot(x, alip_list, marker='*', label='alip_err = h(rank)', linewidth=2)
    plt.plot(x, l2_norm_list, marker='+', label='l2_norm_list = l(rank)', linewidth=2)

    plt.xlabel('Rank')
    plt.ylabel('Error')
    plt.title(' err with rank')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.show()

def main():
    rank_s = 700
    rank_e = 900
    opt_val_dic, org_tensor = Val_list(rank_s, rank_e, "../model/twice_svd.onnx", "../test_data_set_0/")
    max_abs_err_list = Max_absolute_err(rank_s, rank_e, opt_val_dic, org_tensor)
    mean_abs_err_list = Mean_absolute_err(rank_s, rank_e, opt_val_dic, org_tensor)
    clip_err_list = Clipped_relative_err(rank_s, rank_e, opt_val_dic, org_tensor)
    l2_norm_list = L2_norm_err(rank_s, rank_e, opt_val_dic, org_tensor)
    Plot(rank_s, rank_e, max_abs_err_list, mean_abs_err_list, clip_err_list, l2_norm_list)
    return 0

if __name__ == "__main__":
    sys.exit(main())