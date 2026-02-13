import onnxruntime as ort
import numpy as np
import matplotlib.pyplot as plt


model_path = "../model/resnet50_fused_inserted.onnx"            
intermediate_output_name = "resnetv17_stage4_batchnorm1_fwd"  

sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

for inp in sess.get_inputs():
    print(f"Input name: {inp.name}, shape: {inp.shape}, type: {inp.type}")

# 假设输出为 [1, 3, 416, 416]
input_name = sess.get_inputs()[0].name

batch_size = 1
channels = 3
height = 224
width = 224

input_data = np.random.rand(batch_size, channels, height, width).astype(np.float32)

print("Molde outputs: ")
for out in sess.get_outputs():
    print(out.name, out.shape, out.type)

outputs = sess.run([intermediate_output_name], {input_name: input_data})

feature_map = outputs[0]  # 输出形状通常为 [1, C, H, W]
print(f"Raw feature map shape: {feature_map.shape}")

feature_map = feature_map[0]  # shape: (C, H, W)
C, H, W = feature_map.shape
print(f"Feature map after removing batch dimension: {feature_map.shape}")

X = feature_map.reshape(C, H * W)
print(f"Reshaped feature map (C x H*W): {X.shape}")

# 去均值
X = X - X.mean(axis=1, keepdims=True)

# 特征值分解
_, S, _ = np.linalg.svd(X, full_matrices=False)
eigvals = S**2

# 计算累计能量
energy = np.cumsum(eigvals) / np.sum(eigvals)

# 画能量曲线
plt.figure(figsize=(6,4))
plt.plot(energy)
plt.xlabel("Number of components")
plt.ylabel("Cumulative energy")
plt.title("Energy curve of convolution2d_8_output")
plt.grid(True)
plt.show()