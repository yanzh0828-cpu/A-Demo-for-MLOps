import os
import time
import numpy as np
import onnxruntime as ort
import cv2
from tqdm import tqdm

# ==============================
# 1. 配置
# ==============================
model_path = "../model/four_svd.onnx"
val_dir = "./val"  # 验证集已经按类别拆分
warmup_iters = 20
max_images = 2000  # 测试用

# ==============================
# 2. 加载 ONNX 模型
# ==============================
session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name

# ==============================
# 3. 构建类别映射
# ==============================
class_names = sorted(os.listdir(val_dir))
class_to_idx = {cls_name: idx for idx, cls_name in enumerate(class_names)}

# ==============================
# 4. 获取所有图片路径和对应标签
# ==============================
image_paths = []
labels = []

for cls_name in class_names:
    cls_dir = os.path.join(val_dir, cls_name)
    cls_images = sorted(os.listdir(cls_dir))
    for img_name in cls_images:
        image_paths.append(os.path.join(cls_dir, img_name))
        labels.append(class_to_idx[cls_name])

if max_images:
    image_paths = image_paths[:max_images]
    labels = labels[:max_images]

# ==============================
# 5. 预热
# ==============================
print("Warming up...")
dummy_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
for _ in range(warmup_iters):
    session.run(None, {input_name: dummy_input})
print("Warmup done.\n")

# ==============================
# 6. 推理 + 计时 + Top-1/Top-5
# ==============================
top1 = 0
top5 = 0

total_start_time = time.time()  # 端到端计时
model_infer_time = 0.0         # 仅模型推理时间

# ONNX Model Zoo ResNet50-v1-12 预处理参数
mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
std  = np.array([58.395, 57.12, 57.375], dtype=np.float32)

for img_path, gt in tqdm(zip(image_paths, labels), total=len(image_paths)):

    # ======================
    # 图片读取 + 预处理 (不计入模型推理时间)
    # ======================
    img = cv2.imread(img_path).astype(np.float32)

    h, w, _ = img.shape
    scale = 256.0 / min(h, w)
    new_h = int(round(h * scale))
    new_w = int(round(w * scale))
    img = cv2.resize(img, (new_w, new_h))

    start_h = (new_h - 224) // 2
    start_w = (new_w - 224) // 2
    img = img[start_h:start_h+224, start_w:start_w+224, :]

    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, 0)
    img = (img - mean[:, None, None]) / std[:, None, None]
    img = img.astype(np.float32)

    # ======================
    # 模型推理 (单独计时)
    # ======================
    t0 = time.time()
    outputs = session.run(None, {input_name: img})
    t1 = time.time()
    model_infer_time += (t1 - t0)

    logits = outputs[0]

    pred = np.argsort(logits[0])[::-1]
    if pred[0] == gt:
        top1 += 1
    if gt in pred[:5]:
        top5 += 1

total_end_time = time.time()

# ==============================
# 7. 结果输出
# ==============================
total_images = len(image_paths)
total_time = total_end_time - total_start_time
avg_latency = total_time / total_images
fps = total_images / total_time
avg_model_latency = model_infer_time / total_images
model_fps = total_images / model_infer_time

print("\n==============================")
print("Evaluation Results (ResNet50-v1-12)")
print("==============================")
print(f"Total Images: {total_images}")
print(f"Top-1 Accuracy: {top1 / total_images:.4f}")
print(f"Top-5 Accuracy: {top5 / total_images:.4f}")
print("\n--- End-to-End Performance ---")
print(f"Total Time: {total_time:.4f} sec")
print(f"Average Latency: {avg_latency*1000:.4f} ms/image")
print(f"Throughput (FPS): {fps:.2f}")
print("\n--- Model Inference Performance ---")
print(f"Model Inference Time: {model_infer_time:.4f} sec")
print(f"Average Model Latency: {avg_model_latency*1000:.4f} ms/image")
print(f"Model Throughput (FPS): {model_fps:.2f}")
print("==============================")
