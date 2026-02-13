import onnx
import onnxruntime as ort
import numpy as np

model_before = "../model/resnet50-v1-12.onnx"
model_after = "../model/resnet50_fused_opt.onnx"

batch_input = np.random.randn(1, 3, 224, 224).astype(np.float32)

sess_before = ort.InferenceSession(model_before)
sess_after = ort.InferenceSession(model_after)

def run_model(sess, input_data):
    inputs = {sess.get_inputs()[0].name: input_data}
    outputs = sess.run(None, inputs)
    return outputs[0]

out_before = run_model(sess_before, batch_input)
out_after = run_model(sess_after, batch_input)

abs_diff = np.abs(out_before - out_after)
rel_diff = abs_diff / (np.abs(out_before) + 1e-8)

print("Max absolute difference is: ", np.max(abs_diff))
print("Mean absolute difference is: ", np.mean(abs_diff))
print("Max relative difference is: ", np.max(rel_diff))
print("Mean relative difference is: ", np.mean(rel_diff))