import onnxruntime as ort

sess = ort.InferenceSession("decoder_model.onnx")

print("=== INPUTS ===")
for x in sess.get_inputs():
    print(x.name, x.shape, x.type)

print("\n=== OUTPUTS ===")
for x in sess.get_outputs():
    print(x.name, x.shape, x.type)