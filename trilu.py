# import onnx

# model = onnx.load("onnx_model/decoder_model.onnx")

# for i, node in enumerate(model.graph.node):
#     if node.op_type == "Trilu":
#         print("=" * 60)
#         print(i)
#         print(node.name)
#         print(node.input)
#         print(node.output)

import onnx

m = onnx.load("onnx_model/decoder_model.onnx")
print(any(n.op_type == "Trilu" for n in m.graph.node))