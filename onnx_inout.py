import onnx

for model_name in ["encoder_model.onnx", "decoder_model.onnx"]:
    print("=" * 80)
    print(model_name)

    model = onnx.load(f"onnx_model/{model_name}")

    print("\nInputs:")
    for inp in model.graph.input:
        dims = []
        for d in inp.type.tensor_type.shape.dim:
            dims.append(d.dim_param if d.dim_param else d.dim_value)
        print(f"  {inp.name}: {dims}")

    print("\nOutputs:")
    for out in model.graph.output:
        dims = []
        for d in out.type.tensor_type.shape.dim:
            dims.append(d.dim_param if d.dim_param else d.dim_value)
        print(f"  {out.name}: {dims}")