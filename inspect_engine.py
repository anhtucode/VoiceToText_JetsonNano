import tensorrt as trt

logger = trt.Logger(trt.Logger.WARNING)

with open("encoder_fp16.engine", "rb") as f:
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(f.read())

print("="*60)

print("Implicit Batch:", engine.has_implicit_batch_dimension)

print()

for i in range(engine.num_bindings):

    print("Binding", i)

    print("Name :", engine.get_binding_name(i))

    print("Input:", engine.binding_is_input(i))

    print("Shape:", engine.get_binding_shape(i))

    print("Type :", engine.get_binding_dtype(i))

    print()