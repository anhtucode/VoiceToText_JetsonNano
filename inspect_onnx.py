import onnxruntime as ort
import sys

model = sys.argv[1]

sess = ort.InferenceSession(model)

print("=" * 60)
print(model)

print("\nINPUTS")
for i, x in enumerate(sess.get_inputs()):
    print(f"{i}")
    print(" Name :", x.name)
    print(" Shape:", x.shape)
    print(" Type :", x.type)

print("\nOUTPUTS")
for i, x in enumerate(sess.get_outputs()):
    print(f"{i}")
    print(" Name :", x.name)
    print(" Shape:", x.shape)
    print(" Type :", x.type)