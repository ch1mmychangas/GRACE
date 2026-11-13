import os
import numpy as np
import onnx_tool

# Find ONNX model and load it
model_path = os.path.join(os.path.dirname(__file__), "gestureModel", "gesturesModel.onnx")

if not os.path.exists(model_path):
    raise FileNotFoundError("ONNX file path not found. Run exportONNX.py to create one.")

m = onnx_tool.Model(model_path)



# Dummy input to test model
inputs = {
    "input" : np.zeros((1, 3, 224, 224), dtype=np.float32),
}

# Find out the model's dimension and profile it
m.graph.shape_infer(inputs)
m.graph.profile() # Calculate the MACs

# Print results
# m.graph.print_node_map()

total_macs = m.graph.macs[0]
total_flops = total_macs * 2 # FLOPs are roughly 2x MAC operations

model_weight = os.path.getsize(model_path) + os.path.getsize(model_path + ".data")

clean_total_macs = total_macs / 1000000
clean_total_flops = total_flops / 1000000

print("======= Raw Numbers =======")
print(f"Total math operations (MACs): {total_macs}")
print(f"Total math operations (FLOPs): {total_flops}")

print("\n======= Polished Version =======")
print(f"Total math operations (MACs): {clean_total_macs:.2f} Million")
print(f"Total math operations (FLOPs): {clean_total_flops:.2f} Million")
print(f"Model Weight: {model_weight:,} bytes ({(model_weight / 1024 ** 2):.2f} MB)")