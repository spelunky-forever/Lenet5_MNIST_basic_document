# verify_onnx_debug.py
"""
Debug script:
 - uses a deterministic fixed input (np array)
 - runs both PyTorch & ONNXRuntime, prints outputs and diffs
 - prints some useful meta about ONNX graph inputs
Usage:
    python3 verify_onnx_debug.py
Purpose:
    find if mismatch is reproducible and quantify difference
Alternative:
    use a real MNIST sample (see comments below)
"""
import os
import numpy as np
import torch
import onnx
import onnxruntime as ort
from train_lenet import LeNet5, transform  # reuse model class & transform if you used it
from export_onnx import MODEL_PATH, ONNX_PATH, load_model  # assumes export_onnx.py exists

# ensure deterministic
torch.manual_seed(0)
np.random.seed(0)

device = torch.device("cpu")

# 1) load pytorch model (in eval)
pt_model = load_model(MODEL_PATH, device)
pt_model.eval()

# 2) prepare deterministic input: use zeros, ones, and fixed random for testing
# Option A: zeros
inp_zero = np.zeros((1,1,28,28), dtype=np.float32)

# Option B: ones
inp_one = np.ones((1,1,28,28), dtype=np.float32) * 0.5

# Option C: fixed random
inp_rand = np.random.RandomState(1234).randn(1,1,28,28).astype(np.float32)

tests = {
    "zeros": inp_zero,
    "ones": inp_one,
    "fixed_rand": inp_rand
}

# 3) Run PyTorch on each test input
def run_pytorch(np_input):
    t = torch.from_numpy(np_input).to(device)
    # If your model expects normalized input, apply transform (uncomment if used)
    # NOTE: train used transforms.Normalize((0.1307,), (0.3081,))
    # So if you want to mimic real input you should normalize:
    # t = (t - 0.1307) / 0.3081
    with torch.no_grad():
        out = pt_model(t).cpu().numpy()
    return out

# 4) Run ONNXRuntime
sess = ort.InferenceSession(ONNX_PATH, providers=['CPUExecutionProvider'])
input_name = sess.get_inputs()[0].name

def run_onnx(np_input):
    # Ensure dtype is float32
    arr = np_input.astype(np.float32)
    ort_out = sess.run(None, {input_name: arr})
    return ort_out[0]

# 5) compare
for name, arr in tests.items():
    pt_out = run_pytorch(arr)
    onnx_out = run_onnx(arr)
    abs_diff = np.abs(pt_out - onnx_out)
    max_abs = abs_diff.max()
    mean_abs = abs_diff.mean()
    print(f"=== Test: {name} ===")
    print("PyTorch out:", pt_out)
    print("ONNXRuntime out:", onnx_out)
    print(f"max_abs_diff={max_abs:.6f}, mean_abs_diff={mean_abs:.6f}")
    # print per-element differences
    print("diff per element:", (pt_out - onnx_out).tolist())
    print()
