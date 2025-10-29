# export_onnx.py
import torch
import onnx
from train_lenet import LeNet5  # 假設你的模型 class 在 train_lenet.py 裡
import numpy as np
import sys

MODEL_PATH = "pytorch_model.bin"
ONNX_PATH = "lenet.onnx"
DEVICE = torch.device("cpu")  # 匯出到 ONNX 用 CPU 就可以

def load_model(path, device):
    model = LeNet5().to(device)
    sd = torch.load(path, map_location=device)
    model.load_state_dict(sd)
    model.eval()
    return model

def export_onnx(model, onnx_path):
    # 建立一個 dummy input：batch=1, 1x28x28（MNIST）
    dummy_input = torch.randn(1, 1, 28, 28, device=next(model.parameters()).device)
    # 匯出時我們常設定動態 batch size 與 opset_version（兼容性）
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        opset_version=13,                # 常用且穩定的 opset
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
        do_constant_folding=True,
        verbose=False
    )
    print("Exported ONNX model to", onnx_path)

def check_onnx(onnx_path):
    # 用 onnx.checker 檢查 model 是否有效
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model is valid. IR version:", onnx_model.ir_version)
    # 可列出簡要 inputs/outputs
    print("Inputs:", [i.name for i in onnx_model.graph.input])
    print("Outputs:", [o.name for o in onnx_model.graph.output])

if __name__ == "__main__":
    model = load_model(MODEL_PATH, DEVICE)
    export_onnx(model, ONNX_PATH)
    check_onnx(ONNX_PATH)
