# export_lenet_opset18.py
import torch
from train_lenet import LeNet5  # 使用你已有的 model class
import os

MODEL_PATH = "pytorch_model.bin"   # 你的權重檔（train 結束已存）
OUT_ONNX = "lenet_opset18.onnx"

def load_model(path):
    model = LeNet5()
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model

if __name__ == "__main__":
    model = load_model(MODEL_PATH)
    # dummy input: MNIST 1x28x28
    dummy_input = torch.randn(1, 1, 28, 28, dtype=torch.float32)
    # 匯出 (參數依 Novatek 建議)
    torch.onnx.export(
        model,
        dummy_input,
        OUT_ONNX,
        export_params=True,
        do_constant_folding=False,   # 文件建議關閉 constant folding
        opset_version=18,
        input_names=["input_1"],
        output_names=["output_1"],
        dynamic_axes={'input_1': {0: 'batch_size'}, 'output_1': {0: 'batch_size'}}
    )
    print("Exported ONNX to:", OUT_ONNX)
