import argparse, os, sys
import torch

class LeNet5(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 6, kernel_size=5)
        self.tanh = torch.nn.Tanh()
        self.pool = torch.nn.AvgPool2d(2)
        self.conv2 = torch.nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = torch.nn.Linear(16*4*4, 120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 10)
    def forward(self, x):
        x = self.pool(self.tanh(self.conv1(x)))
        x = self.pool(self.tanh(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.tanh(self.fc1(x))
        x = self.tanh(self.fc2(x))
        x = self.fc3(x)
        return x

def try_load_weights(model, wpath):
    data = torch.load(wpath, map_location='cpu')
    # common cases
    if isinstance(data, dict):
        # state_dict or wrapped object
        if 'state_dict' in data:
            sd = data['state_dict']
            try:
                model.load_state_dict(sd)
                return model
            except Exception:
                sd = {k.replace('module.', ''): v for k,v in sd.items()}
                model.load_state_dict(sd)
                return model
        try:
            model.load_state_dict(data)
            return model
        except Exception:
            pass
    # full model object
    try:
        if hasattr(data, 'state_dict'):
            model.load_state_dict(data.state_dict())
            return model
    except Exception:
        pass
    # fallback: maybe it's saved Module
    if isinstance(data, torch.nn.Module):
        return data
    raise RuntimeError("無法解析權重檔格式: " + wpath)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--weights', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--opset', type=int, default=18)
    p.add_argument('--batch', type=int, default=1)
    p.add_argument('--input-name', default='input_1')
    p.add_argument('--output-name', default='Gemm_output_1_Y')  # 你之前 ONNX 顯示的 output 名稱
    args = p.parse_args()

    model = LeNet5()
    try:
        model = try_load_weights(model, args.weights)
    except Exception as e:
        print("載入權重失敗:", e)
        sys.exit(2)

    model.eval()
    # Lenet 的 channel=1, size=28x28
    dummy_input = torch.randn(args.batch, 1, 28, 28, device='cpu')

    outdir = os.path.dirname(args.out)
    if outdir and not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)

    # Try to use keep_initializers_as_inputs if available; wrap in try/except
    export_kwargs = dict(
        model=model,
        args=(dummy_input,),
        f=args.out,
        export_params=True,
        opset_version=args.opset,
        verbose=False,
        input_names=[args.input_name],
        output_names=[args.output_name],
        do_constant_folding=False  # per Novatek 的建議
    )

    # conditional flag (some torch versions accept it)
    try:
        export_kwargs['keep_initializers_as_inputs'] = False
    except Exception:
        pass

    print("Export params:", {k: (v if k in ('f','args') else type(v).__name__ if k=='model' else v) for k,v in export_kwargs.items() if k!='args'})
    # perform export
    torch.onnx.export(**export_kwargs)
    print("Export 完成:", args.out)
    print("檔案大小:", os.path.getsize(args.out), "bytes")

if __name__ == '__main__':
    main()
