# test_infer.py
import torch
from train_lenet import LeNet5, transform  # reuse model class and transform
from torchvision import datasets
from torch.utils.data import DataLoader

MODEL_PATH = "pytorch_model.bin"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 載入模型
model = LeNet5().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# 拿一張測試資料來做推論
test_dataset = datasets.MNIST(root="./data", train=False, download=False, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

import random
idx = random.randint(0, len(test_dataset)-1)
img, label = test_dataset[idx]
with torch.no_grad():
    output = model(img.unsqueeze(0).to(device))
    pred = output.argmax(dim=1).item()

print("Ground truth:", label)
print("Prediction  :", pred)
