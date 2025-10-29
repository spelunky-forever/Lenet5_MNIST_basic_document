# train_lenet.py
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

# ----- 超參數 -----
BATCH_SIZE = 64
EPOCHS = 5
LR = 0.001
MODEL_PATH = "pytorch_model.bin"  # 我們會把 state_dict 儲存在這個檔案

# ----- 資料前處理 (Transforms) -----
# MNIST 的像素值在 0~1 或 0~255，常見標準化 mean/std 如下
transform = transforms.Compose([
    transforms.ToTensor(),  # 轉為張量，將像素變成 [0,1]
    transforms.Normalize((0.1307,), (0.3081,))  # 常用的 MNIST mean/std
])

# ----- 載入資料集 -----
train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_dataset  = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
test_loader  = DataLoader(test_dataset, batch_size=1000, shuffle=False, num_workers=2)

# ----- 定義 LeNet-5 (PyTorch 風格) -----
class LeNet5(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)   # input 1x28x28 -> conv -> 6x24x24
        self.pool1 = nn.AvgPool2d(2)                  # -> 6x12x12
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)  # -> 16x8x8
        self.pool2 = nn.AvgPool2d(2)                  # -> 16x4x4
        self.fc1 = nn.Linear(16*4*4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.tanh(self.conv1(x))
        x = self.pool1(x)
        x = self.tanh(self.conv2(x))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = self.tanh(self.fc1(x))
        x = self.tanh(self.fc2(x))
        x = self.fc3(x)
        return x

# ----- 準備訓練 -----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = LeNet5().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# ----- 訓練回圈 -----
for epoch in range(1, EPOCHS+1):
    model.train()
    running_loss = 0.0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        pbar.set_postfix(loss=running_loss/((pbar.n//BATCH_SIZE)+1))

    # ----- 每個 epoch 結束做一次測試 -----
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc = 100.0 * correct / total
    print(f"Epoch {epoch} Test Accuracy: {acc:.2f}%")

# ----- 儲存模型權重 -----
torch.save(model.state_dict(), MODEL_PATH)
print("Saved model to", MODEL_PATH)
