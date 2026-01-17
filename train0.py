import os
import pandas as pd
from glob import glob

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms, models, datasets

device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------
# 1. 数据增强 & 预处理
# -------------------------
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# -------------------------
# 2. 数据集
# -------------------------
train_dataset = datasets.ImageFolder("train", transform=transform_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

class_names = train_dataset.classes
print("Classes:", class_names)

# -------------------------
# 3. 定义模型（MobileNetV2）
# -------------------------
model = models.mobilenet_v2(pretrained=True)
model.classifier[1] = nn.Linear(model.last_channel, 4)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# -------------------------
# 4. 训练
# -------------------------
EPOCHS = 20

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {running_loss/len(train_loader):.4f}")

# 保存模型
torch.save(model.state_dict(), "model0.pth")
print("模型已保存为 model0.pth")
