
# ==============================
# train.py
# ==============================

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =====================================
# Dataset Class (Handles NA + Missing)
# =====================================

class MultiLabelDataset(Dataset):
    def __init__(self, image_dir, label_file, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        
        available_images = set(os.listdir(image_dir))
        self.data = []
        skipped = 0

        with open(label_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                img_name = parts[0]
                labels = parts[1:]

                if img_name in available_images:
                    self.data.append((img_name, labels))
                else:
                    skipped += 1

        print(f"Skipped {skipped} missing images")
        self.num_classes = len(self.data[0][1])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name, labels = self.data[idx]
        img_path = os.path.join(self.image_dir, img_name)

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        label_tensor = []
        mask_tensor = []

        for l in labels:
            if l == "NA":
                label_tensor.append(0.0)
                mask_tensor.append(0.0)
            else:
                label_tensor.append(float(l))
                mask_tensor.append(1.0)

        return image, torch.tensor(label_tensor), torch.tensor(mask_tensor)


# =====================================
# Transforms
# =====================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

dataset = MultiLabelDataset("images", "labels.txt", transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

num_classes = dataset.num_classes


# =====================================
# Handle Class Imbalance
# =====================================

label_counts = np.zeros(num_classes)
total_counts = np.zeros(num_classes)

for _, labels, masks in dataset:
    label_counts += (labels.numpy() * masks.numpy())
    total_counts += masks.numpy()

pos_weights = (total_counts - label_counts) / (label_counts + 1e-5)
pos_weights = torch.tensor(pos_weights).to(device)


# =====================================
# Model (Fine-tuning ResNet18)
# =====================================

model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

criterion = nn.BCEWithLogitsLoss(reduction='none', pos_weight=pos_weights)
optimizer = optim.Adam(model.parameters(), lr=1e-4)


# =====================================
# Training Loop
# =====================================

epochs = 10
loss_list = []

model.train()

for epoch in range(epochs):
    for images, labels, masks in dataloader:

        images = images.to(device)
        labels = labels.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)

        loss = criterion(outputs, labels)
        loss = loss * masks
        loss = loss.mean()

        loss.backward()
        optimizer.step()

        loss_list.append(loss.item())

    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")




torch.save(model.state_dict(), "multilabel_model.pth")


# =====================================
# Loss Curve Plot
# =====================================

plt.figure()
plt.plot(loss_list)
plt.xlabel("iteration_number")
plt.ylabel("training_loss")
plt.title("Aimonk_multilabel_problem")
plt.savefig("loss_plot.png")
plt.show()
