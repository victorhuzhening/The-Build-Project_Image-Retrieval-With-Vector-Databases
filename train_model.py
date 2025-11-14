import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from tqdm import tqdm
from src.model import VGGFineTuned

# ---- 1. Config ----
data_dir = ".../caltech101/101_ObjectCategories"
save_path = "weights/model.pth"
batch_size = 16
num_epochs = 2
learning_rate = 1e-4
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

print(f"Using dataset from: {data_dir}")
print(f"Training on: {device}")

# ---- 2. Dataset ----
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])



val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

raw_dataset = datasets.ImageFolder(root=data_dir, transform=train_transform)
raw_total = len(raw_dataset)
val_size = int(0.2 * raw_total)
train_size = raw_total - val_size

train_dataset, val_dataset = random_split(raw_dataset, [train_size, val_size])
# Override val_dataset transform
val_dataset.dataset.transform = val_transform

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

print(f"Dataset contains {len(raw_dataset.classes)} classes and {raw_total} images")
print(f"Training samples: {train_size}, Validation samples: {val_size}")

model = VGGFineTuned(num_classes=len(raw_dataset.classes), embedding_size=128, pretrained=True).to(device)

# ---- 4. Training setup ----
criterion_ce = nn.CrossEntropyLoss()
criterion_triplet = nn.TripletMarginLoss(margin=1.0)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

# ---- 5. Training loop ----
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)  # model only returns logits
        ce_loss = criterion_ce(outputs, labels)
        ce_loss.backward()
        optimizer.step()

        total_loss += ce_loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    avg_loss = total_loss / len(train_loader)
    acc = 100. * correct / total

    # Validation loop
    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            ce_loss_val = criterion_ce(outputs, labels)
            val_loss += ce_loss_val.item()

            _, predicted_val = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted_val.eq(labels).sum().item()

    avg_val_loss = val_loss / len(val_loader)
    val_acc = 100. * val_correct / val_total

    print(f"Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.4f} | Accuracy: {acc:.2f}% | Val Loss: {avg_val_loss:.4f} | Val Accuracy: {val_acc:.2f}%")

    scheduler.step()

# ---- 6. Save ----
os.makedirs("weights", exist_ok=True)
torch.save({"model_state_dict": model.state_dict()}, save_path)
print(f" Model saved to {save_path}")