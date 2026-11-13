import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import datasets, models
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
import time
import json

device = (
    torch.accelerator.current_accelerator().device_type()
    if torch.accelerator.is_available()
    else "cpu"
)

print("Hardware acceleration available: ", torch.accelerator.is_available())
print("Device: ", device.upper())

print(f"Training on: {device}")

# --- Parameters ---
data_dir = "gestureData"
img_size = 224
batch_size = 32
epochs = 150
train_split = 0.8
model_save_path = "gestureModel/best_model.pth"
class_names_path = "gestureModel/gestureNamesM4.txt"

# --- Transforms ---
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- Dataset and DataLoader ---
dataset = datasets.ImageFolder(root=data_dir, transform=transform)
class_names = dataset.classes
num_classes = len(class_names)

# Save class names
os.makedirs(os.path.dirname(class_names_path), exist_ok=True)
with open(class_names_path, "w") as f:
    for name in class_names:
        f.write(f"{name}\n")

train_size = int(train_split * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# --- Load pretrained MobileNetV2 ---
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
model.classifier[1] = nn.Linear(model.last_channel, num_classes)
model = model.to(device)

# --- Loss, Optimizer, Scheduler ---
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)

# --- Training Utilities ---
best_val_acc = 0
train_acc_list, val_acc_list = [], []
train_loss_list, val_loss_list = [], []
early_stop_counter = 0
patience = 8

# --- Training Loop ---
for epoch in range(epochs):
    epoch_start_time = time.time()

    model.train()
    train_loss, correct, total = 0.0, 0, 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)


        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    train_acc = correct / total
    train_loss /= total
    train_acc_list.append(train_acc)
    train_loss_list.append(train_loss)

    # --- Validation ---
    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            val_correct += predicted.eq(labels).sum().item()
            val_total += labels.size(0)

    val_acc = val_correct / val_total
    val_loss /= val_total
    val_acc_list.append(val_acc)
    val_loss_list.append(val_loss)

    epoch_duration = time.time() - epoch_start_time
    print(f"Epoch {epoch+1}/{epochs}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}, Time: {epoch_duration:.2f}s")

    # Adjust learning rate based on validation accuracy
    scheduler.step(val_acc)

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), model_save_path)
        early_stop_counter = 0
    else:
        early_stop_counter += 1

    if early_stop_counter >= patience:
        print("Early stopping triggered.")
        break

# --- Plot training history ---
epochs_range = range(1, len(train_acc_list) + 1)
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, train_acc_list, label='Training Accuracy')
plt.plot(epochs_range, val_acc_list, label='Validation Accuracy')
plt.title('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs_range, train_loss_list, label='Training Loss')
plt.plot(epochs_range, val_loss_list, label='Validation Loss')
plt.title('Loss')
plt.legend()

plt.tight_layout()
plt.show()
