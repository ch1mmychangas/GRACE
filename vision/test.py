import os
import cv2
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from cvzone.HandTrackingModule import HandDetector

# Device Setup
device = (
    torch.accelerator.current_accelerator().device_type()
    if torch.accelerator.is_available()
    else "cpu"
)

print(f"Using device: {device.upper()}")

# Load class labels
with open("gestureModel/gestureNamesM4.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]
num_classes = len(labels)

# Model Definition
model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.last_channel, num_classes)
model.load_state_dict(torch.load("gestureModel/best_model.pth", map_location=device))
model = model.to(device)
model.eval()

# Image transforms
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

#Setup webcam and hand detector
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
detector = HandDetector(maxHands=1)

offset = 20
imgSize = 300

# Live Webcam Detection
while True:
    success, img = cap.read()
    if not success:
        print("Failed to grab frame")
        break
    imgOutput = img.copy()
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        y1 = max(0, y - offset)
        y2 = min(img.shape[0], y + h + offset)
        x1 = max(0, x - offset)
        x2 = min(img.shape[1], x + w + offset)
        imgCrop = img[y1:y2, x1:x2]

        if imgCrop.size == 0 or imgCrop.shape[0] == 0 or imgCrop.shape[1] == 0:
            continue

        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:min(wGap + imgResize.shape[1], imgSize)] = imgResize[:, :imgSize - wGap]
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:min(hGap + imgResize.shape[0], imgSize), :] = imgResize[:imgSize - hGap, :]

        imgRGB = cv2.cvtColor(imgWhite, cv2.COLOR_BGR2RGB)
        img_tensor = transform(imgRGB).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(img_tensor)
            probs = F.softmax(output, dim=1)
            pred_idx = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred_idx].item()

        label = labels[pred_idx]

        print("Probabilities:", probs.cpu().numpy())
        print("Index:", pred_idx)
        print("Label:", label)
        print("Confidence:", confidence)

        cv2.putText(imgOutput, f"{label} ({int(confidence * 100)}%)", (x, y - 10),
                    cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("imageCrop", imgCrop)
        cv2.imshow("imageWhite", imgWhite)

    cv2.imshow("Image", imgOutput)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()