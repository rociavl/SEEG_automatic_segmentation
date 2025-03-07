import os
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import SimpleITK as sitk
import matplotlib.pyplot as plt

def load_nrrd(file_path):
    image = sitk.ReadImage(file_path)
    return sitk.GetArrayFromImage(image)

def normalize_ct(ct_array):
    return (ct_array - np.min(ct_array)) / (np.max(ct_array) - np.min(ct_array))

def augment_data(ct_array, mask_array):
    # Random horizontal flip
    if random.random() > 0.5:
        ct_array = np.flip(ct_array, axis=2)
        mask_array = np.flip(mask_array, axis=2)

    # Random vertical flip
    if random.random() > 0.5:
        ct_array = np.flip(ct_array, axis=1)
        mask_array = np.flip(mask_array, axis=1)

    # Random depth flip (along the z-axis)
    if random.random() > 0.5:
        ct_array = np.flip(ct_array, axis=0)
        mask_array = np.flip(mask_array, axis=0)

    return ct_array, mask_array

def process_patient(ct_path, mask_folder):
    # Load CT
    ct_array = load_nrrd(ct_path)
    ct_array = normalize_ct(ct_array)

    # Load and combine masks
    mask_arrays = [load_nrrd(os.path.join(mask_folder, mask)) for mask in os.listdir(mask_folder)]
    combined_mask = np.sum(mask_arrays, axis=0) > 0  
    ct_array, combined_mask = augment_data(ct_array, combined_mask)
    return ct_array, combined_mask.astype(np.uint8)

# Dataset Class
class BrainDataset(Dataset):
    def __init__(self, ct_images, masks):
        self.ct_images = torch.tensor(ct_images, dtype=torch.float32).unsqueeze(1)  
        self.masks = torch.tensor(masks, dtype=torch.float32).unsqueeze(1)  

    def __len__(self):
        return len(self.ct_images)

    def __getitem__(self, idx):
        return self.ct_images[idx], self.masks[idx]

# Load Data
ct_dir = "C:/dataset/CTs"
mask_dir = "C:/dataset/Masks"

patients = os.listdir(ct_dir)
ct_images, masks = [], []

for patient_ct in patients:
    patient_id = os.path.splitext(patient_ct)[0]  
    mask_folder = os.path.join(mask_dir, patient_id)

    if os.path.exists(mask_folder):
        ct_path = os.path.join(ct_dir, patient_ct)
        ct_img, mask = process_patient(ct_path, mask_folder)

        ct_images.append(ct_img)
        masks.append(mask)

ct_images = np.array(ct_images)
masks = np.array(masks)

print(f"✅ Loaded Dataset: {len(ct_images)} CT scans, {len(masks)} masks")
print(f"Shape: CT {ct_images.shape}, Masks {masks.shape}")

dataset = BrainDataset(ct_images, masks)
train_loader = DataLoader(dataset, batch_size=2, shuffle=True)

class UNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(UNet3D, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool3d(2)
        )

        self.decoder = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv3d(64, out_channels, kernel_size=3, padding=1), nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

device = "cuda" if torch.cuda.is_available() else "cpu"
model = UNet3D().to(device)

# Optimizer & Loss Function
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.BCELoss()  # Binary segmentation loss

# Training Loop
epochs = 20
for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    for ct_batch, mask_batch in train_loader:
        ct_batch, mask_batch = ct_batch.to(device), mask_batch.to(device)

        optimizer.zero_grad()
        output = model(ct_batch)
        loss = loss_fn(output, mask_batch)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss / len(train_loader):.4f}")

# Save the model
torch.jit.script(model).save("brain_mask_model_3D.pth")
print("Model saved successfully!")

# Evaluation function
def evaluate_model(model, ct_images, masks, batch_size=2):
    model.eval()
    with torch.no_grad():
        total_correct = 0
        total_pixels = 0
        for ct_batch, mask_batch in zip(ct_images, masks):
            ct_batch = torch.tensor(ct_batch).unsqueeze(0).unsqueeze(0).float().to(device)
            mask_batch = torch.tensor(mask_batch).unsqueeze(0).unsqueeze(0).float().to(device)

            output = model(ct_batch)
            pred = output > 0.5  # Threshold the output

            total_correct += (pred == mask_batch).sum().item()
            total_pixels += torch.numel(mask_batch)

        accuracy = total_correct / total_pixels
        print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Evaluate the model on the training data
evaluate_model(model, ct_images, masks)

