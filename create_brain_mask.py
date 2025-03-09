import os
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import SimpleITK as sitk
import matplotlib.pyplot as plt
import os
import numpy as np
import SimpleITK as sitk
from collections import defaultdict

# Set dataset paths
ct_dir = "C:/dataset/CTs"
mask_dir = "C:/dataset/Masks"

# Patients available in dataset
valid_patients = {"patient1", "patient4", "patient5", "patient6", "patient8"}

# Function to load NRRD files safely
def load_nrrd(file_path):
    """Load an NRRD file as a NumPy array with error handling."""
    try:
        image = sitk.ReadImage(file_path)
        array = sitk.GetArrayFromImage(image)
        return array.astype(np.float32)
    except Exception as e:
        print(f"❌ Error loading {file_path}: {e}")
        return None

# Normalize CT scans to [0, 1] range
def normalize_ct(ct_array):
    """Normalize CT scan values between 0 and 1."""
    min_val, max_val = np.min(ct_array), np.max(ct_array)
    return (ct_array - min_val) / (max_val - min_val + 1e-8)  # Avoid division by zero

# Data Augmentation
def augment_data(ct_array, mask_array):
    """Apply random flips, intensity scaling, and Gaussian noise."""
    # Random horizontal flip
    if random.random() > 0.5:
        ct_array = np.flip(ct_array, axis=2)
        mask_array = np.flip(mask_array, axis=2)

    # Random vertical flip
    if random.random() > 0.5:
        ct_array = np.flip(ct_array, axis=1)
        mask_array = np.flip(mask_array, axis=1)

    # Random depth flip (Z-axis)
    if random.random() > 0.5:
        ct_array = np.flip(ct_array, axis=0)
        mask_array = np.flip(mask_array, axis=0)

    # Random intensity scaling
    if random.random() > 0.5:
        scale_factor = random.uniform(0.9, 1.1)
        ct_array *= scale_factor

    # Add random Gaussian noise
    if random.random() > 0.5:
        noise = np.random.normal(0, 0.02, ct_array.shape)
        ct_array = np.clip(ct_array + noise, 0, 1)  # Keep values in [0,1]

    return ct_array, mask_array

# Load all patient CTs and masks
ct_images, masks = [], []

for patient_id in valid_patients:
    ct_path = os.path.join(ct_dir, f"{patient_id}_CT.nrrd")
    if not os.path.exists(ct_path):
        print(f"⚠️ Missing CT scan for {patient_id}, skipping...")
        continue

    # Load CT scan
    ct_array = load_nrrd(ct_path)
    if ct_array is None:
        continue
    ct_array = normalize_ct(ct_array)

    # Load corresponding masks
    patient_masks = [os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.startswith(f"{patient_id}_mask")]
    if not patient_masks:
        print(f"⚠️ No masks found for {patient_id}, skipping...")
        continue

    # Combine all masks into a single binary mask
    mask_arrays = [load_nrrd(mask_path) for mask_path in patient_masks if load_nrrd(mask_path) is not None]
    if not mask_arrays:
        continue
    combined_mask = np.clip(np.sum(mask_arrays, axis=0), 0, 1)  # Ensure binary mask

    # Apply augmentation
    ct_array, combined_mask = augment_data(ct_array, combined_mask)

    # Store in lists
    ct_images.append(ct_array)
    masks.append(combined_mask.astype(np.uint8))

# Convert lists to NumPy arrays
ct_images = np.array(ct_images)
masks = np.array(masks)

# Print dataset information
print(f"✅ Loaded Dataset: {len(ct_images)} CT scans, {len(masks)} masks")
print(f"Shape: CT {ct_images.shape}, Masks {masks.shape}")

# Torch Dataset Class
class BrainDataset(Dataset):
    def __init__(self, ct_images, masks):
        self.ct_images = torch.tensor(ct_images, dtype=torch.float32).unsqueeze(1)  # Add channel dim
        self.masks = torch.tensor(masks, dtype=torch.float32).unsqueeze(1)  # Add channel dim

    def __len__(self):
        return len(self.ct_images)

    def __getitem__(self, idx):
        return self.ct_images[idx], self.masks[idx]

# Create Dataset and DataLoader
dataset = BrainDataset(ct_images, masks)
train_loader = DataLoader(dataset, batch_size=2, shuffle=True)

print("🔥 DataLoader Ready!")


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

