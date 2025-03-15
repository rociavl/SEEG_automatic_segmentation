import os
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from monai import transforms
from monai.networks.nets import UNet
from monai.data import DataLoader, Dataset
from monai.transforms import LoadImage  # Correct import for LoadImage
import nrrd
from sklearn.metrics import jaccard_score
import torch.amp
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# prompt: import drive mount

from google.colab import drive
drive.mount('/content/drive')


ct_dir = "/content/drive/MyDrive/TFG 💪🧠/Code/Modelos /Brain_mask_model/Dataset/CT"
mask_dir = "/content/drive/MyDrive/TFG 💪🧠/Code/Modelos /Brain_mask_model/Dataset/MASK"

# Function to pair CT scans and masks
def get_paired_files(ct_path, mask_path):
    ct_files = {f.split('_')[0]: f for f in os.listdir(ct_path) if f.endswith(".nrrd")}
    mask_files = {f.split('_')[0]: f for f in os.listdir(mask_path) if f.endswith(".nrrd")}
    
    paired_files = []
    for patient_id in ct_files.keys():
        if patient_id in mask_files:
            paired_files.append((ct_files[patient_id], mask_files[patient_id]))
        else:
            print(f"Warning: No mask found for {ct_files[patient_id]}")

    return paired_files

file_pairs = get_paired_files(ct_dir, mask_dir)

class NRRDDataset(Dataset):
    def __init__(self, ct_path, mask_path, file_pairs, transform=None):
        self.ct_path = ct_path
        self.mask_path = mask_path
        self.file_pairs = file_pairs
        self.transform = transform

        # LoadImage is a MONAI transform to load .nrrd files
        self.load_ct = LoadImage(image_only=True)
        self.load_mask = LoadImage(image_only=True)

    def __len__(self):
        return len(self.file_pairs)

    def __getitem__(self, idx):
        ct_file, mask_file = self.file_pairs[idx]
        
        ct_data = self.load_ct(os.path.join(self.ct_path, ct_file))
        mask_data = self.load_mask(os.path.join(self.mask_path, mask_file))

        # Normalize CT scans to [0,1]
        ct_data = np.clip(ct_data, -1000, 1000)  
        ct_data = (ct_data + 1000) / 2000  

        # Ensure binary masks
        mask_data = (mask_data > 0).astype(np.float32)  

        sample = {"image": ct_data, "mask": mask_data}

        if self.transform:
            # Apply transforms to 'image' and 'mask' separately
            sample["image"] = self.transform(sample["image"])  
            sample["mask"] = self.transform(sample["mask"])

        return sample["image"], sample["mask"]
from monai.transforms import Compose, Resize, RandRotate, RandFlip, RandShiftIntensity, RandAffine, ToTensor

# Define transformations for data augmentation
transform = Compose([
    Resize(spatial_size=(256, 256, 256)),  # Resize the 3D volumes to 256x256x256
    RandRotate(range_x=90, range_y=90, range_z=90),  # Random 90-degree rotation for each axis
    RandFlip(prob=0.5, spatial_axis=(0, 1, 2)),  # Random flip with a 50% probability along all axes
    RandShiftIntensity(offsets=(0.1, 0.2)),  # Random intensity shift (brightness adjustment)
    RandAffine(prob=0.5, translate_range=(10, 10, 10), rotate_range=(10, 10, 10), scale_range=(0.1, 0.1, 0.1)),  # Random affine transformation
    ToTensor(),  # Convert data to torch tensors
])


dataset = NRRDDataset(ct_dir, mask_dir, file_pairs, transform=transform)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

model = UNet(
    spatial_dims=3,  # 3D convolutions
    in_channels=1,   # Single channel input (grayscale CT)
    out_channels=1,  # Binary output (mask)
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),  # Downsampling strides
    num_res_units=2,  # Number of residual units
).to(device)

# Define Dice + BCE Loss
class DiceBCELoss(torch.nn.Module):
    def __init__(self):
        super(DiceBCELoss, self).__init__()

    def forward(self, pred, target):
        bce = F.binary_cross_entropy_with_logits(pred, target)
        smooth = 1e-5
        pred = torch.sigmoid(pred)
        intersection = (pred * target).sum()
        dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
        return 1 - dice + bce

# Optimizer & Scheduler
loss_fn = DiceBCELoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

# Mixed Precision Scaler
scaler = torch.amp.GradScaler()


def train_model(model, dataloader, loss_fn, optimizer, scheduler, epochs=10):
    model.train()

    for epoch in range(epochs):
        epoch_loss = 0
        for images, masks in dataloader:
            images, masks = images.to(device).float(), masks.to(device).float()  # Ensure float32

            optimizer.zero_grad()

            # Mixed precision
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = loss_fn(outputs, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(dataloader):.4f}")

# Evaluation function
def evaluate_model(model, dataloader):
    model.eval()
    dice_scores = []
    iou_scores = []

    with torch.no_grad():
        for images, masks in dataloader:
            images, masks = images.to(device).float(), masks.to(device).float()  # Ensure float32

            with torch.amp.autocast('cuda'):
                outputs = torch.sigmoid(model(images)).cpu().numpy()
                masks = masks.cpu().numpy()

            outputs = (outputs > 0.5).astype(np.uint8)

            for i in range(len(outputs)):
                intersection = (outputs[i] * masks[i]).sum()
                dice = (2. * intersection) / (outputs[i].sum() + masks[i].sum() + 1e-5)
                iou = jaccard_score(masks[i].flatten(), outputs[i].flatten(), average='binary')

                dice_scores.append(dice)
                iou_scores.append(iou)

    print(f"Mean Dice Score: {np.mean(dice_scores):.4f}")
    print(f"Mean IoU Score: {np.mean(iou_scores):.4f}")

train_model(model, dataloader, loss_fn, optimizer, scheduler, epochs=20)

evaluate_model(model, dataloader)
