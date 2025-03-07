import torch
import numpy as np
import SimpleITK as sitk
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import nn
from torch import optim


class ElectrodeDataset(Dataset):
    def __init__(self, patient_data):
        self.patient_data = patient_data
        self.patient_ids = list(patient_data.keys())

    def __len__(self):
        return sum(len(self.patient_data[pid]["images"]) for pid in self.patient_ids)

    def load_nrrd(self, path):
        image = sitk.ReadImage(path)
        array = sitk.GetArrayFromImage(image)
        array = (array - np.min(array)) / (np.max(array) - np.min(array))  # Normalize
        return torch.tensor(array, dtype=torch.float32).unsqueeze(0)  # Add channel dim

    def __getitem__(self, idx):
        patient_id = np.random.choice(self.patient_ids)

        image_path = np.random.choice(self.patient_data[patient_id]["images"])
        noisy_mask_path = np.random.choice(self.patient_data[patient_id]["noisy_masks"])
        
        image = self.load_nrrd(image_path)
        noisy_mask = self.load_nrrd(noisy_mask_path)
        correct_mask = self.load_nrrd(self.patient_data[patient_id]["correct_mask"])

        return image, noisy_mask, correct_mask

class ElectrodeDataset(Dataset):
    def __init__(self, patient_data):
        self.patient_data = patient_data
        self.patient_ids = list(patient_data.keys())

    def __len__(self):
        return sum(len(self.patient_data[pid]["images"]) for pid in self.patient_ids)

    def load_nrrd(self, path):
        image = sitk.ReadImage(path)
        array = sitk.GetArrayFromImage(image)
        array = (array - np.min(array)) / (np.max(array) - np.min(array))  
        return torch.tensor(array, dtype=torch.float32).unsqueeze(0)  

    def __getitem__(self, idx):

        patient_id = np.random.choice(self.patient_ids)

        image_path = np.random.choice(self.patient_data[patient_id]["images"])
        noisy_mask_path = np.random.choice(self.patient_data[patient_id]["noisy_masks"])
        
        image = self.load_nrrd(image_path)
        noisy_mask = self.load_nrrd(noisy_mask_path)
        correct_mask = self.load_nrrd(self.patient_data[patient_id]["correct_mask"])

        return image, noisy_mask, correct_mask

train_dataloader = DataLoader(ElectrodeDataset(train_patient_data), batch_size=2, shuffle=True)

for epoch in range(50):
    for image, noisy_mask, correct_mask in train_dataloader:
        image, noisy_mask, correct_mask = image.to(device), noisy_mask.to(device), correct_mask.to(device)

        optimizer.zero_grad()
        enhanced_pred, seg_pred = model(image)
        
        loss = total_loss(seg_pred, noisy_mask, correct_mask)
        loss.backward()
        optimizer.step()
    
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")
