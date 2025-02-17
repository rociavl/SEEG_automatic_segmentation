import torch
import monai
from monai.transforms import Compose, LoadImage, AddChannel, ToTensor, Resize, SpatialPad
from torch.utils.data import DataLoader, Dataset
import os

# Custom dataset class
class BrainMaskDataset(Dataset):
    def __init__(self, dataset_path, transform=None):
        self.dataset_path = dataset_path
        self.transform = transform
        self.images = sorted([os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if f.endswith('.nrrd')])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        image = LoadImage(image_only=True)(image_path)  # Load image (modify as needed)
        label = image  # For simplicity, using the same image for mask (replace with actual mask logic)

        if self.transform:
            image = self.transform(image)

        return image, label

def train_model(dataset_path):
    # Define transforms
    transform = Compose([AddChannel(), ToTensor(), SpatialPad(spatial_size=(128, 128, 128))])

    # Load dataset
    dataset = BrainMaskDataset(dataset_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    # Initialize model, loss, and optimizer
    model = monai.networks.nets.UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=[16, 32, 64, 128],
        strides=[2, 2, 2],
    ).cuda()

    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Training loop
    model.train()
    for epoch in range(10):
        for batch_idx, (image, label) in enumerate(dataloader):
            image, label = image.cuda(), label.cuda()

            # Forward pass
            output = model(image)
            loss = loss_fn(output, label)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch + 1}/10], Step [{batch_idx}/{len(dataloader)}], Loss: {loss.item()}")

    # Save the trained model
    torch.save(model.state_dict(), "brain_mask_model.pth")
    print("Model training complete. Model saved as 'brain_mask_model.pth'")

if __name__ == "__main__":
    dataset_path = "path_to_your_training_data"  
    train_model(dataset_path)
