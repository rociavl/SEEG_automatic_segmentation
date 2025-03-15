import numpy as np
import torch
import torch.nn as nn
import SimpleITK as sitk
from torch.amp import autocast

# Define UNet3D
class UNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(UNet3D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, 8, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool3d(2)
        )
        self.decoder = nn.Sequential(
            nn.Conv3d(8, out_channels, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size, num_patches, channels, d, h, w = x.shape
        x = x.view(batch_size * num_patches, self.in_channels, d, h, w)
        x = self.encoder(x)
        x = self.decoder(x)
        x = x.view(batch_size, num_patches, self.out_channels, d, h, w)
        return x

# Load NRRD with downsampling
def load_nrrd(file_path, downsample_factor=2):
    try:
        image = sitk.ReadImage(file_path)
        original_size = image.GetSize()
        image = sitk.Resample(image, [int(s/downsample_factor) for s in original_size],
                            sitk.Transform(), sitk.sitkLinear,
                            image.GetOrigin(), [s*downsample_factor for s in image.GetSpacing()],
                            image.GetDirection(), 0.0, image.GetPixelID())
        array = sitk.GetArrayFromImage(image)
        del image
        return array.astype(np.float16), original_size  # Return original size for upsampling later
    except Exception as e:
        print(f"❌ Error loading {file_path}: {e}")
        return None, None

# Normalize CT
def normalize_ct(ct_array):
    ct_array -= ct_array.min()
    ct_array /= (ct_array.max() + 1e-8)
    return ct_array

# Extract patches
def extract_patches(ct_array, patch_size=(16, 16, 8), stride=(8, 8, 4)):
    d, h, w = ct_array.shape
    pd, ph, pw = patch_size
    sd, sh, sw = stride
    patches = []
    coords = []
    for z in range(0, max(d - pd + 1, 1), sd):
        for y in range(0, max(h - ph + 1, 1), sh):
            for x in range(0, max(w - pw + 1, 1), sw):
                patch = ct_array[z:z+pd, y:y+ph, x:x+pw]
                if patch.shape == patch_size:
                    patches.append(patch)
                    coords.append((z, y, x))
    return np.array(patches), coords

# Reconstruct volume
def reconstruct_volume(patches, coords, original_shape, patch_size=(16, 16, 8)):
    d, h, w = original_shape
    pd, ph, pw = patch_size
    output = np.zeros(original_shape, dtype=np.float32)
    count = np.zeros(original_shape, dtype=np.float32)
    for patch, (z, y, x) in zip(patches, coords):
        output[z:z+pd, y:y+ph, x:x+pw] += patch
        count[z:z+pd, y:y+ph, x:x+pw] += 1
    count[count == 0] = 1
    return output / count

# Upsample mask to original size
def upsample_mask(mask, original_size):
    mask_image = sitk.GetImageFromArray(mask)
    mask_image = sitk.Resample(mask_image, original_size,
                              sitk.Transform(), sitk.sitkNearestNeighbor,  # Nearest neighbor for binary mask
                              mask_image.GetOrigin(), mask_image.GetSpacing(),
                              mask_image.GetDirection(), 0.0, mask_image.GetPixelID())
    return sitk.GetArrayFromImage(mask_image)

# Save NRRD
def save_nrrd(array, output_path, reference_image_path):
    ref_image = sitk.ReadImage(reference_image_path)
    output_image = sitk.GetImageFromArray(array)
    output_image.CopyInformation(ref_image)
    sitk.WriteImage(output_image, output_path)
    print(f"✅ Mask saved to {output_path}")

# Generate brain mask
def generate_brain_mask(ct_file_path, model_path="unet3d_model.pth", output_path="brain_mask.nrrd", device=torch.device("cpu")):
    model = UNet3D(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    ct_array, original_size = load_nrrd(ct_file_path)
    if ct_array is None:
        return

    downsampled_shape = ct_array.shape
    ct_array = normalize_ct(ct_array)

    patch_size = (16, 16, 8)
    patches, coords = extract_patches(ct_array, patch_size=patch_size)
    if len(patches) == 0:
        print("❌ No valid patches extracted.")
        return

    patches_tensor = torch.from_numpy(patches).float().unsqueeze(1)
    patches_tensor = patches_tensor.view(1, len(patches), 1, *patch_size)

    with torch.no_grad():
        with autocast(device_type='cpu'):
            pred_patches = model(patches_tensor.to(device)) 
    
    pred_patches = pred_patches.squeeze(0).to(torch.float32).cpu().numpy()  # Convert bfloat16 to float32
    pred_patches = pred_patches[:, 0, :, :, :]  # [num_patches, D, H, W]

    brain_mask = reconstruct_volume(pred_patches, coords, downsampled_shape, patch_size=patch_size)
    brain_mask = (brain_mask > 0.5).astype(np.uint8)

    # Upsample to original size
    brain_mask = upsample_mask(brain_mask, original_size)

    save_nrrd(brain_mask, output_path, ct_file_path)

# Example usage
if __name__ == "__main__":
    # Specify your CT file path
    ct_file_path = r"C:\\Users\\rocia\\Downloads\\TFG\\Cohort\\Enhance_ctp_tests\\6_CTp.3D.nrrd"  
    output_mask_path = r"C:\\Users\\rocia\\Downloads\\TFG\\Cohort\\Enhance_ctp_tests\\6_brain_model_CTp.3D.nrrd" 
    
    # Generate and save the brain mask
    generate_brain_mask(ct_file_path, output_path=output_mask_path)