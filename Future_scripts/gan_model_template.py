import os
import sys
import numpy as np
import SimpleITK as sitk
import tensorflow as tf
from tensorflow.keras.models import load_model
from skimage import transform
from skimage import exposure

def preprocess_image(image):
    """
    Preprocess the image: resize, normalize, etc.
    Adjust these steps depending on the GAN model's requirements.
    """
    # Resize the image if necessary (for example, to a 256x256 size)
    target_size = (256, 256)  # Example size, adapt to your model
    resized_image = transform.resize(image, target_size, mode='reflect', anti_aliasing=True)

    # Normalize the image to [0, 1]
    resized_image = resized_image / np.max(resized_image)

    return resized_image

def postprocess_image(image):
    """
    Postprocess the image: resize back and convert to original scale.
    """
    # Rescale the image back to original intensity range (assuming 0 to 255 for the output)
    image = exposure.rescale_intensity(image, in_range=(0, 1), out_range=(0, 255))
    return image

def load_input_image(input_path):
    """
    Load the input NRRD file.
    """
    # Load the image using SimpleITK (supports NRRD format)
    itk_image = sitk.ReadImage(input_path)
    image = sitk.GetArrayFromImage(itk_image)  # Convert to numpy array
    return image, itk_image

def save_enhanced_image(output_path, image, original_itk_image):
    """
    Save the enhanced image as an NRRD file.
    """
    enhanced_itk_image = sitk.GetImageFromArray(image)
    enhanced_itk_image.CopyInformation(original_itk_image)  # Copy metadata (spacing, origin, etc.)
    sitk.WriteImage(enhanced_itk_image, output_path)

def apply_gan_model(input_path, output_path, gan_model_path):
    """
    Apply the GAN model to the input volume.
    """
    # Load the input image
    image, original_itk_image = load_input_image(input_path)

    # Preprocess the image
    preprocessed_image = preprocess_image(image)

    # Expand dimensions to fit the model's input shape (e.g., [batch_size, height, width, channels])
    preprocessed_image = np.expand_dims(preprocessed_image, axis=-1)  # Add channel dimension
    preprocessed_image = np.expand_dims(preprocessed_image, axis=0)  # Add batch dimension

    # Load the pre-trained GAN model
    gan_model = load_model(gan_model_path)

    # Run the image through the GAN model
    enhanced_image = gan_model.predict(preprocessed_image)

    # Postprocess the enhanced image
    enhanced_image = enhanced_image.squeeze()  # Remove batch and channel dimensions
    enhanced_image = postprocess_image(enhanced_image)

    # Save the enhanced image
    save_enhanced_image(output_path, enhanced_image, original_itk_image)
    print(f"Enhanced image saved to {output_path}")

def main():
    """
    Main function to handle arguments and apply GAN model.
    """
    # Parse command-line arguments
    if len(sys.argv) < 4:
        print("Usage: apply_gan_model.py --input <input_path> --output <output_path> --model <gan_model_path>")
        sys.exit(1)

    input_path = sys.argv[sys.argv.index('--input') + 1]
    output_path = sys.argv[sys.argv.index('--output') + 1]
    gan_model_path = sys.argv[sys.argv.index('--model') + 1]

    # Apply the GAN model
    apply_gan_model(input_path, output_path, gan_model_path)

if __name__ == '__main__':
    main()
