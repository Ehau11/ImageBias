import os
import cv2
import tensorflow as tf
import numpy as np
from scipy.ndimage import gaussian_filter
from concurrent.futures import ThreadPoolExecutor

def extract_color_cues(image):
    """Convert RGB image to HSV and extract V channel and HS channels."""
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    v_channel = hsv_image[:, :, 2]  # Value channel
    hs_channels = hsv_image[:, :, :2]  # Hue and Saturation channels
    return v_channel, hs_channels

def extract_texture_cues(image, segmentation_mask):
    """Extract texture patches based on the segmentation mask."""
    smoothed = gaussian_filter(image, sigma=1)
    patches = []
    unique_classes = np.unique(segmentation_mask)
    for cls in unique_classes:
        cls_mask = (segmentation_mask == cls).astype(np.uint8)
        cls_texture = cv2.bitwise_and(smoothed, smoothed, mask=cls_mask)
        patches.append(cls_texture)
    return patches

def extract_shape_cues(image):
    """Extract shape cues using Canny edge detection."""
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return cv2.Canny(gray_image, threshold1=50, threshold2=150)

def process_image(image_file, input_folder, color_output_dir, shape_output_dir, texture_output_dir):
    """Process a single image file to extract color, shape, and texture cues."""
    image_path = os.path.join(input_folder, image_file)
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load {image_file}. Skipping.")
        return

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Extract Color cues
    v_channel, hs_channels = extract_color_cues(image)
    
    # Save Color cues
    base_name = os.path.splitext(image_file)[0]
    cv2.imwrite(os.path.join(color_output_dir, f"{base_name}_color_v.png"), v_channel)
    hs_image = cv2.cvtColor(hs_channels, cv2.COLOR_HSV2BGR)
    cv2.imwrite(os.path.join(color_output_dir, f"{base_name}_color_hs.png"), hs_image)

    # Extract Shape cues
    shape_edges = extract_shape_cues(image)
    cv2.imwrite(os.path.join(shape_output_dir, f"{base_name}_shape.png"), shape_edges)

    # Simulated segmentation mask (replace with actual segmentation output)
    segmentation_mask = np.random.randint(0, 5, size=image.shape[:2], dtype=np.uint8)  # Random classes
    texture_patches = extract_texture_cues(image, segmentation_mask)

    # Save Texture cues
    for idx, patch in enumerate(texture_patches):
        cv2.imwrite(os.path.join(texture_output_dir, f"{base_name}_texture_{idx}.png"), patch)

def decompose_images_from_folder(input_folder, output_dir):
    """Decompose images from the input folder into color, shape, and texture cues."""
    # Create separate folders for each cue inside the output directory
    color_output_dir = os.path.join(output_dir, "color")
    shape_output_dir = os.path.join(output_dir, "shape")
    texture_output_dir = os.path.join(output_dir, "texture")
    os.makedirs(color_output_dir, exist_ok=True)
    os.makedirs(shape_output_dir, exist_ok=True)
    os.makedirs(texture_output_dir, exist_ok=True)
    
    # Get list of all image files in the folder
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]

    # Process images in parallel
    with ThreadPoolExecutor() as executor:
        for image_file in image_files:
            executor.submit(process_image, image_file, input_folder, color_output_dir, shape_output_dir, texture_output_dir)

    print(f"Decomposition complete. Results saved to {output_dir}")

# Example usage
if __name__ == "__main__":
    input_folder = "/Users/ethan/Documents/python/datasets/images"  # Folder containing input images
    output_dir = "/Users/ethan/Documents/python/datasets/output"     # Directory to save results
    decompose_images_from_folder(input_folder, output_dir)