import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import glob
import random
from dsnn import build_dsnn

def load_and_preprocess(image_path):
    """Loads and preps a single image exactly how we trained it."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read {image_path}")
    
    # Keep original for plotting later
    original_img = img.copy()
    
    # Resize and filter
    img = cv2.resize(img, (256, 256))
    img_filtered = cv2.medianBlur(img, 3)
    
    # Min-Max Scaling
    img_scaled = img_filtered.astype(np.float32)
    img_scaled = (img_scaled - np.min(img_scaled)) / (np.max(img_scaled) - np.min(img_scaled) + 1e-8)
    
    # Expand dims for the model: (1, 256, 256, 1)
    input_tensor = np.expand_dims(np.expand_dims(img_scaled, axis=-1), axis=0)
    
    return original_img, input_tensor

def show_results(original, enhanced):
    """Plots the before and after side-by-side."""
    plt.figure(figsize=(10, 5))
    
    # Plot Original
    plt.subplot(1, 2, 1)
    plt.title("Original Low-Quality Image")
    plt.imshow(original, cmap='gray')
    plt.axis('off')
    
    # Plot Enhanced
    plt.subplot(1, 2, 2)
    plt.title("NC-FODE + DSNN Enhanced")
    plt.imshow(enhanced, cmap='gray')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("Loading optimized model...")
    # Rebuild architecture and load your trained weights
    model = build_dsnn()
    model.load_weights("nc_fode_dsnn_weights.h5")
    
    # Grab a random image from your dataset folder
    # Update this path if your images are somewhere else!
    dataset_folder = "/Users/anubhavkishoreanand/mncproject/data/train/*.jpg" 
    images = glob.glob(dataset_folder)
    
    if len(images) == 0:
        print("No images found to test!")
    else:
        test_image_path = random.choice(images)
        print(f"Testing on: {test_image_path}")
        
       # Preprocess
        original, input_tensor = load_and_preprocess(test_image_path)
        
        # Grab the original dimensions (Height, Width)
        orig_h, orig_w = original.shape
        
        # Run Inference
        pred_fuse, _ = model.predict(input_tensor)
        
        # Extract the 2D image array
        enhanced_img = pred_fuse[0, :, :, 0]
        
        # Apply Sigmoid to convert raw logits into proper [0, 1] pixel values
        enhanced_img = tf.nn.sigmoid(enhanced_img).numpy()
        
        # Scale back to standard 0-255 image format
        enhanced_img = (enhanced_img * 255).astype(np.uint8)
        
        # THE FIX: Stretch the 256x256 image back to the original panoramic dimensions
        enhanced_img = cv2.resize(enhanced_img, (orig_w, orig_h))
        
        # Display!
        show_results(original, enhanced_img)