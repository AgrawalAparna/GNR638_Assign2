from PIL import Image
import numpy as np

def get_tiny_images(image_paths, size=72):
    """
    Resize images to `size x size`, normalize them, and flatten them.
    
    Input:
        image_paths: List of image file paths.
        size: Target size (default is 72x72).
    
    Output:
        tiny_images: (N, size*size*channels) NumPy array of vectorized images.
    """
    tiny_images = []
    
    for path in image_paths:
        image = Image.open(path).convert("RGB")  # Ensure 3 channels (RGB)
        image = image.resize((size, size))  # Resize to 72x72
        image = np.array(image, dtype=np.float32)  # Convert to NumPy array
        image = (image - np.mean(image)) / (np.std(image) + 1e-8)  # Normalize
        image = image.flatten()  # Flatten to a vector
        tiny_images.append(image)
        
    return np.array(tiny_images)  # Convert list to NumPy array

