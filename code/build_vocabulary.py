import cv2
import numpy as np
from time import time

def build_vocabulary(image_paths, vocab_size):
    """
    This function samples SIFT descriptors from the training images,
    clusters them with kmeans, and returns the cluster centers.

    Args:
        image_paths: A list of training image paths.
        vocab_size: Number of clusters desired.

    Returns:
        vocab: Cluster centers from kmeans (vocabulary).
    """
    bag_of_features = []

    print("Extract SIFT features")

    # Create a SIFT detector
    sift = cv2.SIFT_create()

    for path in image_paths:
        # Read image using cv2 and convert to grayscale
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        
        # Check if image is loaded correctly
        if img is None:
            print(f"Error: Could not load image {path}")
            continue

        # Detect and compute SIFT descriptors
        keypoints, descriptors = sift.detectAndCompute(img, None)
        
        # Add descriptors to the bag of features if available
        if descriptors is not None:
            bag_of_features.append(descriptors)

    # Concatenate all descriptors into a single numpy array
    bag_of_features = np.concatenate(bag_of_features, axis=0).astype('float32')

    print("Compute vocab")
    start_time = time()

    # Apply k-means clustering using OpenCV
    # Criteria: (type, max_iter, epsilon)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.1)
    _, labels, vocab = cv2.kmeans(
        data=bag_of_features,
        K=vocab_size,
        bestLabels=None,
        criteria=criteria,
        attempts=10,
        flags=cv2.KMEANS_PP_CENTERS,
    )

    end_time = time()
    print(f"It takes {end_time - start_time:.2f} seconds to compute vocab.")

    return vocab
