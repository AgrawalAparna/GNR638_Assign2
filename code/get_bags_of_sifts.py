import cv2
import numpy as np
from scipy.spatial import distance
import pickle
import matplotlib.pyplot as plt
from time import time


def get_bags_of_sifts(image_paths):
    """
    Create bags of SIFT features for given images.

    Parameters:
        image_paths: List of image file paths.

    Returns:
        image_feats: Feature matrix of shape (N, d), where N is the number of images
                     and d is the vocabulary size (number of clusters).
    """
    # Load the visual word vocabulary
    with open('vocab.pkl', 'rb') as handle:
        vocab = pickle.load(handle)

    image_feats = []
    
    # Create a SIFT detector
    sift = cv2.SIFT_create()

    start_time = time()
    print("Constructing bags of SIFTs...")

    for path in image_paths:
        # Read the image in RGB format (convert from BGR)
        img = cv2.imread(path, cv2.IMREAD_COLOR)  # Read the image in color (BGR format)
        
        if img is None:
            print(f"Error: Could not load image {path}")
            continue
        
        # Convert the BGR image to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Detect and compute SIFT descriptors
        keypoints, descriptors = sift.detectAndCompute(img_rgb, None)

        if descriptors is None:
            #print(f"No keypoints detected for image {path}, using zero histogram.")
            hist = np.zeros(len(vocab))  # Create a zeroed histogram
            image_feats.append(hist)     # Append the zero histogram
        else:
            # Compute distances between descriptors and vocabulary
            dist = distance.cdist(vocab, descriptors, metric='euclidean')
            idx = np.argmin(dist, axis=0)

            # Build a histogram of visual word occurrences
            hist, _ = np.histogram(idx, bins=len(vocab), range=(0, len(vocab)))
            hist_norm = hist / np.sum(hist)  # Normalize the histogram

            image_feats.append(hist_norm)
            
    image_feats = np.asarray(image_feats)

    end_time = time()
    print(f"It takes {end_time - start_time:.2f} seconds to construct bags of SIFTs.")

    return image_feats
