import cv2
import numpy as np
from scipy.spatial import distance
import pickle
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from time import time


def visualize_tsne(all_descriptors):
    """
    Perform t-SNE on SIFT descriptors and visualize the 2D projection.

    Parameters:
        all_descriptors: numpy array, shape (total_descriptors, 128)
            SIFT descriptors for all images combined.
    """
    # Perform PCA to reduce dimensionality first (to e.g., 50 components)
    print("Performing PCA...")
    pca = PCA(n_components=50)
    reduced_descriptors = pca.fit_transform(all_descriptors)

    print("Performing t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, n_iter=300, learning_rate=200)

    # Run t-SNE on reduced descriptors
    descriptors_2d = tsne.fit_transform(reduced_descriptors)

    # Visualize the t-SNE output
    plt.figure(figsize=(10, 8))
    plt.scatter(descriptors_2d[:, 0], descriptors_2d[:, 1], s=1, alpha=0.6)
    plt.title("t-SNE Visualization of SIFT Descriptors")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.show()


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
    all_descriptors = []  # List to hold all SIFT descriptors for t-SNE visualization

    # Create a SIFT detector
    sift = cv2.SIFT_create()

    start_time = time()
    print("Constructing bags of SIFTs...")

    for path in image_paths:
        # Read the image and convert it to grayscale
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        # Check if image is loaded correctly
        if img is None:
            print(f"Error: Could not load image {path}")
            continue

        # Detect and compute SIFT descriptors
        keypoints, descriptors = sift.detectAndCompute(img, None)

        if descriptors is not None:
            # Compute distances between descriptors and vocabulary
            dist = distance.cdist(vocab, descriptors, metric='euclidean')
            idx = np.argmin(dist, axis=0)

            # Build a histogram of visual word occurrences
            hist, _ = np.histogram(idx, bins=len(vocab), range=(0, len(vocab)))
            hist_norm = hist / np.sum(hist)  # Normalize the histogram

            image_feats.append(hist_norm)
            all_descriptors.append(descriptors)  # Collect descriptors for visualization

    image_feats = np.asarray(image_feats)

    # Combine all descriptors for t-SNE visualization
    all_descriptors = np.vstack(all_descriptors)

    # Perform t-SNE visualization
    visualize_tsne(all_descriptors)

    end_time = time()
    print(f"It takes {end_time - start_time:.2f} seconds to construct bags of SIFTs.")

    return image_feats
