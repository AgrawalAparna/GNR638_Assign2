# Scene Recognition with Bag of Words and Tiny Images

This project implements a scene recognition system for the UC Merced dataset. The system classifies images into various scene categories using a three-layer Multi-Layer Perceptron (MLP). Additionally, it evaluates the impact of activation functions and overfitting behavior.

## Dataset

The UC Merced Land Use Dataset:
- Contains 21 scene categories.
- Each category has 100 images of size 256x256 pixels.
- Dataset Link: [UC Merced Dataset](http://weegee.vision.ucmerced.edu/datasets/landuse.html)

## Methodology

### Data Splitting
- **70% Training Data**: Used for feature extraction and vocabulary building.
- **10% Validation Data**: Used for deciding the optimal number of codewords.
- **20% Testing Data**: Used for evaluating the final model's performance.

### Features and Classifiers
1. **Bag of SIFT Features + Three-Layer MLP**
2. **Tiny Images (Downscaled to 72x72) + Three-Layer MLP**

### Steps
1. **Feature Extraction**:
   - Extract SIFT descriptors from images (Bag of SIFT approach).
   - Downscale images to 72x72 and flatten them into vectors (Tiny Images approach).
2. **Vocabulary Creation**:
   - Use k-means clustering on SIFT descriptors to create a visual vocabulary.
3. **Feature Representation**:
   - Represent images as histograms of visual words (Bag of SIFT).
   - Linearize downscaled images to use as input features.
4. **MLP Training**:
   - Experiment with different activation functions (ReLU, Tanh, Sigmoid).
   - Tune the number of hidden layer neurons.
5. **Evaluation**:
   - Compute accuracy for both approaches.
   - Generate a confusion matrix.

## Results

All the results can be found in the `Results` folder of the repository.

- Used **Tanh** and **Sigmoid**, but accuracy decreased compared to ReLU.

### Tiny Images Implementation
- Overfitting observed: Training loss is low, but test accuracy is significantly lower.

### Bag of SIFTs Implementation
- Achieved better results as overfitting was avoided.

## Usage

### Prerequisites
- Python 3.8+
- Required libraries: `numpy`, `scikit-learn`, `matplotlib`, `tensorflow`, `pickle`

### Steps to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/agaparna2468/GNR638_Assign2.git
   ```
2. Place the UC Merced dataset in the `data/Images` directory.
3. Run the main script:
   ```bash
   python A2_1.py #For part1
   ```
   ```bash
   python A2_2.py  #For part2
   ```

## Contributors
- Aparna Agrawal
- Shravani Kode
