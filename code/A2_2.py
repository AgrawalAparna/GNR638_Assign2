from __future__ import print_function
import argparse
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

from get_image_paths import get_image_paths
from get_tiny_images import get_tiny_images

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Step 0: Set up parameters, category list, and image paths.
parser = argparse.ArgumentParser()
parser.add_argument('--feature', help='feature', type=str, default='bag_of_sift')
parser.add_argument('--classifier', help='classifier', type=str, default='mlp')
args = parser.parse_args()

DATA_PATH = r"C:\Users\Aparna Agrawal\Desktop\GNR638\Assign1\Assign2\v2-Scene-Recognition-with-Bag-of-Words-master\Scene-Recognition-with-Bag-of-Words-master\data\Images"


CATEGORIES = ['agricultural', 'airplane', 'baseballdiamond', 'beach', 'buildings', 'chaparral',
              'denseresidential', 'forest', 'freeway', 'golfcourse', 'harbor', 'intersection',
              'mediumresidential', 'mobilehomepark', 'overpass', 'parkinglot', 'river', 'runway',
              'sparseresidential', 'storagetanks', 'tenniscourt']

CATE2ID = {v: k for k, v in enumerate(CATEGORIES)}

FEATURE = 'tiny_image'
CLASSIFIER = 'mlp'

NUM_TRAIN_PER_CAT = 80

def plot_confusion_matrix(y_true, y_pred, labels, title='Confusion Matrix', normalize=True):
    """
    This function plots a confusion matrix.

    Parameters:
        y_true: The true labels (ground truth).
        y_pred: The predicted labels.
        labels: List of class labels (ordered).
        title: Title of the plot.
        normalize: Whether to normalize the matrix to show proportions.
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    if normalize:
        # Normalize the matrix by the total number of instances (sum of all elements)
        cm = cm.astype('float') / 20

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', xticklabels=labels, yticklabels=labels)

    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()


class MLPClassifier:
    def __init__(self, input_size, hidden1, hidden2, hidden3, num_classes, 
                 activation='relu', learning_rate=0.01, max_iter=2000): #, penalty='l2'):
        self.lr = learning_rate
        self.max_iter = max_iter
        self.activation = activation
        #self.penalty = penalty
        
        # Xavier initialization for weights
        self.W1 = np.random.randn(input_size, hidden1) * np.sqrt(2. / input_size)
        self.b1 = np.zeros((1, hidden1))
        self.W2 = np.random.randn(hidden1, hidden2) * np.sqrt(2. / hidden1)
        self.b2 = np.zeros((1, hidden2))
        self.W3 = np.random.randn(hidden2, hidden3) * np.sqrt(2. / hidden2)
        self.b3 = np.zeros((1, hidden3))
        self.W4 = np.random.randn(hidden3, num_classes) * np.sqrt(2. / hidden3)
        self.b4 = np.zeros((1, num_classes))

    def leaky_relu(self, Z, alpha=0.01):
        return np.where(Z > 0, Z, alpha * Z)

    def sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))

    def tanh(self, Z):
        return np.tanh(Z)

    def activation_function(self, Z):
        if self.activation == 'relu':
            return np.maximum(0, Z)
        elif self.activation == 'leaky_relu':
            return self.leaky_relu(Z)
        elif self.activation == 'sigmoid':
            return self.sigmoid(Z)
        elif self.activation == 'tanh':
            return self.tanh(Z)
        else:
            raise ValueError(f"Unknown activation function {self.activation}")

    def activation_derivative(self, Z):
        if self.activation == 'relu':
            return Z > 0
        elif self.activation == 'leaky_relu':
            return (Z > 0) + 0.01 * (Z <= 0)  # Derivative for leaky ReLU
        elif self.activation == 'sigmoid':
            sig = self.sigmoid(Z)
            return sig * (1 - sig)
        elif self.activation == 'tanh':
            return 1 - np.tanh(Z) ** 2
        else:
            raise ValueError(f"Unknown activation function {self.activation}")

    def softmax(self, Z):
        expZ = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return expZ / expZ.sum(axis=1, keepdims=True)

    def forward(self, X):
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = self.activation_function(self.Z1)
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = self.activation_function(self.Z2)
        self.Z3 = np.dot(self.A2, self.W3) + self.b3
        self.A3 = self.activation_function(self.Z3)
        self.Z4 = np.dot(self.A3, self.W4) + self.b4
        self.A4 = self.softmax(self.Z4)
        return self.A4

    def backward(self, X, Y, output):
        m = X.shape[0]
        dZ4 = output - Y
        dW4 = np.dot(self.A3.T, dZ4) / m
        db4 = np.sum(dZ4, axis=0, keepdims=True) / m
        dZ3 = np.dot(dZ4, self.W4.T) * self.activation_derivative(self.Z3)
        dW3 = np.dot(self.A2.T, dZ3) / m
        db3 = np.sum(dZ3, axis=0, keepdims=True) / m
        dZ2 = np.dot(dZ3, self.W3.T) * self.activation_derivative(self.Z2)
        dW2 = np.dot(self.A1.T, dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m
        dZ1 = np.dot(dZ2, self.W2.T) * self.activation_derivative(self.Z1)
        dW1 = np.dot(X.T, dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m

        """# L2 regularization on weights
        if self.penalty == 'l2':
            dW1 += 0.01 * self.W1
            dW2 += 0.01 * self.W2
            dW3 += 0.01 * self.W3
            dW4 += 0.01 * self.W4
        """
        # Update weights
        self.W4 -= self.lr * dW4
        self.b4 -= self.lr * db4
        self.W3 -= self.lr * dW3
        self.b3 -= self.lr * db3
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

    def fit(self, X, Y):
        for epoch in range(self.max_iter):
            output = self.forward(X)
            self.backward(X, Y, output)
            if epoch % 100 == 0:
                loss = -np.mean(np.sum(Y * np.log(output), axis=1))
                print(f'Epoch {epoch}/{self.max_iter}, Loss: {loss:.4f}')

    def predict(self, X):
        output = self.forward(X)
        return np.argmax(output, axis=1)

def one_hot_encode(labels, num_classes):
    one_hot = np.zeros((labels.size, num_classes))
    one_hot[np.arange(labels.size), labels] = 1
    return one_hot

def train_mlp(train_features, train_labels, test_features, test_labels):
    label_encoder = LabelEncoder()
    train_labels_encoded = label_encoder.fit_transform(train_labels)
    test_labels_encoded = label_encoder.transform(test_labels)

    num_classes = len(label_encoder.classes_)
    input_size = train_features.shape[1]
    hidden1 = 128
    hidden2 = 64
    hidden3 = 32

    train_labels_one_hot = one_hot_encode(train_labels_encoded, num_classes)

    # Hyperparameters to test
    learning_rates = [0.1, 0.08, 0.03, 0.01]
    max_iters = [500, 1000]

    results = []

    # Loop through different hyperparameters and train the model
    for lr in learning_rates:
        for max_iter in max_iters:
            model = MLPClassifier(input_size, hidden1, hidden2, hidden3, num_classes,  
                                  activation='relu', 
                                  learning_rate = lr, 
                                  max_iter=max_iter)
            
            model.fit(train_features, train_labels_one_hot)

            # Evaluate the model on the test set
            test_predictions = model.predict(test_features)
            accuracy = accuracy_score(test_labels_encoded, test_predictions)
            
            # Store the result
            results.append((lr, max_iter, accuracy))

    # Print results
    print_results(results, learning_rates, max_iters)

    # Return predictions and true labels in original form
    final_predictions = model.predict(test_features)
    return final_predictions, label_encoder.inverse_transform(final_predictions)

def print_results(results, learning_rates, max_iters):
    print(f"{'Learning Rate':<15} {'Max Iterations':<15} {'Test Accuracy':<15}")
    print("="*45)

    for result in results:
        lr, max_iter, accuracy = result
        print(f"{lr:<15} {max_iter:<15} {accuracy * 100:.2f}%")
    

def main():
    print("Getting paths and labels for all train and test data")
    train_image_paths, test_image_paths, train_labels, test_labels = \
        get_image_paths(DATA_PATH, CATEGORIES, NUM_TRAIN_PER_CAT)
    
    if FEATURE == 'tiny_image':
        train_image_feats = get_tiny_images(train_image_paths)
        test_image_feats = get_tiny_images(test_image_paths)
    
    if CLASSIFIER == 'mlp':
        encoded_test_predictions, test_predictions = train_mlp(train_image_feats, train_labels, test_image_feats, test_labels)
        
        #plot_confusion_matrix(test_labels, test_predictions, CATEGORIES)

if __name__ == '__main__':
    main()