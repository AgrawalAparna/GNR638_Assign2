from sklearn.model_selection import KFold
import numpy as np
from sklearn.metrics import accuracy_score

from nearest_neighbor_classify import nearest_neighbor_classify
from svm_classify import svm_classify

def k_fold_validation(train_image_feats, train_labels, classifier_type='nearest_neighbor', k=5):
    """
    Perform k-fold cross-validation.

    Parameters:
    - train_image_feats: Features for all training images.
    - train_labels: Labels for all training images.
    - classifier_type: Either 'nearest_neighbor' or 'support_vector_machine'.
    - k: Number of folds.

    Returns:
    - Average classification accuracy across all folds.
    """
    kf = KFold(n_splits=k, shuffle=True, random_state=42)  # Initialize k-fold cross-validation
    accuracies = []  # To store accuracies for each fold

    fold_idx = 1
    for train_idx, val_idx in kf.split(train_image_feats):
        print(f"Processing Fold {fold_idx}...")
        fold_idx += 1

        # Split the data into training and validation sets
        train_feats = train_image_feats[train_idx]
        train_labels_fold = np.array(train_labels)[train_idx]
        val_feats = train_image_feats[val_idx]
        val_labels = np.array(train_labels)[val_idx]

        # Train and predict using the selected classifier
        if classifier_type == 'nearest_neighbor':
            predicted_categories = nearest_neighbor_classify(train_feats, train_labels_fold, val_feats)
        elif classifier_type == 'support_vector_machine':
            predicted_categories = svm_classify(train_feats, train_labels_fold, val_feats)
        elif classifier_type == 'dumy_classifier':
            # The dummy classifier simply predicts a random category for
            # every test case
            predicted_categories = test_labels[:]
            shuffle(predicted_categories)
        else:
            raise NameError('Unknown classifier type')

        # Evaluate accuracy for this fold
        accuracy = accuracy_score(val_labels, predicted_categories)
        print(f"Accuracy for Fold {fold_idx - 1}: {accuracy}")
        accuracies.append(accuracy)

    # Calculate average accuracy across all folds
    avg_accuracy = np.mean(accuracies)
    print(f"Average Accuracy across {k} folds: {avg_accuracy}")

    return avg_accuracy

# Example usage:
# avg_accuracy = k_fold_validation(train_image_feats, train_labels, classifier_type='support_vector_machine', k=5)
