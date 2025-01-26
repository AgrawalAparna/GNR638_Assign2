import os
from glob import glob

def get_image_paths(data_path, categories, num_train_per_cat, num_val_per_cat):
    num_categories = len(categories)

    train_image_paths = []
    test_image_paths = []

    val_image_paths = []

    val_labels = []

    train_labels = []
    test_labels = []

    for category in categories:

        image_paths = glob(os.path.join(data_path, category, '*.tif'))
    
        for i in range(num_train_per_cat):
            train_image_paths.append(image_paths[i])
            train_labels.append(category)

        image_paths = glob(os.path.join(data_path, category, '*.tif'))

        for i in range(num_val_per_cat):
            val_image_paths.append(image_paths[num_train_per_cat + i])
            val_labels.append(category)

        image_paths = glob(os.path.join(data_path, category, '*.tif'))
        for i in range(40-num_train_per_cat-num_val_per_cat):
            test_image_paths.append(image_paths[num_train_per_cat + num_val_per_cat+ i])
            test_labels.append(category)

    return train_image_paths, test_image_paths, val_image_paths, train_labels, test_labels, val_labels
