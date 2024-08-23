import os
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from skimage.io import imread
from skimage.transform import resize
from skimage.color import rgb2gray
from skimage.feature import hog

# Specify the main directory where the 8 subdirectories for dog breeds are located.
main_directory = 'dog-breeds'

# Define the input image size.
input_size = (128, 128)

# Define the number of neighbors for SVM (you can adjust this).
svm_C = 5.0

# Load and preprocess the dataset, including feature extraction
def load_dataset(dataset_dir):
    X = []  # Features
    y = []  # Corresponding labels (breeds)

    breed_labels = os.listdir(dataset_dir)

    for label, breed in enumerate(breed_labels):
        breed_dir = os.path.join(dataset_dir, breed)
        for image_file in os.listdir(breed_dir):
            image_path = os.path.join(breed_dir, image_file)
            image = imread(image_path)
            image = resize(image, input_size)

            # Extract HOG features
            gray_image = rgb2gray(image)
            features = hog(gray_image)

            X.append(features)
            y.append(label)

    return np.array(X), np.array(y)

X, y = load_dataset(main_directory)

# Create and train the SVM classifier
svm_classifier = SVC(C=svm_C)

# Perform cross-validation to get multiple accuracy values.
cv_scores = cross_val_score(svm_classifier, X, y, cv=10)  # You can adjust the number of folds (e.g., cv=5)

# Print the accuracy for each fold
for i, accuracy in enumerate(cv_scores):
    print(f"Accuracy for Fold {i+1}: {accuracy * 100:.2f}%")

# Compute and print the mean accuracy across all folds
mean_accuracy = np.mean(cv_scores) * 100
print(f"Mean Accuracy: {mean_accuracy:.2f}%")
