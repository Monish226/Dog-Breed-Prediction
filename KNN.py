import os
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from skimage.io import imread
from skimage.transform import resize
from skimage.color import rgb2gray
from skimage.feature import hog

# Specify the main directory where the 8 subdirectories for dog breeds are located.
main_directory = 'dog-breeds'

# Define the input image size.
input_size = (128, 128)

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

# Adjust the number of neighbors for KNN
k_neighbors = 1

# Define a pipeline with PCA and KNN classifier
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=100)),  # You can adjust the number of components here
    ('knn', KNeighborsClassifier(n_neighbors=k_neighbors, metric='manhattan'))
])

# Perform cross-validation with 10 iterations
cv_scores = cross_val_score(pipeline, X, y, cv=10)

# Print the accuracy for each fold
for i, accuracy in enumerate(cv_scores):
    print(f"Iteration {i+1}: Accuracy = {accuracy * 100:.2f}%")

# Compute and print the mean accuracy across all folds
mean_accuracy = np.mean(cv_scores) * 100
print(f"Mean Accuracy: {mean_accuracy:.2f}%")
