import os
import numpy as np
from sklearn.model_selection import KFold
from skimage.io import imread
from skimage.transform import resize
import tensorflow as tf
from tensorflow.keras import layers, models

# Specify the main directory where the 8 subdirectories for dog breeds are located.
main_directory = 'dog-breeds'

# Define the input image size.
input_size = (128, 128)

# Load and preprocess the dataset
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

            X.append(image)
            y.append(label)

    return np.array(X), np.array(y)

X, y = load_dataset(main_directory)

# Define the CNN architecture
def create_cnn_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(8, activation='softmax')  # Assuming 8 dog breeds
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model

# Custom wrapper function for cross-validation
def cnn_cv_score(model, X, y, cv):
    scores = []
    for train_index, test_index in cv.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)
        _, accuracy = model.evaluate(X_test, y_test, verbose=0)
        scores.append(accuracy)

    return np.array(scores)

# Perform cross-validation to get multiple accuracy values
cv = KFold(n_splits=10, shuffle=True, random_state=42)
cv_scores = cnn_cv_score(create_cnn_model(), X, y, cv)

# Print the accuracy for each fold
for accuracy in cv_scores:
    print(f"{accuracy * 100:.2f}")

# Compute and print the mean accuracy across all folds
print(f"Accuracy for Fold :{np.mean(cv_scores) * 100:.2f}")
