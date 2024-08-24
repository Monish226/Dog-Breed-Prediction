# 🐶 Dog Breed Prediction

Welcome to the **Dog Breed Prediction** project! This repository contains scripts, data, and models to classify dog breeds using machine learning techniques. The project leverages a dataset of dog images and breed labels to train a Machine learning model capable of accurately identifying the breed of a dog from an image.

![IMG_20240824_194111_596 width=50px](https://github.com/user-attachments/assets/43630503-ed85-4f98-86b4-cbceea4b2162)


---

## 🚀 Project Overview

This project walks through the entire process of building a Machine learning model to classify dog breeds from images. The key steps include data preparation, model building, training, and evaluation.

---

## 🛠️ Steps to Build the Project

1. **Load the Dataset**  
   Download the dataset from Kaggle.  
   **IMPORTANT**: Download the dataset from [here](https://www.dropbox.com/scl/fi/07ot4h9zzhzc6f2ugcvkf/archive.zip?rlkey=8fundwycq0vo2v4a9ervt93gc&dl=0).  
   **Size**: 750MB

2. **Load Labels**  
   Load the breed labels from the CSV file, which contains the image ID and corresponding breed.

3. **Check Breed Count**  
   Analyze the distribution of breeds in the dataset to understand the data balance.

4. **One-Hot Encoding on Labels**  
   Apply one-hot encoding to the labels to convert breed names into a format suitable for model training.

5. **Load and Preprocess Images**  
   Load the images, convert them into arrays, and normalize them to ensure consistency in model input.

6. **Check Data Shape and Size**  
   Verify the shape and size of the input data (`X`) and labels (`Y`) to ensure they are correctly formatted.

7. **Build the Model Architecture**  
   Design the Convolutional neural network architecture using popular Machine learning frameworks.

8. **Train the Model**  
   Split the data into training and validation sets, and fit the model to the data. Track accuracy and loss metrics during training.

9. **Evaluate the Model**  
   Assess the model's performance on the validation set by calculating the accuracy score.

10. **Predict Using the Model**  
    Use the trained model to predict the breed of new dog images and evaluate its performance on unseen data.

---

## 🔧 Prerequisites

- **Python 2.7**  
  Make sure Python 2.7 is installed on your system.

- **Dependencies**  
  Install the required Python libraries.

