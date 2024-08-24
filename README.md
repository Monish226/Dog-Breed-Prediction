# 🐶 Dog Breed Prediction

Welcome to the **Dog Breed Prediction** project! This repository contains scripts, data, and models to classify dog breeds using machine learning techniques. The project leverages a dataset of dog images and breed labels to train a Machine learning model capable of accurately identifying the breed of a dog from an image.
<p align="center">
<img src ="https://github.com/Monish226/Dog-Breed-Prediction/blob/master/Dataset/asst.jpg" width="390" height="390" >
</p>

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

## 🤖 Model Performance
Our model is designed to achieve high accuracy in predicting dog breeds. Below is an example of the confusion matrix generated from the model’s predictions:


**Algorithm Comparison**
We evaluated five different machine learning algorithms:

Convolutional Neural Network (CNN): Achieved an impressive accuracy of 97.8150%, making it the top performer.
Support Vector Machine (SVM)
Decision Tree
Random Forest
K-Nearest Neighbors (KNN)
Among these, the CNN outperformed the others in terms of accuracy, demonstrating its effectiveness in handling complex image classification tasks.

## 📊 Results
<p align="center">
**Accuracy Details**
| Algorithm | Accuracy |
| ------ | ------ |
| Convolutional Neural Network (CNN) | 97.8150% |
| Support Vector Machine (SVM) |  79.6680 |
| Decision Tree | 78.3800% |
| Random Forest | 80.5990  |
| K-Nearest Neighbors (KNN) | 78.9230% |
</p>

## 📈 Future Work
Model Optimization: Continue to fine-tune the CNN model to push the accuracy even higher.
More Data: Incorporate additional dog breed images to enhance the model’s capabilities.
Deployment: Develop a web app that allows users to upload images and receive breed predictions in real-time.
## 📄 License
This project is licensed under the MIT License. See the LICENSE file for details.

## 👨‍💻 Contributing
We welcome contributions! Please fork this repository and submit a pull request with your changes.

Thank you for checking out this project! If you have any questions or suggestions, feel free to open an issue or reach out.


