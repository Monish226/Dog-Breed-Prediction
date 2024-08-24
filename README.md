# 🐶 Dog Breed Prediction

Welcome to the **Dog Breed Prediction** project! This repository contains scripts, data, and models to classify dog breeds using `machine learning` techniques. The project leverages a dataset of `dog images` and breed labels to train a Machine learning model capable of accurately identifying the breed of a dog from an image.
<p align="center">
<img src ="https://github.com/Monish226/Dog-Breed-Prediction/blob/master/Dataset/asst.jpg" width="400" height="400" >
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
   **Images**:10000

3. **Load Labels**  
   Load the breed labels from the CSV file, which contains the image ID and corresponding breed.

4. **Check Breed Count**  
   Analyze the distribution of breeds in the dataset to understand the data balance.

5. **One-Hot Encoding on Labels**  
   Apply one-hot encoding to the labels to convert breed names into a format suitable for model training.

6. **Load and Preprocess Images**  
   Load the images, convert them into arrays, and normalize them to ensure consistency in model input.

7. **Check Data Shape and Size**  
   Verify the shape and size of the input data (`X`) and labels (`Y`) to ensure they are correctly formatted.

8. **Build the Model Architecture**  
   Design the Convolutional neural network architecture using popular Machine learning frameworks.

9. **Train the Model**  
   Split the data into training and validation sets, and fit the model to the data. Track accuracy and loss metrics during training.

10. **Evaluate the Model**  
   Assess the model's performance on the validation set by calculating the accuracy score.

11. **Predict Using the Model**  
    Use the trained model to predict the breed of new dog images and evaluate its performance on unseen data.

---

## 🔧 Prerequisites

- **Python 2.7**  
  Make sure `Python 2.7` is installed on your system.
- **Dependencies**  
  Install the required Python libraries.


**Algorithm Comparison**
We evaluated five different 🤖 machine learning algorithms:

 `Convolutional Neural Network (CNN)`: Achieved an impressive accuracy of `97.8150%`, making it the top performer.
 `Support Vector Machine (SVM)`
 `Decision Tree`
 `Random Forest`
 `K-Nearest Neighbors (KNN)`
 Among these, the CNN outperformed the others in terms of accuracy, demonstrating its effectiveness in handling complex image classification tasks.

## 📊 Results
**Accuracy Details**
         
| Algorithm | Accuracy |
| ------ | ------ |
| Convolutional Neural Network (CNN) | 97.8150% |
| Support Vector Machine (SVM) |  79.6680% |
| Decision Tree | 78.3800% |
| Random Forest | 80.5990%  |
| K-Nearest Neighbors (KNN) | 78.9230% |


## 📈 Future Work
Model Optimization: Continue to fine-tune the CNN model to push the accuracy even higher.
More Data: Incorporate additional dog breed images to enhance the model’s capabilities.
Deployment: Develop a web app that allows users to upload images and receive breed predictions in real-time.
## 📄 License

This project is licensed under the MIT License. 

### MIT License Summary

The MIT License is a permissive free software license originating at the Massachusetts Institute of Technology (MIT). It is a simple and easy-to-understand license that places very few restrictions on reuse, making it a popular choice for open-source projects.

Key Points:
- **Freedom to Use**: You can use the software for any purpose.
- **Freedom to Modify**: You can modify the software and use it as a base for other projects.
- **Freedom to Distribute**: You can distribute the original or modified software to others.
- **No Warranty**: The software is provided "as is", without warranty of any kind.

For the full text of the MIT License, please see the `LICENSE` file in this repository.

## 👨‍💻 Contributing
We welcome contributions! Please fork this repository and submit a pull request with your changes.

<p align="center">
<img src="https://github.com/Monish226/Dog-Breed-Prediction/blob/master/Dataset/title.jpg" width="400" height="400">
</p>

Thank you for checking out this project! If you have any questions or suggestions, feel free to open an issue or reach out.
## 🐕✨ Happy Coding!!.

