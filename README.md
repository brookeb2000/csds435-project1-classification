# Project 1 – Classification

## Overview
This project performs handwritten digit classification using six machine learning models:

- Naive Bayes
- Random Forest
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- AdaBoost
- Multi-Layer Perceptron (MLP)

The models are trained on the training dataset and used to generate predictions for the testing dataset.  
Based on performance on the training data, **MLP was selected as the best model**.

---

## Files

The following files should be in the same folder:

- model_training_validation_and_testing.ipynb
- training.csv
- testing.csv

---

## Requirements

The notebook uses the following Python libraries:

- numpy
- pandas
- scikit-learn
- joblib
- matplotlib

If needed, install them with:

pip install numpy pandas scikit-learn joblib matplotlib

---

## How to Run

1. Place the following files in the same folder:

model_training_validation_and_testing.ipynb  
training.csv  
testing.csv  

2. Open the notebook:

model_training_validation_and_testing.ipynb

3. Run the notebook from top to bottom.

The notebook will:
- Train all six models
- Evaluate them on the training data
- Generate predictions for the testing dataset

---

## Output Files

Running the notebook will generate the following files:

predictions.csv  
naive_bayes_predictions.csv  
random_forest_predictions.csv  
svm_predictions.csv  
knn_predictions.csv  
adaboost_predictions.csv  

Each file contains the predicted labels for the testing dataset.

The file used for grading is:

predictions.csv
