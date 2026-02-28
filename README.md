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

The following files should be placed in the same folder:

model_training_validation_and_testing.ipynb  
training.csv  
testing.csv   

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

1. Place the files listed above in the same folder.

2. Open the notebook:

model_training_validation_and_testing.ipynb

3. Run the notebook from top to bottom.

The notebook will:
- Train all six models
- Save the trained models as `.joblib` files
- Generate predictions for the testing dataset

---

## Output Files

Running the notebook will generate the following files:

Trained models:
- best_naive_bayes_model.joblib
- best_random_forest_model.joblib
- best_svm_model.joblib
- best_knn_model_no_scaling.joblib
- best_adaboost_model.joblib
- best_mlp_model.joblib

Prediction files:
- predictions.csv (best model – MLP)
- naive_bayes_predictions.csv
- random_forest_predictions.csv
- svm_predictions.csv
- knn_predictions.csv
- adaboost_predictions.csv

Each prediction file contains the predicted labels for the testing dataset.

The file used for grading is:

predictions.csv
