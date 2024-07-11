# Breast Cancer Prediction Project

## Overview
This project focuses on predicting breast cancer outcomes using various machine learning algorithms to improve early detection and reduce mortality rates. The study evaluates models such as Decision Tree, Logistic Regression, Random Forest, Gradient Boosting, Support Vector Machine (SVM), and K-Nearest Neighbors (KNN) to predict breast cancer with high accuracy. Logistic Regression achieved the highest performance among these models.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Data Cleaning](#data-cleaning)
- [Data Exploration](#data-exploration)
- [Feature Engineering](#feature-engineering)
- [Model Building](#model-building)
- [Results and Discussion](#results-and-discussion)
- [Confusion Matrix](#confusion-matrix)
- [Model Deployment](#model-deployment)
- [Conclusion](#conclusion)
- [How to Run](#how-to-run)
- [Dependencies](#dependencies)

## Introduction
Breast cancer is one of the most common cancers among women worldwide, with early detection being crucial for effective treatment. This project aims to develop a machine learning-based predictive framework to accurately classify breast cancer tumors as benign or malignant. By evaluating various algorithms, we aim to identify the most effective model for early breast cancer detection, thus improving patient outcomes and enabling personalized treatment strategies.

## Dataset
The dataset used in this project is the Wisconsin Diagnostic Breast Cancer (WDBC) dataset, available at [UCI Machine Learning Repository](http://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+%28diagnostic%29). It contains 569 instances with 32 attributes, including the diagnosis (M for malignant and B for benign) and 30 real-valued features computed for each cell nucleus.

## Data Cleaning
- Checked for null values (none found).
- Dropped the ID number column as it was not useful for model building.

## Data Exploration
- Visualized the distribution of the diagnosis using a count plot.
- Plotted histograms for each feature to understand their distributions.

## Feature Engineering
- Encoded the diagnosis column to numerical values (M = 1, B = 0) using LabelEncoder.
- Standardized features using StandardScaler to ensure they have a mean of 0 and a standard deviation of 1.
- Performed feature selection by analyzing the correlation of each feature with the diagnosis and dropping features with low correlation.

## Model Building
Developed and evaluated several machine learning models, including:
- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- Gradient Boosting Classifier
- Support Vector Classifier (SVC)
- K-Nearest Neighbors (KNN) Classifier

## Results and Discussion
| Model               | Accuracy | Precision | Recall   | F1 Score | ROC AUC  |
|---------------------|----------|-----------|----------|----------|----------|
| Logistic Regression | 0.964912 | 0.953488  | 0.953488 | 0.953488 | 0.962660 |
| Decision Tree       | 0.938596 | 0.950000  | 0.883721 | 0.915663 | 0.927776 |
| Random Forest       | 0.956140 | 0.952381  | 0.930233 | 0.941176 | 0.951032 |
| Gradient Boosting   | 0.956140 | 0.952381  | 0.930233 | 0.941176 | 0.951032 |
| Support Vector Machine | 0.956140 | 0.975000  | 0.906977 | 0.939759 | 0.946446 |
| K-Nearest Neighbors | 0.964912 | 0.975610  | 0.930233 | 0.952381 | 0.958074 |

Logistic Regression emerged as the best model, achieving the highest accuracy and demonstrating robust performance across all metrics.

## Confusion Matrix
The confusion matrix for each model displays the instances of correct and incorrect predictions, highlighting the number of true positives, true negatives, false positives, and false negatives.

## Model Deployment
The Logistic Regression model was deployed using Streamlit, an open-source app framework. This deployment enables users to input relevant features and obtain real-time predictions on whether a breast cancer diagnosis is benign or malignant.

## Conclusion
This study utilized various machine learning algorithms to develop predictive models for breast cancer diagnosis. Logistic Regression emerged as the optimal model, achieving the highest accuracy of 96.49%. The deployment through Streamlit provides a practical tool for real-time breast cancer diagnosis, enhancing early detection efforts and improving patient outcomes.

## How to Run
1. Clone the repository:
   
   git clone https://github.com/Amy247/Aptech-final-project.git
   
2. Navigate to the project directory:
   
   cd breast-cancer-prediction
   
3. Install the required dependencies:
   
   pip install -r requirements.txt
   
4. Run the Streamlit app:
   
   streamlit run app.py
   

## Dependencies
- Python 3.8+
- NumPy
- pandas
- scikit-learn
- matplotlib
- joblib
- Streamlit

Feel free to explore the code, experiment with different models, and contribute to the project!

---

This README provides an overview of the project, detailing the key aspects, how to set up and run the project, and the dependencies required. Adjust the repository URL and other details as needed to match your specific project setup.
