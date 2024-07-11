README.md
### Breast Cancer Prediction using Machine Learning
This repository contains the code and resources for a machine learning project focused on predicting breast cancer outcomes. The project utilizes various machine learning algorithms to classify tumors as benign or malignant based on features extracted from digitized images of breast mass.

# Table of Contents
Introduction
Problem Statement
Data Source & Description
Data Cleaning
Data Exploration
Feature Engineering
Feature Selection
Model Building
Results and Discussion
Confusion Matrix
Model Deployment
Conclusion
Future Scope
Introduction
Breast cancer is a significant health concern worldwide, ranking as the second most lethal cancer among women. Early detection plays a crucial role in improving treatment outcomes and reducing mortality rates. This project explores the application of machine learning algorithms to develop predictive models for early breast cancer detection.

Problem Statement
The project aims to build a predictive framework to classify breast tumors as benign or malignant using machine learning techniques. Traditional diagnostic methods are often invasive and time-consuming. Therefore, the goal is to create a non-invasive, accurate, and efficient tool for early breast cancer detection.

Data Source & Description
The dataset used for this project is the Wisconsin Diagnostic Breast Cancer (WDBC) dataset obtained from the UCI Machine Learning Repository. It consists of 569 instances with 32 features, including patient ID, diagnosis (Malignant or Benign), and 30 real-valued features computed from digitized images.

Dataset Source: Wisconsin Diagnostic Breast Cancer (WDBC)

Data Cleaning
The dataset was imported into Jupyter Notebook and checked for missing values. No missing values were found. The ID column was dropped as it was not useful for model building.

Data Exploration
Visualized the distribution of diagnoses using a count plot.
Plotted histograms for each feature to understand their distributions.
Feature Engineering
Encoded the diagnosis column (Malignant = 1, Benign = 0) using LabelEncoder.
Standardized features using StandardScaler to ensure zero mean and unit variance.
Feature Selection
Selected features based on their correlation with the diagnosis using a correlation matrix. Features with low correlation were excluded to reduce noise.

Model Building
Implemented several machine learning models:

Logistic Regression
Decision Tree Classifier
Random Forest Classifier
Gradient Boosting Classifier
Support Vector Classifier (SVC)
K-Nearest Neighbors (KNN) Classifier
Evaluated models based on metrics like accuracy, precision, recall, F1 score, and ROC AUC to identify the best performing model.

Results and Discussion
Logistic Regression emerged as the best model with an accuracy of 96.49%.
Detailed evaluation metrics and comparison with other models are provided in the report.
Confusion Matrix
The confusion matrix illustrates the performance of each model in terms of correctly and incorrectly predicted instances of benign and malignant cases.

Model Deployment
Deployed the Logistic Regression model using Streamlit, allowing real-time predictions of breast cancer diagnoses. Users can input relevant features to obtain predictions instantly.

Conclusion
This study demonstrates the effectiveness of machine learning in breast cancer prediction, particularly through Logistic Regression, which achieved high accuracy and robust performance across all metrics. The deployment of this model via Streamlit provides a practical tool for early diagnosis, enhancing patient outcomes and treatment strategies.

Future Scope
Future research could explore additional features and advanced techniques to further improve predictive accuracy and broaden the application of machine learning in oncology.

Contact Me Via
Email: Fridayamy2020@gmail.com
Friday Amarachi Promise
