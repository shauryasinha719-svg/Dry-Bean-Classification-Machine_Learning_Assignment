
# Dry Bean Classification

## 1. Problem Statement

The objective of this project is to develop and
compare five machine learning classification
algorithms to classify dry beans based on their
physical characteristics.

## 2. Dataset Description

Dataset: Dry Bean Classification

Target Variable: Class

Classification Type: Multiclass

Number of Classes: 7

The dataset contains numerical features
representing physical characteristics of dry beans.

## 3. Models Used

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbour
4. Gaussian Naive Bayes
5. Support Vector Machine

## 4. Evaluation Metrics

The models are evaluated using:

1. Accuracy
2. AUC Score
3. Precision
4. Recall
5. F1 Score
6. Matthews Correlation Coefficient

## 5. Model Comparison

Refer to model_comparison.csv for actual
evaluation results.

## 6. Observations

The five classification algorithms were evaluated using six performance metrics: Accuracy, AUC, Precision, Recall, F1 Score and Matthews Correlation Coefficient (MCC).

The following observations were derived from the experimental results:

Support Vector Machine (SVM): Achieved the highest overall performance with 92.21% accuracy and an AUC of 99.37%. It also recorded the highest Precision (92.24%), Recall (92.21%), F1 Score (92.22%) and MCC (0.9059).
Logistic Regression: Demonstrated performance comparable to SVM, achieving 92.07% accuracy and 99.34% AUC. This indicates that Logistic Regression is also highly effective for classifying dry bean varieties.
K-Nearest Neighbours (KNN): Achieved 91.66% accuracy and 98.12% AUC, demonstrating strong classification capabilities based on the similarity between observations.
Decision Tree: Recorded 89.17% accuracy and 93.30% AUC. Although its performance was lower than the first three algorithms, it provided reasonably good classification results.
Gaussian Naive Bayes: Achieved 76.39% accuracy and 96.44% AUC. It recorded the lowest accuracy among the five algorithms, indicating comparatively weaker classification performance.

Overall, SVM and Logistic Regression demonstrated the strongest classification performance, while Naive Bayes recorded the lowest accuracy.

## 7. Overall Winner

Support Vector Machine (SVM) was identified as the best-performing classification algorithm.

It achieved the highest values across all six evaluation metrics:

Evaluation Metric	SVM Performance
Accuracy	92.21%
AUC	99.37%
Precision	92.24%
Recall	92.21%
F1 Score	92.22%
MCC	0.9059

SVM marginally outperformed Logistic Regression, with an accuracy advantage of 0.14 percentage points.

Its consistently superior performance across all six metrics makes it the preferred model for the Dry Bean Classification problem.

## 8. Streamlit Application

The application allows users to upload a
labelled test dataset, select a classification
model and view evaluation results.

## 9. Repository Contents

- app.py
- requirements.txt
- README.md
- test_data.csv
- metadata.json
- model_comparison.csv
- model folder containing five saved models
