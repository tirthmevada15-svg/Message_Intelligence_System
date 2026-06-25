Message Intelligence System: Spam Message Classification
Project Overview

This project develops a Message Intelligence System that automatically classifies incoming digital messages as:

0 → Legitimate Message
1 → Spam Message

The system uses Machine Learning classification algorithms and probability concepts to analyze message-related features and predict whether a message is spam or legitimate.

Objective

The objective of this project is to build and compare multiple classification models using:

K-Nearest Neighbors (KNN)
Support Vector Machine (SVM)
Naive Bayes

The performance of these models is evaluated using standard classification metrics to determine the most suitable model for spam message detection.

Dataset Information

The dataset contains numerical features extracted from message content and sender behavior.

Features
message_length
word_count
num_urls
num_digits
num_special_chars
spam_keyword_score
legit_keyword_score
sender_activity_score
sender_account_age_days
messages_sent_last_24h
hour_of_day
day_of_week
Target Variable
spam_label
0 = Legitimate Message
1 = Spam Message
Dataset Size
Total Records: 5200
Total Columns: 16
Technologies Used
Python
Pandas
NumPy
Scikit-Learn
Jupyter Notebook / VS Code
Machine Learning Models
1. K-Nearest Neighbors (KNN)

KNN classifies messages based on similarity with neighboring data points using distance calculations.

2. Support Vector Machine (SVM)

SVM creates an optimal decision boundary that separates spam and legitimate messages with maximum margin.

3. Naive Bayes

Naive Bayes uses Bayes' Theorem and probability calculations to classify messages.

Data Preprocessing

The following preprocessing steps were performed:

Loaded dataset using Pandas.
Selected relevant features.
Handled missing values using SimpleImputer.
Split data into training and testing datasets.
Applied StandardScaler for feature scaling.
Evaluation Metrics

The models were evaluated using:

Accuracy
Precision
Recall
F1 Score
Confusion Matrix
Project Workflow
Step 1: Data Loading

Load the Message Intelligence Dataset into Python.

Step 2: Data Preprocessing
Remove unwanted columns
Handle missing values
Scale numerical features
Step 3: Train-Test Split

Split dataset into:

Training Set (80%)
Testing Set (20%)
Step 4: Model Training

Train:

KNN Classifier
SVM Classifier
Naive Bayes Classifier
Step 5: Model Evaluation

Calculate:

Accuracy
Precision
Recall
F1 Score
Step 6: Model Comparison

Compare the performance of all three classifiers and identify the best-performing model.

Results Summary
Model	Accuracy	Precision	Recall	F1 Score
KNN	Generated after execution	Generated after execution	Generated after execution	Generated after execution
SVM	Generated after execution	Generated after execution	Generated after execution	Generated after execution
Naive Bayes	Generated after execution	Generated after execution	Generated after execution	Generated after execution
Conclusion

This project successfully implemented and compared three classification algorithms for spam message detection.

Key observations:

KNN performs classification using nearest neighbors.
Naive Bayes utilizes probability theory and Bayes' Theorem.
SVM provides strong classification performance by maximizing class separation.

Based on evaluation metrics, the best model can be selected for deployment in a real-world message filtering system.
