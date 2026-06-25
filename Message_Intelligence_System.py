import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

from sklearn.impute import SimpleImputer

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# ==================================
# Load Dataset
# ==================================

df = pd.read_excel("Message_Intelligence_Dataset_5200.xlsx")

print("Dataset Shape :", df.shape)

# ==================================
# Check Missing Values
# ==================================

print("\nMissing Values:")
print(df.isnull().sum())

# ==================================
# Select Features
# ==================================

X = df.drop(
    columns=[
        "spam_label",
        "message_id",
        "message_text",
        "timestamp"
    ],
    errors="ignore"
)

y = df["spam_label"]

# ==================================
# Handle Missing Values
# ==================================

imputer = SimpleImputer(strategy="mean")

X = pd.DataFrame(
    imputer.fit_transform(X),
    columns=X.columns
)

# ==================================
# Train Test Split
# ==================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ==================================
# Scaling
# ==================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==================================
# KNN
# ==================================

print("\n========== KNN ==========")

for k in [3, 5, 7]:

    knn = KNeighborsClassifier(n_neighbors=k)

    knn.fit(X_train_scaled, y_train)

    pred = knn.predict(X_test_scaled)

    print(f"\nK = {k}")
    print("Accuracy :", accuracy_score(y_test, pred))

knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train_scaled, y_train)

y_pred_knn = knn.predict(X_test_scaled)

# ==================================
# SVM
# ==================================

print("\n========== SVM ==========")

svm = SVC(kernel="rbf")

svm.fit(X_train_scaled, y_train)

y_pred_svm = svm.predict(X_test_scaled)

# ==================================
# Naive Bayes
# ==================================

print("\n========== Naive Bayes ==========")

nb = GaussianNB()

nb.fit(X_train_scaled, y_train)

y_pred_nb = nb.predict(X_test_scaled)

# ==================================
# Evaluation Function
# ==================================

def evaluate(name, y_true, y_pred):

    print("\n" + "=" * 40)
    print(name)
    print("=" * 40)

    print("Accuracy :", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred))
    print("Recall   :", recall_score(y_true, y_pred))
    print("F1 Score :", f1_score(y_true, y_pred))

    print("\nConfusion Matrix")
    print(confusion_matrix(y_true, y_pred))

    print("\nClassification Report")
    print(classification_report(y_true, y_pred))

# ==================================
# Results
# ==================================

evaluate("KNN", y_test, y_pred_knn)

evaluate("SVM", y_test, y_pred_svm)

evaluate("Naive Bayes", y_test, y_pred_nb)

# ==================================
# Model Comparison
# ==================================

results = pd.DataFrame({
    "Model": ["KNN", "SVM", "Naive Bayes"],
    "Accuracy": [
        accuracy_score(y_test, y_pred_knn),
        accuracy_score(y_test, y_pred_svm),
        accuracy_score(y_test, y_pred_nb)
    ],
    "Precision": [
        precision_score(y_test, y_pred_knn),
        precision_score(y_test, y_pred_svm),
        precision_score(y_test, y_pred_nb)
    ],
    "Recall": [
        recall_score(y_test, y_pred_knn),
        recall_score(y_test, y_pred_svm),
        recall_score(y_test, y_pred_nb)
    ],
    "F1 Score": [
        f1_score(y_test, y_pred_knn),
        f1_score(y_test, y_pred_svm),
        f1_score(y_test, y_pred_nb)
    ]
})

print("\n========== Model Comparison ==========\n")
print(results)

best_model = results.loc[results["Accuracy"].idxmax()]

print("\nBest Model:")
print(best_model)