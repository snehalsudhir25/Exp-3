# Experiment 3: Logistic Regression for Classification

## Aim

To compute descriptive statistics and probabilities using NumPy and
Pandas.

## Algorithm

1.  **Start.**

2.  Load the dataset and extract features `X` and target `y`.

3.  Split the data into training and test sets (e.g., 80:20).

4.  Standardize features using `StandardScaler`.

5.  Train `LogisticRegression` on the training data.

    The logistic (sigmoid) function is:

    \[ `\sigma`{=tex}(z) = `\frac{1}{1 + e^{-z}}`{=tex} \]

    where

    \[ z = `\beta`{=tex}\_0 + `\beta`{=tex}\_1 x_1 + `\cdots `{=tex}+
    `\beta`{=tex}\_p x_p \]

6.  Predict labels on the test data.

7.  Evaluate performance using accuracy, confusion matrix, and
    classification report.

8.  Print the evaluation metrics.

9.  **Stop.**




## Program

    import numpy as np
    import pandas as pd
    
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
    
    # Load dataset
    data = load_breast_cancer()
    
    X = data.data
    y = data.target
    
    # Split dataset into training and testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Standardize the features
    scaler = StandardScaler()
    
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Train Logistic Regression model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    
    # Predict test data
    y_pred = model.predict(X_test)
    
    # Evaluate model
    accuracy = accuracy_score(y_test, y_pred)
    
    cm = confusion_matrix(y_test, y_pred)
    
    print("Accuracy:", accuracy)
    
    print("\nConfusion Matrix:")
    print(cm)
    
    print("\nClassification Report:")
    print(classification_report(
        y_test, y_pred, target_names=data.target_names
    ))
    
 ## output
    Accuracy: 0.9824561403508771
    
    Confusion Matrix:
    [[41  1]
     [ 1 71]]
    
    Classification Report:
    
                  precision  recall  f1-score  support
    
    malignant        0.98     0.98     0.98       42
    benign           0.99     0.99     0.99       72
    
    accuracy                           0.98      114
    macro avg        0.98     0.98     0.98      114
    weighted avg     0.99     0.99     0.99      114
    
    
 ## Result
    Thus, Logistic Regression was successfully implemented using Python. The dataset was divided into training and testing sets, standardized, and evaluated using accuracy, confusion matrix and classification report. The model achieved 98.25% accuracy.
