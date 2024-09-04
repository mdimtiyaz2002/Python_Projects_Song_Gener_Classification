# Python_Projects_Song_Gener_Classification(Logistic Regression Model with Data Preprocessing and Hyperparameter Tuning)

# This is my 3rd project using python......
## Logistic Regression Model with Data Preprocessing and Hyperparameter Tuning
This Python script uses logistic regression to predict customer churn based on a mock dataset. It includes data preprocessing steps and hyperparameter tuning using `GridSearchCV`.

### Features
- Data Handling: Utilizes `pandas` to create and manage the dataset.
- Data Preprocessing: Handles missing values, scales numerical features, and encodes categorical features using `scikit-learn` pipelines.
- Model Training: Uses `scikit-learn` to split the data into training and testing sets, and to train a logistic regression model.
- Hyperparameter Tuning: Employs `GridSearchCV` to find the best hyperparameters for the logistic regression model.
- Evaluation Metrics: Calculates and prints accuracy, classification report, and the confusion matrix to evaluate the model's performance.

### Usage
1. Ensure you have the required libraries installed: `pandas`, `numpy`, and `scikit-learn`.
2. Run the script to preprocess the data, train the logistic regression model, and perform hyperparameter tuning.
3. The script will output the model's performance metrics, including accuracy, classification report, and the confusion matrix.

### Example
Accuracy: 0.78
Classification Report:
              precision    recall  f1-score   support

           0       0.80      0.75      0.77       100
           1       0.76      0.81      0.78       100

    accuracy                           0.78       200
   macro avg       0.78      0.78      0.78       200
weighted avg       0.78      0.78      0.78       200

Confusion Matrix:
[[75 25]
 [19 81]]

### Dependencies
- `pandas`
- `numpy`
- `scikit-learn`

### How It Works
1. Data Preparation: A mock dataset is created with various customer attributes and a churn label.
2. Data Preprocessing: Missing values in numerical features are imputed with the median, and categorical features are imputed with the most frequent value. Numerical features are scaled, and categorical features are one-hot encoded.
3. Model Training: The dataset is split into training and testing sets. A logistic regression model is trained on the training set.
4. Hyperparameter Tuning: `GridSearchCV` is used to find the best hyperparameters for the logistic regression model.
5. Model Evaluation: The model's predictions on the test set are evaluated using various metrics.
