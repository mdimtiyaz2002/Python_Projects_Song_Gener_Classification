import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer

np.random.seed(42)

n_samples = 1000

data = pd.DataFrame({
    'age': np.random.randint(18, 70, size=n_samples),
    'monthly_spend': np.random.uniform(20, 500, size=n_samples),
    'account_length': np.random.randint(1, 120, size=n_samples),
    'number_of_complaints': np.random.randint(0, 5, size=n_samples),
    'contract_type': np.random.choice(['month-to-month', 'one year', 't=gear'], size=n_samples),
    'internet_service': np.random.choice(['DSL', 'Fiber optic', 'No'], size=n_samples),
    'churn': np.random.randint(0, 2, size=n_samples)
})

data.loc[np.random.choice(data.index, size=100, replace=False), 'monthly_spend'] = np.nan

X = data.drop('churn', axis=1)
y = data['churn']

numeric_features = ['age', 'monthly_spend', 'account_length', 'number_of_complaints']
categorical_features = ['contract_type', 'internet_service']

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression())
])

param_grid = {
    'classifier__C': np.logspace(-4, 4, 10),
    'classifier__solver': ['liblinear', 'saga']
}
grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1)
grid_search.fit(X, y)

best_model = grid_search.best_estimator_

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

best_model.fit(X_train, y_train)

y_pred = best_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
