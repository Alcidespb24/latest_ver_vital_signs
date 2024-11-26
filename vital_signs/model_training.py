# model_trainning.py

import pandas as pd
import pickle
from sklearn.model_selection import RepeatedKFold, GridSearchCV
from sklearn.ensemble import RandomForestRegressor

# ------------------------------
# Load and Prepare the Data
# ------------------------------

# Load the preprocessed data
data = pd.read_csv('preprocessed_vitals_data.csv')

# Prepare your feature matrix X and target vector y
# Assume 'heartRate' is the target variable
X = data.drop(columns=['heartRate'])  # Features
y = data['heartRate'].copy()          # Target variable

# Ensure X and y have consistent indices
X.reset_index(drop=True, inplace=True)
y.reset_index(drop=True, inplace=True)

# ------------------------------
# Define Cross-Validation Strategy
# ------------------------------

# Define Repeated K-Fold cross-validation
cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)

# ------------------------------
# Hyperparameter Optimization
# ------------------------------

# Define the parameter grid to search
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10],
    'min_samples_split': [2, 5],
    'max_features': ['auto', 'sqrt'],
}

# Initialize the model
rf = RandomForestRegressor(random_state=42)

# Initialize GridSearchCV
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=cv,
    scoring='neg_mean_absolute_error',
    n_jobs=-1,
    verbose=2
)

# ------------------------------
# Train the Model with Cross-Validation
# ------------------------------

# Fit the GridSearch to find the best parameters
grid_search.fit(X, y)

# Get the best model
best_model = grid_search.best_estimator_

print(f"Best parameters found: {grid_search.best_params_}")

# ------------------------------
# Save the Best Model
# ------------------------------

# Save the model to a file
with open('vitals_model.pkl', 'wb') as model_file:
    pickle.dump(best_model, model_file)

print("\nModel training complete. Best model saved as vitals_model.pkl.")