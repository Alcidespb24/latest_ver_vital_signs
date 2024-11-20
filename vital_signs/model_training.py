import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pickle

# Load the processed data
data = pd.read_csv('combined_vitals_data.csv')
data = data.dropna() 

# Prepare your feature matrix X and target vector y
# Assume 'heartRate' is the target variable, modify if necessary
X = data.drop(columns=['heartRate', 'breathRate'])  # Features
y = data['heartRate'].copy()  # Target variable

# Handle missing values
y.fillna(y.mean(), inplace=True)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Save the model to a file
with open('vitals_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

print("Model training complete. Model saved as vitals_model.pkl.")
