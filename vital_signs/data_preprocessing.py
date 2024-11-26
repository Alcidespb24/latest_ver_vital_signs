# data_preprocessing.py

import pandas as pd

# ------------------------------
# Load and Preprocess the Data
# ------------------------------

# Load the combined vitals data
data = pd.read_csv('combined_vitals_data.csv')

# Handle missing values
data = data.dropna()

# Optionally, perform any additional preprocessing steps here
# For example, normalize or scale features if necessary

# Save the preprocessed data to a new CSV file
data.to_csv('preprocessed_vitals_data.csv', index=False)
print("Data preprocessing complete. Saved to preprocessed_vitals_data.csv.")