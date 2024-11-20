import os
import json
import pandas as pd

# Directory containing all your folders with JSON files
data_directory = r'C:\ti\radar_toolbox_2_20_00_05\tools\visualizers\Applications_Visualizer\Industrial_Visualizer\binData'

# List to store the vitals data
all_vitals_data = []

# Walk through the directory recursively
for root, dirs, files in os.walk(data_directory):
    for file_name in files:
        if file_name.endswith('.json'):
            file_path = os.path.join(root, file_name)
            with open(file_path) as file:
                data = json.load(file)
                for entry in data.get("data", []):
                    frame_data = entry.get("frameData", {})
                    vitals = frame_data.get("vitals", None)
                    if vitals:
                        # Extract heartRate and breathRate directly
                        heart_rate = vitals.get("heartRate")
                        breath_rate = vitals.get("breathRate")

                        # Flatten heartWaveform and breathWaveform into separate features
                        heart_waveform = vitals.get("heartWaveform", [])
                        breath_waveform = vitals.get("breathWaveform", [])

                        # Prepare a dictionary to hold the current row of data
                        vitals_dict = {
                            "heartRate": heart_rate,
                            "breathRate": breath_rate
                        }

                        # Add individual waveform values as separate columns
                        for i, value in enumerate(heart_waveform):
                            vitals_dict[f"heartWaveform_{i}"] = value
                        for i, value in enumerate(breath_waveform):
                            vitals_dict[f"breathWaveform_{i}"] = value

                        all_vitals_data.append(vitals_dict)

# Load Apple Watch resting heart rate data
apple_watch_directory = 'vital_signs\\Apple_Watch'
for file_name in os.listdir(apple_watch_directory):
    if file_name.endswith('.csv'):
        apple_watch_path = os.path.join(apple_watch_directory, file_name)
        apple_data = pd.read_csv(apple_watch_path)
        
        # Process the Apple Watch data
        for index, row in apple_data.iterrows():
            vitals_dict = {
                "heartRate": row['Value'],  # Assuming the heart rate value is in the 'Value' column
                "breathRate": None  # Placeholder for breath rate if not available
            }
            # Optionally add dummy waveform data if needed
            for i in range(10):  # Assuming you want 10 dummy waveform points
                vitals_dict[f"heartWaveform_{i}"] = 0.0  # Placeholder values
                vitals_dict[f"breathWaveform_{i}"] = 0.0  # Placeholder values
            
            all_vitals_data.append(vitals_dict)

# Convert the vitals data into a Pandas DataFrame
combined_df = pd.DataFrame(all_vitals_data)

# Handle missing values explicitly
combined_df['heartRate'] = combined_df['heartRate'].fillna(combined_df['heartRate'].mean())
combined_df['breathRate'] = combined_df['breathRate'].fillna(combined_df['breathRate'].mean())

# Save the preprocessed data to a CSV file for easier analysis
combined_df.to_csv('combined_vitals_data.csv', index=False)
print("Data preprocessing complete. Saved to combined_vitals_data.csv.")