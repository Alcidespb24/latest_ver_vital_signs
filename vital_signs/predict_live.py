# predict_live.py

import os
import json
import pandas as pd
import pickle
import tkinter as tk
import threading
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import queue
from json import JSONDecodeError

# ------------------------------
# Load the Trained Model
# ------------------------------
try:
    with open('vitals_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    print("Model loaded successfully.")
except FileNotFoundError:
    print("Error: 'vitals_model.pkl' not found. Please ensure the model file is in the script directory.")
    exit(1)
except Exception as e:
    print(f"Unexpected error loading model: {e}")
    exit(1)

# ------------------------------
# Create a Queue for Thread-Safe GUI Updates
# ------------------------------
gui_queue = queue.Queue()

# ------------------------------
# GUI Update Function
# ------------------------------
def update_vital_signs_in_gui(predicted_heart_rate, visualizer_heart_rate, breath_rate):
    predicted_heart_rate_var.set(f"Predicted Heart Rate: {predicted_heart_rate:.2f} bpm")
    visualizer_heart_rate_var.set(f"Visualizer Heart Rate: {visualizer_heart_rate:.2f} bpm")
    breath_rate_var.set(f"Breath Rate: {breath_rate:.2f} breaths/min")
    difference = abs(predicted_heart_rate - visualizer_heart_rate)
    difference_var.set(f"Difference: {difference:.2f} bpm")

# ------------------------------
# Function to Process JSON File with Retries and Delay
# ------------------------------
def process_json_file(json_file_path, retries=3, delay=10):
    for attempt in range(1, retries + 1):
        try:
            print(f"Attempt {attempt} - Processing file: {json_file_path}")
            time.sleep(delay)  # Ensure the file write operation is complete

            if os.path.getsize(json_file_path) == 0:
                raise ValueError("File is empty.")

            with open(json_file_path, 'r') as file:
                data = json.load(file)

            # Process the JSON data
            for entry in data.get("data", []):
                frame_data = entry.get("frameData", {})
                vitals = frame_data.get("vitals", None)
                if vitals:
                    heart_waveform = vitals.get("heartWaveform", [])
                    breath_waveform = vitals.get("breathWaveform", [])

                    # Prepare a dictionary for prediction
                    vitals_dict = {}

                    # Flatten waveform values into features
                    for i, value in enumerate(heart_waveform):
                        vitals_dict[f"heartWaveform_{i}"] = value
                    for i, value in enumerate(breath_waveform):
                        vitals_dict[f"breathWaveform_{i}"] = value

                    # Ensure all required features are present
                    for col in model.feature_names_in_:
                        if col not in vitals_dict:
                            vitals_dict[col] = 0.0

                    # Create DataFrame with the necessary columns for the model
                    df = pd.DataFrame([vitals_dict])
                    df = df[model.feature_names_in_]

                    # Make prediction
                    predicted_heart_rate = model.predict(df)[0]

                    # Get visualizer's own heart rate reading
                    visualizer_heart_rate = vitals.get("heartRate", 0.0)

                    # Get breath rate value
                    breath_rate_value = vitals.get("breathRate", 0.0)

                    # Put the results in the queue
                    gui_queue.put((predicted_heart_rate, visualizer_heart_rate, breath_rate_value))

            print(f"Successfully processed {json_file_path}")
            break

        except (JSONDecodeError, ValueError) as e:
            print(f"Attempt {attempt} - Error processing {json_file_path}: {e}")
            if attempt < retries:
                print(f"Retrying in {delay} seconds...")
            else:
                print(f"Failed to process {json_file_path} after {retries} attempts.")
        except Exception as e:
            print(f"Unexpected error processing {json_file_path}: {e}")
            print(f"Skipping file: {json_file_path}")
            break

# ------------------------------
# Event Handler for New Files
# ------------------------------
class NewFileHandler(FileSystemEventHandler):
    def __init__(self):
        super().__init__()
        self.processed_files = set()
        self.lock = threading.Lock()

    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith('.json'):
            with self.lock:
                if event.src_path not in self.processed_files:
                    self.processed_files.add(event.src_path)
                    print(f"Detected new JSON file: {event.src_path}")
                    threading.Thread(target=process_json_file, args=(event.src_path,), daemon=True).start()

# ------------------------------
# Function to Start the Watchdog Observer
# ------------------------------
def start_observer():
    live_data_directory = r'C:\ti\radar_toolbox_2_20_00_05\tools\visualizers\Applications_Visualizer\Industrial_Visualizer\binData'
    if not os.path.exists(live_data_directory):
        print(f"Error: Directory '{live_data_directory}' does not exist.")
        return

    event_handler = NewFileHandler()
    observer = Observer()
    observer.schedule(event_handler, live_data_directory, recursive=True)
    observer.start()
    print(f"Started monitoring directory: {live_data_directory}")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

# ------------------------------
# Function to Periodically Check the Queue and Update the GUI
# ------------------------------
def process_gui_queue():
    try:
        while True:
            predicted_heart_rate, visualizer_heart_rate, breath_rate = gui_queue.get_nowait()
            update_vital_signs_in_gui(predicted_heart_rate, visualizer_heart_rate, breath_rate)
    except queue.Empty:
        pass
    root.after(100, process_gui_queue)

# ------------------------------
# GUI Setup
# ------------------------------
root = tk.Tk()
root.title("Vital Sign Monitor")

# Initialize StringVars with default values
predicted_heart_rate_var = tk.StringVar(value="Predicted Heart Rate: -- bpm")
visualizer_heart_rate_var = tk.StringVar(value="Visualizer Heart Rate: -- bpm")
breath_rate_var = tk.StringVar(value="Breath Rate: -- breaths/min")
difference_var = tk.StringVar(value="Difference: -- bpm")

# Create labels
predicted_label = tk.Label(root, textvariable=predicted_heart_rate_var, font=("Helvetica", 16))
predicted_label.pack(pady=5)

visualizer_label = tk.Label(root, textvariable=visualizer_heart_rate_var, font=("Helvetica", 16))
visualizer_label.pack(pady=5)

difference_label = tk.Label(root, textvariable=difference_var, font=("Helvetica", 16))
difference_label.pack(pady=5)

breath_label = tk.Label(root, textvariable=breath_rate_var, font=("Helvetica", 16))
breath_label.pack(pady=5)

# Start the file watching in a separate thread
observer_thread = threading.Thread(target=start_observer, daemon=True)
observer_thread.start()

# Start processing the GUI queue
root.after(100, process_gui_queue)

# Start the GUI event loop
root.mainloop()