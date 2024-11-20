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
def update_vital_signs_in_gui(heart_rate, breath_rate):
    heart_rate_var.set(f"Heart Rate: {heart_rate:.2f} bpm")
    breath_rate_var.set(f"Breath Rate: {breath_rate:.2f} breaths/min")

# ------------------------------
# Function to Process JSON File with Retries and Delay
# ------------------------------
def process_json_file(json_file_path, retries=3, delay=10):
    """
    Processes a JSON file to extract vitals and make predictions using the trained model.

    Args:
        json_file_path (str): Path to the JSON file.
        retries (int): Number of retry attempts if processing fails.
        delay (int): Delay in seconds between retry attempts.
    """
    for attempt in range(1, retries + 1):
        try:
            print(f"Attempt {attempt} - Processing file: {json_file_path}")
            
            # Wait for the specified delay to ensure file writing is complete
            time.sleep(delay)

            # Check if file is not empty
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
                    vitals_dict = {
                        "heartRate": vitals.get("heartRate"),
                        "breathRate": vitals.get("breathRate")
                    }

                    # Flatten waveform values into features
                    for i, value in enumerate(heart_waveform):
                        vitals_dict[f"heartWaveform_{i}"] = value
                    for i, value in enumerate(breath_waveform):
                        vitals_dict[f"breathWaveform_{i}"] = value

                    # Ensure all required features are present
                    for col in model.feature_names_in_:
                        if col not in vitals_dict:
                            vitals_dict[col] = 0.0  # Fill missing features with zero or appropriate value

                    # Create DataFrame with the necessary columns for the model
                    df = pd.DataFrame([vitals_dict])
                    df = df[model.feature_names_in_]

                    # Make prediction
                    predictions = model.predict(df)

                    # Get breath rate value, handle None
                    breath_rate_value = vitals.get("breathRate", 0.0)

                    # Put the results in the queue
                    gui_queue.put((predictions[0], breath_rate_value))

            # If processing is successful, exit the loop
            print(f"Successfully processed {json_file_path}")
            break  # Exit the retry loop

        except (JSONDecodeError, ValueError) as e:
            print(f"Attempt {attempt} - Error processing {json_file_path}: {e}")
            if attempt < retries:
                print(f"Retrying in {delay} seconds...")
            else:
                print(f"Failed to process {json_file_path} after {retries} attempts.")
        except Exception as e:
            print(f"Unexpected error processing {json_file_path}: {e}")
            print(f"Skipping file: {json_file_path}")
            break  # Exit on unexpected errors

# ------------------------------
# Event Handler for New Files
# ------------------------------
class NewFileHandler(FileSystemEventHandler):
    def __init__(self):
        super().__init__()
        self.processed_files = set()
        self.lock = threading.Lock()

    def on_created(self, event):
        # Process only files, not directories
        if not event.is_directory and event.src_path.endswith('.json'):
            with self.lock:
                if event.src_path not in self.processed_files:
                    self.processed_files.add(event.src_path)
                    print(f"Detected new JSON file: {event.src_path}")
                    # Start a new thread to process the file to avoid blocking
                    threading.Thread(target=process_json_file, args=(event.src_path,), daemon=True).start()

# ------------------------------
# Function to Start the Watchdog Observer
# ------------------------------
def start_observer():
    # Path to your binData directory
    live_data_directory = r'C:\ti\radar_toolbox_2_20_00_05\tools\visualizers\Applications_Visualizer\Industrial_Visualizer\binData'
    if not os.path.exists(live_data_directory):
        print(f"Error: Directory '{live_data_directory}' does not exist.")
        return

    event_handler = NewFileHandler()
    observer = Observer()
    # Set recursive=True to monitor all subdirectories
    observer.schedule(event_handler, live_data_directory, recursive=True)
    observer.start()
    print(f"Started monitoring directory: {live_data_directory}")
    try:
        while True:
            time.sleep(1)  # Keep the thread running
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

# ------------------------------
# Function to Periodically Check the Queue and Update the GUI
# ------------------------------
def process_gui_queue():
    try:
        while True:
            heart_rate, breath_rate = gui_queue.get_nowait()
            update_vital_signs_in_gui(heart_rate, breath_rate)
    except queue.Empty:
        pass
    root.after(100, process_gui_queue)  # Check the queue every 100 ms

# ------------------------------
# GUI Setup
# ------------------------------
root = tk.Tk()
root.title("Vital Sign Monitor")

# Initialize StringVars with default values
heart_rate_var = tk.StringVar(value="Heart Rate: -- bpm")
breath_rate_var = tk.StringVar(value="Breath Rate: -- breaths/min")

# Create labels for heart rate and breath rate
heart_rate_label = tk.Label(root, textvariable=heart_rate_var, font=("Helvetica", 16))
heart_rate_label.pack(pady=10)

breath_rate_label = tk.Label(root, textvariable=breath_rate_var, font=("Helvetica", 16))
breath_rate_label.pack(pady=10)

# Start the file watching in a separate thread
observer_thread = threading.Thread(target=start_observer, daemon=True)
observer_thread.start()

# Start processing the GUI queue
root.after(100, process_gui_queue)

# Start the GUI event loop
root.mainloop()