The Live Vital Sign Monitoring Application serves as a real-time system to monitor and display vital signs such as heart rate and breath rate. The application performs the following key functions:

Directory Monitoring: Continuously watches a specified directory (and its subdirectories) for new JSON files containing vital sign data using the watchdog library.

JSON File Processing: Upon detection of a new JSON file, the application waits for a specified delay to ensure the file is fully written. It then attempts to parse and process the file, extracting relevant features required by the pre-trained machine learning model.

Prediction: Utilizes a pre-trained Random Forest model (vitals_model.pkl) to predict vital signs based on the extracted features.

GUI Display: Updates a user-friendly Tkinter-based GUI in real-time to display the latest heart rate and breath rate predictions.

Robust Error Handling: Implements a retry mechanism to handle scenarios where JSON files might be incomplete or malformed, ensuring that each file is processed up to three times before logging a failure.

Detailed Explanation
Let's delve into the various components of the script to understand how each part contributes to the overall functionality.

1. Imports and Dependencies
python
Copy code
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
Standard Libraries:
os: Interacts with the operating system, handling file paths and sizes.
json: Parses JSON files.
pickle: Loads the pre-trained machine learning model.
tkinter: Creates the GUI for real-time display.
threading: Manages concurrent execution to ensure the GUI remains responsive.
time: Implements delays between retries.
queue: Facilitates thread-safe communication between processing threads and the GUI.
Third-Party Libraries:
pandas: Structures data for model predictions.
watchdog: Monitors the filesystem for new JSON files.
Ensure all third-party libraries are installed using pip:

bash
Copy code
pip install watchdog pandas scikit-learn
2. Loading the Pre-Trained Model
python
Copy code
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
Purpose: Loads the pre-trained Random Forest model (vitals_model.pkl) necessary for making predictions.
Error Handling:
FileNotFoundError: Alerts if the model file is missing and exits the script.
Generic Exception: Catches any other unexpected errors during model loading and exits gracefully.
3. Creating a Thread-Safe Queue
python
Copy code
# ------------------------------
# Create a Queue for Thread-Safe GUI Updates
# ------------------------------
gui_queue = queue.Queue()
Purpose: Establishes a queue to safely pass prediction results from background processing threads to the main GUI thread, ensuring thread safety.
4. GUI Update Function
python
Copy code
# ------------------------------
# GUI Update Function
# ------------------------------
def update_vital_signs_in_gui(heart_rate, breath_rate):
    """
    Updates the GUI labels with the latest heart rate and breath rate values.
    
    Args:
        heart_rate (float): Predicted heart rate.
        breath_rate (float): Predicted breath rate.
    """
    heart_rate_var.set(f"Heart Rate: {heart_rate:.2f} bpm")
    breath_rate_var.set(f"Breath Rate: {breath_rate:.2f} breaths/min")
Purpose: Updates the Tkinter GUI labels with the latest predictions.
Parameters:
heart_rate: The predicted heart rate.
breath_rate: The predicted breath rate.
5. Processing JSON Files with Retries and Delay
python
Copy code
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
Purpose: Handles the processing of each JSON file with a robust retry mechanism to ensure reliability.
Parameters:
json_file_path: The path to the JSON file to be processed.
retries: The maximum number of retry attempts (default is 3).
delay: The delay in seconds between each retry attempt (default is 10 seconds).
Workflow:
Attempt Loop: Iterates through the number of specified retries.
Delay: Waits for 10 seconds before each attempt to ensure the file is fully written.
File Size Check: Raises an error if the file is empty.
JSON Parsing: Loads the JSON data.
Feature Extraction: Extracts heartRate, breathRate, and waveform data.
Feature Consistency: Ensures all features required by the model are present, filling missing ones with zeros.
Prediction: Uses the loaded model to make predictions.
Queueing Results: Places the prediction results into the gui_queue for the GUI to update.
Error Handling: Catches and logs errors, retries processing if applicable, and skips the file after maximum retries.
6. Event Handler for New Files
python
Copy code
# ------------------------------
# Event Handler for New Files
# ------------------------------
class NewFileHandler(FileSystemEventHandler):
    """
    Custom event handler that processes new JSON files as they are created.
    """
    def __init__(self):
        super().__init__()
        self.processed_files = set()
        self.lock = threading.Lock()

    def on_created(self, event):
        """
        Called when a file or directory is created.
        
        Args:
            event: The event object containing event information.
        """
        # Process only files, not directories
        if not event.is_directory and event.src_path.endswith('.json'):
            with self.lock:
                if event.src_path not in self.processed_files:
                    self.processed_files.add(event.src_path)
                    print(f"Detected new JSON file: {event.src_path}")
                    # Start a new thread to process the file to avoid blocking
                    threading.Thread(target=process_json_file, args=(event.src_path,), daemon=True).start()
Purpose: Listens for new JSON file creation events and initiates their processing.
Components:
processed_files Set: Keeps track of files that have already been processed to prevent duplicate processing.
lock: Ensures thread-safe access to the processed_files set.
on_created Method: Triggered when a new file is created. It checks if the file is a JSON file and hasn't been processed before, then starts a new thread to process it.
7. Starting the Watchdog Observer
python
Copy code
# ------------------------------
# Function to Start the Watchdog Observer
# ------------------------------
def start_observer():
    """
    Initializes and starts the Watchdog observer to monitor the specified directory for new JSON files.
    """
    # Path to your binData directory
    live_data_directory = r'C:\ti\radar_toolbox_2_20_00_05\tools\visualizers\Applications_Visualizer\Industrial_Visualizer\binData'
    
    # Verify that the directory exists
    if not os.path.exists(live_data_directory):
        print(f"Error: Directory '{live_data_directory}' does not exist.")
        return

    # Initialize event handler and observer
    event_handler = NewFileHandler()
    observer = Observer()
    
    # Schedule the observer to monitor the directory recursively
    observer.schedule(event_handler, live_data_directory, recursive=True)
    
    # Start the observer
    observer.start()
    print(f"Started monitoring directory: {live_data_directory}")
    
    try:
        while True:
            time.sleep(1)  # Keep the thread running
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
Purpose: Sets up and starts the watchdog observer to monitor the specified directory and its subdirectories for new JSON files.
Components:
Directory Path: Specifies the path to the binData directory where JSON files are generated.
Directory Check: Ensures that the specified directory exists before starting the observer.
Event Handler and Observer Initialization: Connects the custom NewFileHandler to the Observer.
Recursive Monitoring: Enables monitoring of all subdirectories within the specified directory.
Observer Lifecycle: Starts the observer and keeps it running until a keyboard interrupt (Ctrl+C) is detected, at which point it stops gracefully.
8. Processing the GUI Queue
python
Copy code
# ------------------------------
# Function to Periodically Check the Queue and Update the GUI
# ------------------------------
def process_gui_queue():
    """
    Periodically checks the GUI queue for new prediction results and updates the GUI accordingly.
    """
    try:
        while True:
            heart_rate, breath_rate = gui_queue.get_nowait()
            update_vital_signs_in_gui(heart_rate, breath_rate)
    except queue.Empty:
        pass
    # Schedule the next queue check after 100 milliseconds
    root.after(100, process_gui_queue)
Purpose: Continuously monitors the gui_queue for new prediction results and updates the GUI labels with the latest values.
Mechanism:
Non-Blocking: Uses a try-except block to attempt to retrieve items from the queue without blocking if the queue is empty.
Scheduled Checks: Utilizes Tkinter's after method to schedule the next queue check every 100 milliseconds, ensuring timely GUI updates.
9. GUI Setup
python
Copy code
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
Components:
Main Window: Initializes the main Tkinter window with the title "Vital Sign Monitor".
StringVars: heart_rate_var and breath_rate_var hold the dynamic values to be displayed in the GUI.
Labels: Two labels display the heart rate and breath rate, respectively, initialized with placeholder text.
Observer Thread: Starts the watchdog observer in a separate daemon thread to ensure the GUI remains responsive.
Queue Processing: Initiates the periodic check of the gui_queue to update the GUI with new prediction results.
Event Loop: Starts the Tkinter main event loop, keeping the GUI window active.
