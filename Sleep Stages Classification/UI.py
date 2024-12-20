import os
import numpy as np
import pandas as pd
from mne.io import read_raw_edf
from scipy.signal import butter, filtfilt
import tensorflow as tf
from tensorflow.keras.models import load_model
from tkinter import Tk, filedialog, Label, Button, Frame, ttk, PhotoImage
import tkinter as tk
from tkinter import messagebox
import re
import threading
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

TARGET_CHANNELS = ['EEG F3-REF', 'EEG F4-REF', 'EEG C3-REF',
                   'EEG C4-REF', 'EEG O1-REF', 'EEG O2-REF']
FS = 500  

# Sleep Stage Mapping
SLEEP_STAGE_MAPPING = {'W': 0, 'N1': 1, 'N2': 2, 'N3': 3, 'R': 4}
INVERSE_SLEEP_STAGE_MAPPING = {v: k for k, v in SLEEP_STAGE_MAPPING.items()}

# Helper Functions
def bandpass_filter(data, low_cutoff, high_cutoff, fs, order=5):
    nyquist = 0.5 * fs
    low = low_cutoff / nyquist
    high = high_cutoff / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

def process_edf_file(edf_file, fs=500):
    try:
        raw = read_raw_edf(edf_file, preload=True, verbose=False)
        print(f"Available Channels: {raw.ch_names}")

        # Select target channels
        target_channel_indices = [raw.ch_names.index(ch) for ch in TARGET_CHANNELS if ch in raw.ch_names]
        selected_channels = [raw.ch_names[i] for i in target_channel_indices]
        print(f"Selected Channels: {selected_channels}")

        if not target_channel_indices:
            print(f"No matching channels found in {edf_file}.")
            return None, None

        # Keep only the TARGET_CHANNELS in raw data for plotting
        raw.pick_channels(selected_channels)

        # Calculate the number of 30-second segments
        segment_length = int(30 * fs) 
        num_segments = raw.n_times // segment_length
        X = []

        for segment_idx in range(int(num_segments)):
            start = segment_idx * segment_length
            end = start + segment_length
            if end > raw.n_times:
                print(f"Skipping segment {segment_idx}: exceeds EDF time range.")
                continue

            try:
                segment_data = raw.get_data(start=start, stop=end)
            except Exception as e:
                print(f"Error extracting segment {segment_idx} from {edf_file}: {e}")
                continue

            try:
                # Apply bandpass filter to all channels with low_cutoff=1 Hz and high_cutoff=30 Hz
                filtered_data = [bandpass_filter(ch_data, 1, 30, fs) for ch_data in segment_data]
                # Downsample the data by factor of 5
                downsampled_data = [ch_data[::5] for ch_data in filtered_data]  # Each ch_data is now length 3000
                # Stack channels to create multichannel input
                segment_array = np.stack(downsampled_data, axis=-1)  # Shape: (sequence_length, num_channels)
                X.append(segment_array)
            except Exception as e:
                print(f"Error processing segment {segment_idx}: {e}")
                continue

        if len(X) > 0:
            X = np.array(X)  # Shape: (num_samples, sequence_length, num_channels)
            # Normalize per sample
            X = (X - np.mean(X, axis=1, keepdims=True)) / \
                (np.std(X, axis=1, keepdims=True) + 1e-7)
            return X, raw 
        else:
            print(f"No valid data extracted from {edf_file}.")
            return None, None

    except Exception as e:
        print(f"Error processing EDF file {edf_file}: {e}")
        return None, None

def load_actual_sleep_stages(csv_file):
    try:
        labels = pd.read_csv(csv_file)
        labels.columns = labels.columns.str.strip()
        if "Sleep Stage" not in labels.columns:
            print(f"Missing 'Sleep Stage' column in {csv_file}.")
            return None

        # Convert to string and strip whitespace
        labels["Sleep Stage"] = labels["Sleep Stage"].astype(str).str.strip()

        # Remove any special characters from labels
        labels["Sleep Stage"] = labels["Sleep Stage"].apply(lambda x: re.sub(r'\s+', '', x))

        # Replace empty strings with NaN without using inplace=True
        labels["Sleep Stage"] = labels["Sleep Stage"].replace('', np.nan)

        # Drop rows with missing labels
        labels.dropna(subset=["Sleep Stage"], inplace=True)

        # Map sleep stages using SLEEP_STAGE_MAPPING
        sleep_stages = labels["Sleep Stage"].map(SLEEP_STAGE_MAPPING)

        # Identify invalid labels
        invalid_labels = labels["Sleep Stage"][sleep_stages.isnull()].unique()
        if len(invalid_labels) > 0:
            print(f"Invalid sleep stage labels found in {csv_file}: {invalid_labels}")
            # Drop rows with invalid labels
            labels = labels[~sleep_stages.isnull()]
            sleep_stages = sleep_stages.dropna()
            sleep_stages = sleep_stages.values
        else:
            sleep_stages = sleep_stages.values

        return sleep_stages
    except Exception as e:
        print(f"Error loading CSV file {csv_file}: {e}")
        return None

# GUI Application
class SleepStageApp:
    def __init__(self, master):
        self.master = master
        master.title("Sleep Stage Prediction")
        master.geometry("800x600")
        master.resizable(False, False)
        
        # Set style
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TButton', font=('Helvetica', 12), padding=5)
        style.configure('TLabel', font=('Helvetica', 12))
        style.configure('Treeview.Heading', font=('Helvetica', 12, 'bold'))
        style.configure('Treeview', font=('Helvetica', 11))
        
        # Load an image for the header (optional)
        try:
            self.header_image = PhotoImage(file="header.png")  
            header_label = Label(master, image=self.header_image)
            header_label.pack(pady=10)
        except Exception as e:
            header_label = Label(master, text="Sleep Stage Prediction App", font=("Helvetica", 20, 'bold'))
            header_label.pack(pady=10)
        
        # Frame for buttons
        button_frame = Frame(master)
        button_frame.pack(pady=10)
        
        self.load_model_button = ttk.Button(button_frame, text="Load Model", command=self.load_model)
        self.load_model_button.grid(row=0, column=0, padx=10)
        
        self.load_edf_button = ttk.Button(button_frame, text="Load EDF and CSV Files", command=self.load_edf_and_csv_files, state='disabled')
        self.load_edf_button.grid(row=0, column=1, padx=10)

        self.display_waveform_button = ttk.Button(button_frame, text="Display EEG Waveforms", command=self.display_eeg_waveforms, state='disabled')
        self.display_waveform_button.grid(row=0, column=2, padx=10)
        
        # Progress label
        self.progress_label = Label(master, text="", font=("Helvetica", 12))
        self.progress_label.pack(pady=5)
        
        # Frame for table
        self.table_frame = Frame(master)
        self.table_frame.pack(pady=10, fill='both', expand=True)
        
        # Add a scrollbar to the table
        self.tree_scroll = ttk.Scrollbar(self.table_frame)
        self.tree_scroll.pack(side='right', fill='y')
        
        self.tree = None

    def load_model(self):
        model_path = filedialog.askopenfilename(title="Select Trained Model",
                                                filetypes=[("H5 files", "*.h5")])
        if model_path:
            self.progress_label.config(text="Loading model...")
            self.master.update()
            try:
                self.model = load_model(model_path)
                print(f"Model loaded from {model_path}")
                self.load_edf_button.config(state='normal')
                self.progress_label.config(text="Model loaded successfully.")
                messagebox.showinfo("Success", "Model loaded successfully.")
            except Exception as e:
                print(f"Error loading model: {e}")
                self.progress_label.config(text="Error loading model.")
                messagebox.showerror("Error", f"Error loading model: {e}")

    def load_edf_and_csv_files(self):
        edf_file = filedialog.askopenfilename(title="Select EDF file",
                                              filetypes=[("EDF files", "*.edf")])
        if edf_file:
            csv_file = filedialog.askopenfilename(title="Select CSV file with Actual Sleep Stages",
                                                  filetypes=[("CSV files", "*.csv")])
            if csv_file:
                threading.Thread(target=self.process_and_predict, args=(edf_file, csv_file)).start()
            else:
                messagebox.showwarning("Warning", "No CSV file selected.")

    def process_and_predict(self, edf_file, csv_file):
        self.progress_label.config(text="Processing EDF file...")
        self.master.update()
        X, self.raw_data = process_edf_file(edf_file, fs=FS)
        if X is not None:
            self.display_waveform_button.config(state='normal')  
            self.progress_label.config(text="Loading actual sleep stages...")
            self.master.update()
            actual_stages = load_actual_sleep_stages(csv_file)
            if actual_stages is not None:
                # Ensure that the number of epochs matches
                if len(actual_stages) >= len(X):
                    self.actual_sleep_stages = actual_stages[:len(X)]
                    self.predict_sleep_stages(X)
                else:
                    self.progress_label.config(text="Mismatch in number of epochs.")
                    messagebox.showerror("Error", "Mismatch in number of epochs between EDF and CSV files.")
                    print("Mismatch in number of epochs between EDF and CSV files")
            else:
                self.progress_label.config(text="Error loading CSV file.")
                messagebox.showerror("Error", "Error loading CSV file.")
        else:
            self.progress_label.config(text="Error processing EDF file.")
            messagebox.showerror("Error", "Error processing EDF file.")

    def predict_sleep_stages(self, X):
        self.progress_label.config(text="Predicting sleep stages...")
        self.master.update()
        try:
            predictions = self.model.predict(X)
            predicted_labels = predictions.argmax(axis=1)
            predicted_stages = [INVERSE_SLEEP_STAGE_MAPPING[label] for label in predicted_labels]
            actual_stages = [INVERSE_SLEEP_STAGE_MAPPING.get(label, 'Unknown') for label in self.actual_sleep_stages]
            self.display_results(predicted_stages, actual_stages)
            self.progress_label.config(text="Prediction completed.")
            messagebox.showinfo("Success", "Prediction completed.")
        except Exception as e:
            print(f"Error during prediction: {e}")
            self.progress_label.config(text="Error during prediction.")
            messagebox.showerror("Error", f"Error during prediction: {e}")

    def display_results(self, predicted_stages, actual_stages=None):
        if self.tree:
            self.tree.destroy()
            self.tree_scroll.destroy()
            self.tree_scroll = ttk.Scrollbar(self.table_frame)
            self.tree_scroll.pack(side='right', fill='y')

        columns = ('Epoch', 'Predicted Sleep Stage')
        if actual_stages is not None:
            columns += ('Actual Sleep Stage',)

        self.tree = ttk.Treeview(self.table_frame, columns=columns, show='headings', yscrollcommand=self.tree_scroll.set)
        self.tree_scroll.config(command=self.tree.yview)

        self.tree.heading('Epoch', text='Epoch', anchor='center')
        self.tree.heading('Predicted Sleep Stage', text='Predicted Sleep Stage', anchor='center')
        if actual_stages is not None:
            self.tree.heading('Actual Sleep Stage', text='Actual Sleep Stage', anchor='center')

        self.tree.column('Epoch', anchor='center', width=80)
        self.tree.column('Predicted Sleep Stage', anchor='center', width=150)
        if actual_stages is not None:
            self.tree.column('Actual Sleep Stage', anchor='center', width=150)

        # Add data to the table
        for i, stage in enumerate(predicted_stages):
            values = [i+1, stage]
            if actual_stages is not None:
                values.append(actual_stages[i])
            self.tree.insert('', 'end', values=values)

        # Apply striped row tags
        self.tree.tag_configure('oddrow', background='#E8E8E8')
        self.tree.tag_configure('evenrow', background='#DFDFDF')

        for idx, item in enumerate(self.tree.get_children()):
            if idx % 2 == 0:
                self.tree.item(item, tags=('evenrow',))
            else:
                self.tree.item(item, tags=('oddrow',))

        self.tree.pack(fill='both', expand=True)

    def display_eeg_waveforms(self):
        if not hasattr(self, 'raw_data'):
            messagebox.showerror("Error", "No EEG data available.")
            return

        # Get EEG data and times
        self.eeg_data, self.eeg_times = self.raw_data.get_data(return_times=True)
        self.current_segment = 0
        self.segment_length = 30 
        self.num_segments = int(self.eeg_times[-1] // self.segment_length) + 1

        # Create a new window for waveform display
        self.waveform_window = tk.Toplevel(self.master)
        self.waveform_window.title("EEG Waveforms")
        self.waveform_window.geometry("1200x700")

        # Frame for navigation buttons and search
        nav_frame = tk.Frame(self.waveform_window)
        nav_frame.pack(side='top', fill='x')

        prev_button = ttk.Button(nav_frame, text="Previous", command=self.prev_segment)
        prev_button.pack(side='left')

        self.segment_entry = ttk.Entry(nav_frame, width=5)
        self.segment_entry.pack(side='left', padx=5)

        search_button = ttk.Button(nav_frame, text="Go to Segment", command=self.go_to_segment)
        search_button.pack(side='left', padx=5)

        next_button = ttk.Button(nav_frame, text="Next", command=self.next_segment)
        next_button.pack(side='right')

        # Frame for plot
        plot_frame = tk.Frame(self.waveform_window)
        plot_frame.pack(side='top', fill='both', expand=True)

        # Figure and canvas
        self.fig, self.ax = plt.subplots(figsize=(12, 7))
        self.canvas_fig = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas_fig.draw()
        self.canvas_fig.get_tk_widget().pack(side='top', fill='both', expand=True)

        self.plot_segment()

    def plot_segment(self):
        start_time = self.current_segment * self.segment_length
        end_time = start_time + self.segment_length
        indices = np.where((self.eeg_times >= start_time) & (self.eeg_times < end_time))[0]
        if len(indices) == 0:
            return

        self.ax.clear()
        num_channels = len(self.raw_data.ch_names)
        scaling_factor = 1e6  # Convert from V to µV
        offset = 200  # Adjust offset to prevent overlapping

        for i, channel_data in enumerate(self.eeg_data):
            channel_segment = channel_data[indices] * scaling_factor
            self.ax.plot(self.eeg_times[indices], channel_segment + i * offset, label=self.raw_data.ch_names[i])

        self.ax.set_xlabel('Time (s)')
        self.ax.set_ylabel('Amplitude (µV)')
        self.ax.set_title(f"EEG Waveforms (Segment {self.current_segment + 1}/{self.num_segments})")
        self.ax.legend(loc='upper right')
        self.ax.set_ylim([-offset, (num_channels) * offset])
        self.canvas_fig.draw()

    def prev_segment(self):
        if self.current_segment > 0:
            self.current_segment -= 1
            self.plot_segment()

    def next_segment(self):
        if self.current_segment < self.num_segments - 1:
            self.current_segment += 1
            self.plot_segment()

    def go_to_segment(self):
        try:
            segment = int(self.segment_entry.get()) - 1 
            if 0 <= segment < self.num_segments:
                self.current_segment = segment
                self.plot_segment()
            else:
                messagebox.showerror("Error", "Segment out of range.")
        except ValueError:
            messagebox.showerror("Error", "Invalid segment number.")


# Run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = SleepStageApp(root)
    root.mainloop()
