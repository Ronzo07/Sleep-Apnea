import os
import numpy as np
import pandas as pd
from mne.io import read_raw_edf
from scipy.signal import butter, filtfilt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling1D, Bidirectional, LSTM
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tkinter import Tk, filedialog
import tensorflow as tf
from imblearn.over_sampling import SMOTE


TARGET_CHANNELS = ['EEG F3-REF', 'EEG F4-REF', 'EEG C3-REF', 'EEG C4-REF', 'EEG O1-REF', 'EEG O2-REF']
FS = 500  # Sampling frequency (500 Hz)

# Sleep Stage Mapping
SLEEP_STAGE_MAPPING = {'W': 0, 'N1': 1, 'N2': 2, 'N3': 3, 'R': 4}


def bandpass_filter(data, low_cutoff, high_cutoff, fs, order=5):
    nyquist = 0.5 * fs
    low = low_cutoff / nyquist
    high = high_cutoff / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

def select_files():
    Tk().withdraw()  
    edf_files = filedialog.askopenfilenames(title="Select EDF files", filetypes=[("EDF files", "*.edf")])
    csv_files = filedialog.askopenfilenames(title="Select CSV files", filetypes=[("CSV files", "*.csv")])
    if len(edf_files) != len(csv_files):
        raise ValueError("The number of EDF files and CSV files must match.")
    return edf_files, csv_files

def augment_time_series(X, y):
   
    augmented_X = []
    augmented_y = []

    for i in range(len(X)):
        x = X[i]
        label = y[i]
        augmented_X.append(x)
        augmented_y.append(label)

        
        noise = np.random.normal(0, 0.05, x.shape)
        x_noise = x + noise
        augmented_X.append(x_noise)
        augmented_y.append(label)

        
        shift = np.random.randint(-100, 100)
        x_shift = np.roll(x, shift, axis=0)
        augmented_X.append(x_shift)
        augmented_y.append(label)

    return np.array(augmented_X), np.array(augmented_y)

def process_multiple_patients(edf_files, csv_files, fs=500):
    all_X, all_y = [], []
    for edf_file, csv_file in zip(edf_files, csv_files):
        print(f"Processing: {edf_file} and {csv_file}")
        try:
            raw = read_raw_edf(edf_file, preload=True)
            print(f"Available Channels: {raw.ch_names}")

            
            target_channel_indices = [raw.ch_names.index(ch) for ch in TARGET_CHANNELS if ch in raw.ch_names]
            selected_channels = [raw.ch_names[i] for i in target_channel_indices]
            print(f"Selected Channels: {selected_channels}")

            if not target_channel_indices:
                print(f"No matching channels found in {edf_file}. Skipping this patient.")
                continue

            labels = pd.read_csv(csv_file)
            labels.columns = labels.columns.str.strip()
            if "Sleep Stage" not in labels.columns:
                print(f"Missing 'Sleep Stage' column in {csv_file}. Skipping this patient.")
                continue

            
            sleep_stages = labels["Sleep Stage"].str.strip().map(SLEEP_STAGE_MAPPING)
            if sleep_stages.isnull().any():
                print(f"Invalid sleep stage labels found in {csv_file}. Skipping this patient.")
                continue

            sleep_stages = sleep_stages.values
            segment_length = int(30 * fs)  # 30-second segments
            num_segments = min(len(sleep_stages), raw.n_times // segment_length)
            X, y = [], []

            for segment_idx in range(num_segments):
                start = segment_idx * segment_length
                end = start + segment_length
                if end > raw.n_times:
                    print(f"Skipping segment {segment_idx}: exceeds EDF time range.")
                    continue

                try:
                    segment_data = raw.get_data(start=start, stop=end)[target_channel_indices]
                except Exception as e:
                    print(f"Error extracting segment {segment_idx} from {edf_file}: {e}")
                    continue

                try:
                    
                    filtered_data = [bandpass_filter(ch_data, 0.5, 40, fs) for ch_data in segment_data]
                    
                    downsampled_data = [ch_data[::5] for ch_data in filtered_data]  
                    segment_array = np.stack(downsampled_data, axis=-1)  
                    X.append(segment_array)
                    y.append(sleep_stages[segment_idx])
                except Exception as e:
                    print(f"Error processing segment {segment_idx}: {e}")
                    continue

            if len(X) > 0 and len(y) > 0:
                X = np.array(X)  
                y = np.array(y)
                all_X.append(X)
                all_y.append(y)
            else:
                print(f"No valid data extracted for {edf_file}. Skipping this patient.")

        except Exception as e:
            print(f"Error processing patient files {edf_file} and {csv_file}: {e}")
            continue

    if len(all_X) == 0 or len(all_y) == 0:
        raise ValueError("No valid data was extracted for any patient.")

    X_combined = np.concatenate(all_X, axis=0)
    y_combined = np.concatenate(all_y, axis=0)
    return X_combined, y_combined

def create_model(input_shape, num_classes):
    model = Sequential()
    model.add(Conv1D(128, kernel_size=7, activation='relu', padding='same', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))

    model.add(Conv1D(256, kernel_size=5, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))

    model.add(Conv1D(512, kernel_size=3, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))

    model.add(Bidirectional(LSTM(128, return_sequences=True)))
    model.add(Bidirectional(LSTM(64)))

    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    return model


try:
    edf_files, csv_files = select_files()  # Interactive file selection
    print("Loading and preprocessing data...")
    X, y = process_multiple_patients(edf_files, csv_files, fs=FS)

    
    y = y.astype(int)


    X = (X - np.mean(X, axis=1, keepdims=True)) / (np.std(X, axis=1, keepdims=True) + 1e-7)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    X_train_aug, y_train_aug = augment_time_series(X_train, y_train)
    print("Data augmentation completed on training set.")

    
    num_samples, sequence_length, num_channels = X_train_aug.shape
    X_reshaped = X_train_aug.reshape((num_samples, sequence_length * num_channels))

    # Oversample minority classes using SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_reshaped, y_train_aug)

    
    X_resampled = X_resampled.reshape(-1, sequence_length, num_channels)

    
    from tensorflow.keras.utils import to_categorical
    num_classes = len(SLEEP_STAGE_MAPPING)
    y_train_cat = to_categorical(y_resampled, num_classes=num_classes)
    y_val_cat = to_categorical(y_val, num_classes=num_classes)
    y_test_cat = to_categorical(y_test, num_classes=num_classes)

    # Create Model
    input_shape = X_resampled.shape[1:]  
    model = create_model(input_shape, num_classes)
    model.compile(optimizer=Adam(learning_rate=1e-4),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.5)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    
    history = model.fit(
        X_resampled, y_train_cat, validation_data=(X_val, y_val_cat),
        epochs=8, batch_size=64, callbacks=[lr_scheduler, early_stopping]
    )

    
    test_loss, test_acc = model.evaluate(X_test, y_test_cat)
    print(f"Test Accuracy: {test_acc:.2f}")
    
    model.save('sleep_stage_model.h5')
    print("Model saved as 'sleep_stage_model.h5'")


    # Plot training & validation accuracy and loss
    plt.figure(figsize=(12, 4))
    plt.subplot(1,2,1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.subplot(1,2,2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

    # Confusion Matrix
    y_pred = model.predict(X_test).argmax(axis=1)
    y_true = y_test
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=list(SLEEP_STAGE_MAPPING.keys()),
                yticklabels=list(SLEEP_STAGE_MAPPING.keys()))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

    # Classification Report
    print(classification_report(y_true, y_pred, target_names=list(SLEEP_STAGE_MAPPING.keys())))

except Exception as e:
    print(f"An error occurred: {e}")

