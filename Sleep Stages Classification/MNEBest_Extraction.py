import mne
import matplotlib.pyplot as plt
import os
import pandas as pd
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import random
import shutil
from PIL import Image, ImageEnhance, ImageOps, ImageFilter
import os
import matplotlib.pyplot as plt
import numpy as np
import mne
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np
import os
import mne

def getEDF_getExcel():
    EDF = []
    Excel = []
    for root, _, files in os.walk("."):
        for file in files:
            file_path = os.path.join(root, file)
            if file.endswith(".edf"):
                EDF.append(file_path)
            if file.endswith(('.xls', '.xlsx', '.csv')) and "sleepstages" in file.lower():
                Excel.append(file_path)
    return EDF, Excel


def get_label_excel(path): 
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    stages_list = df['Sleep Stage'].tolist()
    stages_list = [s.strip() for s in stages_list]
    return stages_list


def makeDataset(edf_list, excel_list, dataset_folder):
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)
    Labels = ['W', 'N1', 'N2', 'N3', 'R']
    class_folders = {}
    for label in Labels:
        folder = os.path.join(dataset_folder, label)
        if not os.path.exists(folder):
            os.makedirs(folder)
        class_folders[label] = folder

    for edf_file, excel_file in zip(edf_list, excel_list):
        sleep_stages_list = get_label_excel(excel_file)
        
        raw = mne.io.read_raw_edf(edf_file, preload=True, verbose=False)
        raw = raw.pick_types(eeg=True)
        print(f"Processing EDF file: {edf_file}")
        print(f"Available channels: {raw.info['ch_names']}")

        chosen_channels = ['EEG F3-REF', 'EEG F4-REF', 'EEG C3-REF',
                           'EEG C4-REF', 'EEG O1-REF', 'EEG O2-REF']
        
        raw = raw.copy().pick_channels(chosen_channels)
        raw.filter(l_freq=1, h_freq=20, picks="eeg")

        total_length = int(raw.times[-1])
        start_times = list(range(0, total_length, 30))

        # Define colors for each channel
        colors = ['r', 'g', 'b', 'm', 'c', 'y']

        for idx, start_time in enumerate(start_times):
            if idx >= len(sleep_stages_list):
                break
            label = sleep_stages_list[idx]
            if label not in Labels:
                continue
            end_time = min(start_time + 30, raw.times[-1])
            snippet_raw = raw.copy().crop(tmin=start_time, tmax=end_time)
            data, times = snippet_raw.get_data(return_times=True)

            # Normalize each channel
            data_normalized = []
            for channel_data in data:
                scaler = MinMaxScaler(feature_range=(-0.5, 0.5))  # Adjust range as needed
                normalized_channel = scaler.fit_transform(channel_data.reshape(-1, 1)).flatten()
                data_normalized.append(normalized_channel)
            data_normalized = np.array(data_normalized)

            # Cap outlier values
            threshold = 0.5  
            data_clipped = np.clip(data_normalized, -threshold, threshold)

            # Apply offsets for visualization
            offset_step = 1  
            data_offset = []
            for i, channel_data in enumerate(data_clipped):
                data_offset.append(channel_data + i * offset_step)
            data_offset = np.array(data_offset)

            # Plot the signals
            fig, ax = plt.subplots(figsize=(10, 5))
            for i, channel_data in enumerate(data_offset):
                ax.plot(times, channel_data, color=colors[i % len(colors)], linewidth=1)

            # Remove labels, ticks, and grid for clean visualization
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_frame_on(False)
            ax.axis('off')

            plt.tight_layout()

            output_filename = f"subject_{os.path.basename(edf_file).split('.')[0]}_segment_{start_time}_{label}.png"
            output_file_path = os.path.join(class_folders[label], output_filename)
            plt.savefig(output_file_path, bbox_inches='tight', pad_inches=0)
            plt.close(fig)


EDFList, ExcelList = getEDF_getExcel()
dataset_folder = os.path.join(os.getcwd(), "Dataset")
makeDataset(EDFList, ExcelList, dataset_folder)

def get_class_distribution(folder):
    class_counts = {}
    for label in os.listdir(folder):
        label_folder = os.path.join(folder, label)
        if os.path.isdir(label_folder):
            num_images = len([img for img in os.listdir(label_folder) if img.endswith('.png')])
            class_counts[label] = num_images
    return class_counts


class_counts = get_class_distribution(dataset_folder)

print(class_counts)
SortedClassCount = sorted(set(class_counts.values()))
if len(SortedClassCount) > 1:
    SecondSmallest = SortedClassCount[0] 
else:
    SecondSmallest = SortedClassCount[0]

Modify = {}
for key,value in class_counts.items(): 
    Modify[key] = int(SecondSmallest) - int(value)  # If positive need to add elements else we subtract from the list
main_folders = ["Train", "Test", "Validation"]
sub_folders = ["W", "N1", "N2", "N3", "R"]
for folder in main_folders:
    
    # Construct the path for the main folder
    main_folder_path = os.path.join(".", folder)

    # Loop through each subfolder and create them under the main folder
    for sub in sub_folders:
        sub_folder_path = os.path.join(main_folder_path, sub)
        if not os.path.exists(sub_folder_path):
            os.makedirs(sub_folder_path)
            
print(Modify)


def random_crop(image, crop_percent=10):
    """Randomly crop a small portion of the image."""
    width, height = image.size
    crop_width = int(width * crop_percent / 100)
    crop_height = int(height * crop_percent / 100)
    return image.crop((crop_width, crop_height, width - crop_width, height - crop_height))

def resize_image(image, scale_factor=1.1):
    """Resize the image by a given scale factor."""
    width, height = image.size
    return image.resize((int(width * scale_factor), int(height * scale_factor)))

def horizontal_shift(image, shift_pixels):
    """Shift the image horizontally by a given number of pixels."""
    width, height = image.size
    shifted = Image.new("RGB", (width, height), (255, 255, 255))  # White background
    shifted.paste(image, (shift_pixels, 0))
    return shifted

def adjust_brightness(image, factor=1.2):
    """Adjust the brightness of the image."""
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(factor)

def adjust_contrast(image, factor=1.2):
    """Adjust the contrast of the image."""
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(factor)

def apply_gaussian_blur(image, radius=1):
    """Apply Gaussian blur to the image."""
    return image.filter(ImageFilter.GaussianBlur(radius))

def add_border(image, border_size=10):
    """Add a border to the image."""
    return ImageOps.expand(image, border=border_size, fill="white")

def augment_waveform_image(image):
    
    augmentations = [
        lambda img: random_crop(img, crop_percent=10),  
        lambda img: resize_image(img, scale_factor=1.1),  
        lambda img: horizontal_shift(img, shift_pixels=5),
        lambda img: adjust_brightness(img, factor=1.2),  
        lambda img: adjust_contrast(img, factor=1.2),  
        lambda img: apply_gaussian_blur(img, radius=1),  
        lambda img: add_border(img, border_size=10)  
    ]

    # Apply a random augmentation
    return random.choice(augmentations)(image)   
    
def IncreaseDataset(n, folderPath):
    # Collect all image files in the folder
    image_files = [f for f in os.listdir(folderPath) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not image_files:
        print("No images found in the specified folder.")
        return

    print(f"Found {len(image_files)} images. Generating {n} augmented images.")

    for i in range(n):
        # Randomly select an image
        selected_image_file = random.choice(image_files)
        selected_image_path = os.path.join(folderPath, selected_image_file)
        image = Image.open(selected_image_path)

        # Apply augmentation
        augmented_image = augment_waveform_image(image)

        # Generate a unique name for the augmented image
        new_filename = f"aug_{i + 1}_{selected_image_file}"
        new_image_path = os.path.join(folderPath, new_filename)

        # Save the augmented image in the same folder
        augmented_image.save(new_image_path)

    print(f"{n} augmented images saved in {folderPath}.")
    
def DecreaseDataset(n, DFolderPath): 
    n = -1 * n 
    png_files = [f for f in os.listdir(DFolderPath) if f.lower().endswith('.png')]

    files_to_delete = random.sample(png_files, n)

    for file_name in files_to_delete:
        file_path = os.path.join(DFolderPath, file_name)
        try:
            os.remove(file_path)
            print(f"Deleted: {file_name}")
        except Exception as e:
            print(f"Error deleting {file_name}: {e}")
            
    
for subfolder, value in Modify.items():  
    subfolder_path = os.path.join(dataset_folder, subfolder)  

    if value == 0:
        print(f"Subfolder '{subfolder}' requires no changes.")
    elif value > 0:
        IncreaseDataset(value, subfolder_path)  
    else:
        DecreaseDataset(value, subfolder_path)  
class_counts = get_class_distribution(dataset_folder)

print(class_counts)        


def split_dataset_to_new_folders(dataset_folder, output_folder = ".", train_ratio=0.7, test_ratio=0.15, val_ratio=0.15):
    # Subfolder names (W, R, N1, N2, N3)
    subfolders = ["W", "R", "N1", "N2", "N3"]

    # Create Train, Test, and Validation subfolders in the output directory
    splits = ["Train", "Test", "Validation"]
    for split in splits:
        for subfolder in subfolders:
            os.makedirs(os.path.join(output_folder, split, subfolder), exist_ok=True)

    # Process each subfolder in the main dataset
    for subfolder in subfolders:
        subfolder_path = os.path.join(dataset_folder, subfolder)
        if not os.path.exists(subfolder_path):
            print(f"Subfolder {subfolder} does not exist in {dataset_folder}. Skipping...")
            continue

        # List all files in the current subfolder
        files = [f for f in os.listdir(subfolder_path) if os.path.isfile(os.path.join(subfolder_path, f))]
        
        # Shuffle the files for randomness
        random.shuffle(files)

        # Calculate split indices
        total_files = len(files)
        train_count = int(total_files * train_ratio)
        test_count = int(total_files * test_ratio)
        val_count = total_files - train_count - test_count  # Ensure no files are left out

        # Split files
        train_files = files[:train_count]
        test_files = files[train_count:train_count + test_count]
        val_files = files[train_count + test_count:]

        # Copy files to Train
        for file in train_files:
            src_path = os.path.join(subfolder_path, file)
            dest_path = os.path.join(output_folder, "Train", subfolder, file)
            shutil.copy(src_path, dest_path)

        # Copy files to Test
        for file in test_files:
            src_path = os.path.join(subfolder_path, file)
            dest_path = os.path.join(output_folder, "Test", subfolder, file)
            shutil.copy(src_path, dest_path)

        # Copy files to Validation
        for file in val_files:
            src_path = os.path.join(subfolder_path, file)
            dest_path = os.path.join(output_folder, "Validation", subfolder, file)
            shutil.copy(src_path, dest_path)

        print(f"Split for {subfolder}: {train_count} Train, {test_count} Test, {val_count} Validation.")

    print("Dataset split complete!")


split_dataset_to_new_folders(dataset_folder)

