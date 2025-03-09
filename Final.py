import tensorflow as tf
import numpy as np
import pandas as pd
import os
import sys
import markdown2
import pathlib
import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import hamming_loss
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from monai.transforms import (
    LoadImaged, EnsureChannelFirstd, Spacingd, 
    CenterSpatialCropd, ToTensord,Compose,MapTransform,
)
from monai.data import CacheDataset
from monai.utils import set_determinism
from monai.losses import DiceCELoss

import google.generativeai as genai

from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QLineEdit, QComboBox, QPushButton, 
    QVBoxLayout, QHBoxLayout, QMessageBox, QFileDialog
)
from PyQt5.QtCore import pyqtSignal


class CustomDataset(Dataset):
    def __init__(self, output_masks, mri_df):
        self.output_masks = output_masks
        self.mri_data = mri_df.to_numpy()  # Convert DataFrame to numpy array for easy indexing

    def __len__(self):
        return len(self.output_masks)

    def __getitem__(self, idx):
        label_data = self.output_masks[idx]
        mri_data = self.mri_data[idx]

        label_data = torch.tensor(label_data, dtype=torch.float32) if not isinstance(label_data, torch.Tensor) else label_data
        mri_data = torch.tensor(mri_data, dtype=torch.float32) if not isinstance(mri_data, torch.Tensor) else mri_data

        return {'label': label_data, 'mri': mri_data}

class NormalizeToOneRange(MapTransform):
    def __init__(self, keys):
        super().__init__(keys)
    
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key] = (d[key] - d[key].min()) / (d[key].max() - d[key].min())
        return d

class Config:
    def __init__(self):
        self.roi_size = (128, 128, 12)  # Example ROI size

config = Config()

def pad_collate_fn(batch):
    # If batch is empty, return an empty tensor
    if not batch:
        return {'image': torch.tensor([])}
    
    # Determine the maximum size for padding
    max_size = [max([img['image'].shape[i] for img in batch]) for i in range(4)]
    
    padded_images = []

    for item in batch:
        img = item['image']
        
        # Pad image to the maximum size
        padded_image = torch.nn.functional.pad(img, 
                                               (0, max_size[3] - img.shape[3], 
                                                0, max_size[2] - img.shape[2], 
                                                0, max_size[1] - img.shape[1], 
                                                0, max_size[0] - img.shape[0]))
        
        padded_images.append(padded_image)
    
    return {'image': torch.stack(padded_images)}

# Define the function to visualize outputs
def visualize_output(input_image, output_mask, slice_idx=0):
    # Check dimensions
    if slice_idx >= input_image.shape[4] or slice_idx < 0:
        raise ValueError(f"slice_idx {slice_idx} is out of range. Valid range is 0 to {input_image.shape[4]-1}")
    
    input_slice = input_image[0, 0, :, :, slice_idx].cpu().numpy()
    mask_slice = output_mask[0, 0, :, :, slice_idx].cpu().numpy()
    
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.title('Input Image')
    plt.imshow(input_slice, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title('Output Mask')
    plt.imshow(mask_slice, cmap='gray')
    plt.axis('off')
    
    plt.show()

class Simple3DCNN(nn.Module):
    def __init__(self, input_depth=12, input_height=128, input_width=128, num_classes=43, demographic_size=8):
        super(Simple3DCNN, self).__init__()

        self.input_depth = input_depth
        self.input_height = input_height
        self.input_width = input_width
        self.num_classes = num_classes
        self.demographic_size = demographic_size

        # Define the layers of the CNN
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv3d(in_channels=64, out_channels=128, kernel_size=3, padding=1)

        # Define pooling layer
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        # Define dropout layer
        self.dropout = nn.Dropout3d(0.3)

        # Calculate flattened size based on input dimensions
        self.flattened_size = self._calculate_flattened_size()

        # Define fully connected layers
        self.fc1 = nn.Linear(self.flattened_size + self.demographic_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)  # Adjust output size based on your target data

    def _calculate_flattened_size(self):
        # Pass a dummy tensor through the layers to calculate flattened size
        dummy_input = torch.zeros(1, 1, self.input_depth, self.input_height, self.input_width)
        x = F.leaky_relu(self.conv1(dummy_input))
        x = self.pool(x)
        x = F.leaky_relu(self.conv2(x))
        x = self.pool(x)
        x = F.leaky_relu(self.conv3(x))
        x = self.pool(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        flattened_size = x.size(1)  # Number of elements in the tensor
        return flattened_size

    def forward(self, mask, demographics):
        # Ensure mask is in the correct format
        mask = mask.permute(0, 1, 4, 2, 3)  # Reorder dimensions to (batch_size, channels, depth, height, width)

        x = F.leaky_relu(self.conv1(mask))
        x = self.pool(x)
        x = F.leaky_relu(self.conv2(x))
        x = self.pool(x)
        x = F.leaky_relu(self.conv3(x))
        x = self.pool(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor

        # Concatenate features with demographics
        combined = torch.cat((x, demographics), dim=1)
        if self.fc1.weight.shape[1] != combined.shape[1]:
            in_features = combined.shape[1]
            self.fc1 = nn.Linear(in_features, 512)
        x = F.leaky_relu(self.fc1(combined))
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)

        return x


def prepare(image_file, pixdim=(1.5, 1.5, 1.0), spatial_size=[128, 128, 12], cache=True):
    set_determinism(seed=0)

    image_file = pathlib.Path(image_file)

    if not image_file.is_file():
        raise ValueError(f"The provided path is not a valid file: {image_file}")

    print(f"Loading image from file: {image_file}")

    train_files = [{'image': str(image_file)}]

    train_transforms = Compose(
        [
            LoadImaged(keys=["image"]),  # Load image
            EnsureChannelFirstd(keys=["image"]),  # Ensure channel-first format
            Spacingd(keys=["image"], pixdim=pixdim, mode="bilinear"),  # Resample spacing
            CenterSpatialCropd(keys=["image"], roi_size=spatial_size),  # Crop to desired size
            NormalizeToOneRange(keys=["image"]),
            ToTensord(keys=["image"]),  # Convert to PyTorch tensor
        ]
    )

    if cache:
        train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=1.0)
    else:
        train_ds = Dataset(data=train_files, transform=train_transforms)

    train_loader = DataLoader(train_ds, batch_size=1, collate_fn=pad_collate_fn)
    
    return train_loader
# Define the DoubleConv3D class
class DoubleConv3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv3D, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=0.5)  # Dropout layer
        )
        self._initialize_weights()

    def forward(self, x):
        return self.conv(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

# Define the UNet3D class
class UNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256]):
        super(UNet3D, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        for feature in features:
            self.downs.append(DoubleConv3D(in_channels, feature))
            in_channels = feature

        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose3d(
                    feature * 2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(DoubleConv3D(feature * 2, feature))

        self.bottleneck = DoubleConv3D(features[-1], features[-1] * 2)
        self.final_conv = nn.Conv3d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]

            diff_z = skip_connection.shape[2] - x.shape[2]
            diff_y = skip_connection.shape[3] - x.shape[3]
            diff_x = skip_connection.shape[4] - x.shape[4]
            x = F.pad(x, (diff_x // 2, diff_x - diff_x // 2,
                          diff_y // 2, diff_y - diff_y // 2,
                          diff_z // 2, diff_z - diff_z // 2))

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)

        x = self.final_conv(x)
        return x

def normalize_image(image):
    image_min = image.min()
    image_max = image.max()
    if image_max > image_min:
        return (image - image_min) / (image_max - image_min)
    else:
        return np.zeros_like(image)


def visualize_output(input_tensor, output_tensor, ground_truth_tensor=None, slice_idx=None):
    input_image = input_tensor.squeeze().cpu().numpy()
    output_image = output_tensor.squeeze().detach().cpu().numpy()
    ground_truth_image = ground_truth_tensor.squeeze().cpu().numpy() if ground_truth_tensor is not None else None

    if slice_idx is None:
        slice_idx = input_image.shape[2] // 2

    input_image = normalize_image(input_image)
    output_image = normalize_image(output_image)
    if ground_truth_image is not None:
        ground_truth_image = normalize_image(ground_truth_image)

    fig, axes = plt.subplots(1, 3 if ground_truth_image is not None else 2, figsize=(18, 6))
    axes[0].imshow(input_image[..., slice_idx], cmap='gray')
    axes[0].set_title('Input Image')
    
    axes[1].imshow(output_image[..., slice_idx], cmap='gray')
    axes[1].set_title('Output Segmentation')

    if ground_truth_image is not None:
        axes[2].imshow(ground_truth_image[..., slice_idx], cmap='gray')
        axes[2].set_title('Ground Truth')

    plt.show()

import numpy as np

def post_process_predictions(predictions, one_hot_sets, mutually_exclusive_sets, threshold=0.5):
    """
    Ensures only one class per one-hot set is set to 1, handles mutually exclusive sets, and applies thresholding for other predictions.
    :param predictions: numpy array of predictions with shape (num_classes,)
    :param one_hot_sets: list of sets containing indices of one-hot encoded variables
    :param mutually_exclusive_sets: list of sets containing indices of mutually exclusive variables
    :param threshold: threshold for binary classification
    """
    num_classes = predictions.shape[0]
    processed_predictions = np.zeros(num_classes)

    # Handle mutually exclusive sets
    for mutually_exclusive_set in mutually_exclusive_sets:
        subset_indices = list(mutually_exclusive_set)
        # Ensure indices are within bounds
        subset_indices = [i for i in subset_indices if i < num_classes]
        if len(subset_indices) > 0:
            # Get subset predictions
            subset_predictions = predictions[subset_indices]
            # Find index of the maximum value in the subset
            if np.max(subset_predictions) > threshold:
                max_index = np.argmax(subset_predictions)
                # Set that index to 1 in the processed predictions
                processed_predictions[subset_indices[max_index]] = 1
                # Set other indices in the subset to 0
                for idx in subset_indices:
                    if idx != subset_indices[max_index]:
                        processed_predictions[idx] = 0
        else:
            print(f"Warning: Empty or invalid subset for mutually exclusive set {mutually_exclusive_set}")

    # Handle one-hot encoded sets
    for one_hot_set in one_hot_sets:
        subset_indices = list(one_hot_set)
        # Ensure indices are within bounds
        subset_indices = [i for i in subset_indices if i < num_classes]
        if len(subset_indices) > 0:
            # Get subset predictions
            subset_predictions = predictions[subset_indices]
            # If all predictions in the subset are below threshold, skip setting any value
            if np.max(subset_predictions) > threshold:
                # Find index of the maximum value in the subset
                max_index = np.argmax(subset_predictions)
                # Set that index to 1 in the processed predictions
                processed_predictions[subset_indices[max_index]] = 1
                # Set other indices in the subset to 0
                for idx in subset_indices:
                    if idx != subset_indices[max_index]:
                        processed_predictions[idx] = 0
        else:
            print(f"Warning: Empty or invalid subset for set {one_hot_set}")

    # Apply thresholding for the rest of the predictions
    for i in range(num_classes):
        if i not in [index for one_hot_set in one_hot_sets for index in one_hot_set] and \
           i not in [index for mutually_exclusive_set in mutually_exclusive_sets for index in mutually_exclusive_set]:
            processed_predictions[i] = int(predictions[i] > threshold)

    return processed_predictions
# Define the Data Entry Class
class PatientDataEntry(QWidget):
    data_submitted = pyqtSignal(pd.DataFrame)

    def __init__(self):
        super().__init__()
        self.df_mri = pd.DataFrame()  # Initialize as an empty DataFrame
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        self.age_entry = QLineEdit()
        self.add_field(layout, "Age:", self.age_entry)

        self.age_of_onset_entry = QLineEdit()
        self.add_field(layout, "Age of onset:", self.age_of_onset_entry)

        self.edss_entry = QLineEdit()
        self.add_field(layout, "EDSS:", self.edss_entry)

        self.gender_var = QComboBox()
        self.gender_var.addItems(['F', 'M', 'N'])
        self.add_field(layout, "Gender:", self.gender_var)

        self.time_diff_var = QComboBox()
        self.time_diff_var.addItems(['Yes', 'No'])
        self.add_field(layout, "Does the time difference between MRI acquisition and EDSS < two months:", self.time_diff_var)

        self.tr_entry = QLineEdit()
        self.add_field(layout, "Repetition Time (TR):", self.tr_entry)

        self.te_entry = QLineEdit()
        self.add_field(layout, "Echo Time (TE):", self.te_entry)

        self.slice_thickness_entry = QLineEdit()
        self.add_field(layout, "Slice Thickness:", self.slice_thickness_entry)

        self.spacing_between_slices_entry = QLineEdit()
        self.add_field(layout, "Spacing Between Slices:", self.spacing_between_slices_entry)

        submit_button = QPushButton("Submit")
        submit_button.clicked.connect(self.submit)
        layout.addWidget(submit_button)

        self.setLayout(layout)
        self.setWindowTitle('Patient Data Entry')

    def add_field(self, layout, label_text, widget):
        layout_h = QHBoxLayout()
        label = QLabel(label_text)
        layout_h.addWidget(label)
        layout_h.addWidget(widget)
        layout.addLayout(layout_h)

    def submit(self):
        try:
            # Collect input values
            age = int(self.age_entry.text())
            age_of_onset = int(self.age_of_onset_entry.text())
            if age_of_onset > age:
                raise ValueError("Age of onset cannot be greater than Age.")

            edss = float(self.edss_entry.text())

            gender = self.gender_var.currentText()
            gender_f = int(gender == 'F')
            gender_m = int(gender == 'M')
            gender_n = int(gender == 'N')

            time_diff = self.time_diff_var.currentText()
            time_diff_yes = int(time_diff == 'Yes')
            time_diff_no = int(time_diff == 'No')

            tr = int(self.tr_entry.text())
            te = float(self.te_entry.text())
            slice_thickness = float(self.slice_thickness_entry.text())
            spacing_between_slices = float(self.spacing_between_slices_entry.text())

            # Create DataFrame
            data = {
                "Age": [age],
                "Age of onset": [age_of_onset],
                "EDSS": [edss],
                "Gender_F": [gender_f],
                "Gender_M": [gender_m],
                "Gender_N": [gender_n],
                "Does the time difference between MRI acquisition and EDSS < two months_No": [time_diff_no],
                "Does the time difference between MRI acquisition and EDSS < two months_Yes": [time_diff_yes],
                "Repetition Time (TR)": [tr],
                "Echo Time (TE)": [te],
                "Slice Thickness": [slice_thickness],
                "Spacing Between Slices": [spacing_between_slices]
            }

            self.df_mri = pd.DataFrame(data)
            self.df_mri = self.df_mri.astype({
                "Age": 'int64',
                "Age of onset": 'int64',
                "EDSS": 'float64',
                "Gender_F": 'int32',
                "Gender_M": 'int32',
                "Gender_N": 'int32',
                "Does the time difference between MRI acquisition and EDSS < two months_No": 'int32',
                "Does the time difference between MRI acquisition and EDSS < two months_Yes": 'int32',
                "Repetition Time (TR)": 'int64',
                "Echo Time (TE)": 'float64',
                "Slice Thickness": 'float64',
                "Spacing Between Slices": 'float64'
            })
            QMessageBox.information(self, "Success", "Data has been collected and saved.")
            print(self.df_mri)  # Debug print statement

            # Emit the signal with the DataFrame
            self.data_submitted.emit(self.df_mri)

        except ValueError as e:
            QMessageBox.warning(self, "Error", f"Invalid input: {e}")

# Main script
if __name__ == '__main__':
    app = QApplication([])
    def on_data_submitted(df_mri):
        # Path to a single .nii file
        image_file = input("Enter the Path of Image file: ")
        train_loader = prepare(image_file)
        print("Data from GUI has been processed.")
        device="cpu"
        # Load a batch of images
        batch = next(iter(train_loader))
        inputs = batch['image'].to(device)
        # Instantiate the model and load pre-trained weights
        model = UNet3D(in_channels=1, out_channels=1).to(device)
        pretrained_model_path ='/Volumes/My Passport/unet3d_final.pth'

        if os.path.exists(pretrained_model_path):
            model.load_state_dict(torch.load(pretrained_model_path))
            print("Loaded pre-trained model")

        # Run the model on the selected MRI scan
        outputs = model(inputs)
        output_probs = torch.sigmoid(outputs)
        output_masks = (output_probs > 0.5).float()

        # Visualize the selected MRI scan along with the output mask
        slice_index = 10  # Change this to visualize different slices
        visualize_output(inputs, output_masks, slice_idx=slice_index)

        # Define one-hot sets and mutually exclusive sets for post-processing
        one_hot_sets = [
            range(0, 3),       # Pyramidal, Cerebella, Brain stem
            range(3, 6),       # Sensory, Sphincters, Visual
            range(6, 9),       # Mental, Speech, Motor System
            range(9, 12),      # Sensory System, Coordination, Gait
            range(12, 15),     # Bowel and bladder function, Mobility, Mental State
            range(15, 18),     # Optic discs, Fields, Nystagmus
            range(18, 21),     # Ocular Movement, Swallowing, Types of Medicines_Avonex
            range(21, 26),     # Types of Medicines_Betaferon, Types of Medicines_Gelenia,
                            # Types of Medicines_Rebif, Types of Medicines_Tysabri
            range(26, 35),     # Presenting Symptom_Balance, Presenting Symptom_Balance &Motor,
                            # Presenting Symptom_Motor, Presenting Symptom_Motor & Behavioural,
                            # Presenting Symptom_Motor & Sensory, Presenting Symptom_Motor & Visual,
                            # Presenting Symptom_Motore, Presenting Symptom_Pain, Presenting Symptom_Sensory
            range(35, 42),     # Presenting Symptom_Sensory & Visual, Presenting Symptom_Sensory & Motor,
                            # Presenting Symptom_Sensory & Visual, Presenting Symptom_Sensory & Visual ,Balance , Motor, Sexual,Fatigue
                            # Presenting Symptom_Sensory &Motor, Presenting Symptom_Visual, Presenting Symptom_Visual & Balance
            range(42, 44)      # Dose the patient has Co-moroidity_No, Dose the patient has Co-moroidity_Yes
        ]

        # Define mutually exclusive sets
        mutually_exclusive_sets = [
            range(42, 44)  # Dose the patient has Co-moroidity_No, Dose the patient has Co-moroidity_Yes
        ]
        model_load_path = '/Volumes/My Passport/full_model.pth'
        model = torch.load(model_load_path)
        model.eval()  # Set the model to evaluation mode
        print("Loaded full model")
        combined_dataset = CustomDataset(output_masks, df_mri)
        combined_loader = DataLoader(combined_dataset, batch_size=1, shuffle=False)

        # Process predictions
        with torch.no_grad():
            for batch in combined_loader:
                demographics = batch['mri'].to(device)
                outputs = model(output_masks, demographics) 
                outputs = torch.sigmoid(outputs).cpu().numpy()

                # Post-process predictions to ensure only one class per one-hot set
                outputs = post_process_predictions(outputs[0], one_hot_sets,mutually_exclusive_sets)  # Process the first batch
                break  # Exit after processing the first batch

        # Define the labels
        labels = [
            "Pyramidal", "Cerebella", "Brain stem", "Sensory", "Sphincters", "Visual",
            "Mental", "Speech", "Motor System", "Sensory System", "Coordination", "Gait",
            "Bowel and bladder function", "Mobility", "Mental State", "Optic discs", "Fields",
            "Nystagmus", "Ocular Movement", "Swallowing", "Types of Medicines_Avonex",
            "Types of Medicines_Betaferon", "Types of Medicines_Gelenia", "Types of Medicines_Rebif",
            "Types of Medicines_Tysabri", "Presenting Symptom_Balance", "Presenting Symptom_Balance &Motor",
            "Presenting Symptom_Motor", "Presenting Symptom_Motor & Behavioural", "Presenting Symptom_Motor & Sensory",
            "Presenting Symptom_Motor & Visual", "Presenting Symptom_Motore", "Presenting Symptom_Pain",
            "Presenting Symptom_Sensory", "Presenting Symptom_Sensory & Visual", "Presenting Symptom_Sensory & Motor",
            "Presenting Symptom_Sensory & Visual", "Presenting Symptom_Sensory & Visual ,Balance , Motor, Sexual,Fatigue",
            "Presenting Symptom_Sensory &Motor", "Presenting Symptom_Visual", "Presenting Symptom_Visual & Balance",
            "Dose the patient has Co-moroidity_No", "Dose the patient has Co-moroidity_Yes"
        ]

        # Create the report output
        report_output = "Predictions:\n"
        for i, pred in enumerate(outputs):
            report_output += f"{labels[i]}: {pred}\n"

        report_output += "\nMRI Data:\n"
        # Assuming df_mri is a DataFrame with the MRI data
        for column in df_mri.columns:
            value = df_mri[column].values[0]
            report_output += f"{column}: {value}\n"

        genai.configure(api_key='YOUR_API_KEY')

        def upload_to_gemini(path, mime_type=None):
            """Uploads the given file to Gemini.

            See https://ai.google.dev/gemini-api/docs/prompting_with_media
            """
            file = genai.upload_file(path, mime_type=mime_type)
            print(f"Uploaded file '{file.display_name}' as: {file.uri}")
            return file

        # Create the model
        # See https://ai.google.dev/api/python/google/generativeai/GenerativeModel
        generation_config = {
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
        }

        model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config=generation_config,
        # safety_settings = Adjust safety settings
        # See https://ai.google.dev/gemini-api/docs/safety-settings
        )

        # TODO Make these files available on the local file system
        # You may need to update the file paths
        files = [
        upload_to_gemini("/Volumes/My Passport/WhatsApp Image 2024-07-27 at 8.46.46 PM.jpeg", mime_type="image/jpeg"),
        ]

        chat_session = model.start_chat(
        history=[
            {
            "role": "user",
            "parts": [
                "generate a report for brain MRI Scan, for multiple sclerosis detection and classification. Here are the inputs from a neural network model:\nAge                                                                             56.0\nAge of onset                                                                    43.0\nEDSS                                                                             3.0\nGender_F                                                                         1.0\nGender_M                                                                         0.0\nGender_N                                                                         0.0\nDoes the time difference between MRI acquisition and EDSS < two months_No        1.0\nDoes the time difference between MRI acquisition and EDSS < two months_Yes       0.0\nRepetition Time  (TR)                                                         8500.0\nEcho Time (TE)                                                                 106.0\nSlice Thickness                                                                  5.0\nSpacing Between Slices                                                           5.5\nName: 0, dtype: float64\nPyramidal                                                               0\nCerebella                                                               0\nBrain stem                                                              0\nSensory                                                                 0\nSphincters                                                              0\nVisual                                                                  0\nMental                                                                  0\nSpeech                                                                  0\nMotor System                                                            1\nSensory System                                                          1\nCoordination                                                            0\nGait                                                                    0\nBowel and bladder function                                              0\nMobility                                                                0\nMental State                                                            0\nOptic discs                                                             1\nFields                                                                  0\nNystagmus                                                               1\nOcular Movement                                                         0\nSwallowing                                                              0\nTypes of Medicines_Avonex                                               0\nTypes of Medicines_Betaferon                                            0\nTypes of Medicines_Gelenia                                              1\nTypes of Medicines_Rebif                                                0\nTypes of Medicines_Tysabri                                              0\nPresenting Symptom_Balance                                              0\nPresenting Symptom_Balance &Motor                                       0\nPresenting Symptom_Motor                                                1\nPresenting Symptom_Motor & Behavioural                                  0\nPresenting Symptom_Motor & Sensory                                      0\nPresenting Symptom_Motor & Visual                                       0\nPresenting Symptom_Motore                                               0\nPresenting Symptom_Pain                                                 0\nPresenting Symptom_Sensory                                              0\nPresenting Symptom_Sensory  & Visual                                    0\nPresenting Symptom_Sensory & Motor                                      0\nPresenting Symptom_Sensory & Visual                                     0\nPresenting Symptom_Sensory & Visual ,Balance , Motor, Sexual,Fatigue    0\nPresenting Symptom_Sensory &Motor                                       0\nPresenting Symptom_Visual                                               0\nPresenting Symptom_Visual & Balance                                     0\n Dose the patient has Co-moroidity_No                                   1\n Dose the patient has Co-moroidity_Yes                                  0",
            ],
            },
            {
            "role": "model",
            "parts": [
                "## Brain MRI Report for Multiple Sclerosis Detection and Classification\n\n**Patient Information:**\n\n* Age: 56 years old\n* Age of onset: 43 years old\n* Gender: Female\n* EDSS: 3.0\n* Time difference between MRI acquisition and EDSS assessment: Greater than 2 months\n\n**MRI Acquisition Parameters:**\n\n* Repetition Time (TR): 8500 ms\n* Echo Time (TE): 106 ms\n* Slice Thickness: 5 mm\n* Spacing Between Slices: 5.5 mm\n\n**Clinical Presentation:**\n\n* **Presenting Symptoms:** Motor impairment\n* **Co-morbidities:** None\n* **Medications:** Gelenia\n\n**Neurological Examination Findings:**\n\n* **Motor System:** Impaired\n* **Sensory System:** Impaired\n* **Optic Discs:** Abnormal\n* **Nystagmus:** Present\n\n**Neural Network Model Predictions:**\n\n* **Lesion Location:**\n    * **Pyramidal:** Absent\n    * **Cerebella:** Absent\n    * **Brain Stem:** Absent\n    * **Sensory:** Absent\n    * **Sphincters:** Absent\n    * **Visual:** Absent\n    * **Mental:** Absent\n    * **Speech:** Absent\n    * **Motor System:** Present\n    * **Sensory System:** Present\n    * **Coordination:** Absent\n    * **Gait:** Absent\n    * **Bowel and bladder function:** Absent\n    * **Mobility:** Absent\n    * **Mental State:** Absent\n    * **Optic discs:** Present\n    * **Fields:** Absent\n    * **Nystagmus:** Present\n    * **Ocular Movement:** Absent\n    * **Swallowing:** Absent\n\n**Diagnosis:**\n\nBased on the clinical presentation, neurological examination findings, and neural network model predictions, the patient is likely to have **Multiple Sclerosis**. The presence of lesions in the motor and sensory systems, as well as optic discs, strongly supports this diagnosis. \n\n**Recommendations:**\n\n* Further investigation with more specific MRI sequences may be considered to confirm the diagnosis and assess lesion characteristics.\n* Consultation with a neurologist for further evaluation and management of the patient's condition is recommended.\n\n**Note:** This report is based on the provided information and neural network model predictions. It should not be considered a definitive diagnosis and should be interpreted in conjunction with a thorough clinical evaluation by a qualified healthcare professional. \n",
            ],
            },
            {
            "role": "user",
            "parts": [
                files[0],
                "This is a sample mri scan report. in this you need to add the patient demographics and other essential details for Multiple sclerosis. ok? dont generate anything wait for my input",
            ],
            },
            {
            "role": "model",
            "parts": [
                "Okay, I'm ready. Please provide the patient demographics and other essential details for Multiple sclerosis, and I will create a comprehensive MRI report. \n",
            ],
            },
            {
            "role": "user",
            "parts": [
                "Age                                                                             56.0\nAge of onset                                                                    43.0\nEDSS                                                                             3.0\nGender_F                                                                         1.0\nGender_M                                                                         0.0\nGender_N                                                                         0.0\nDoes the time difference between MRI acquisition and EDSS < two months_No        1.0\nDoes the time difference between MRI acquisition and EDSS < two months_Yes       0.0\nRepetition Time  (TR)                                                         8500.0\nEcho Time (TE)                                                                 106.0\nSlice Thickness                                                                  5.0\nSpacing Between Slices                                                           5.5\nPyramidal                                                               0\nCerebella                                                               0\nBrain stem                                                              0\nSensory                                                                 0\nSphincters                                                              0\nVisual                                                                  0\nMental                                                                  0\nSpeech                                                                  0\nMotor System                                                            1\nSensory System                                                          1\nCoordination                                                            0\nGait                                                                    0\nBowel and bladder function                                              0\nMobility                                                                0\nMental State                                                            0\nOptic discs                                                             1\nFields                                                                  0\nNystagmus                                                               1\nOcular Movement                                                         0\nSwallowing                                                              0\nTypes of Medicines_Avonex                                               0\nTypes of Medicines_Betaferon                                            0\nTypes of Medicines_Gelenia                                              1\nTypes of Medicines_Rebif                                                0\nTypes of Medicines_Tysabri                                              0\nPresenting Symptom_Balance                                              0\nPresenting Symptom_Balance &Motor                                       0\nPresenting Symptom_Motor                                                1\nPresenting Symptom_Motor & Behavioural                                  0\nPresenting Symptom_Motor & Sensory                                      0\nPresenting Symptom_Motor & Visual                                       0\nPresenting Symptom_Motore                                               0\nPresenting Symptom_Pain                                                 0\nPresenting Symptom_Sensory                                              0\nPresenting Symptom_Sensory  & Visual                                    0\nPresenting Symptom_Sensory & Motor                                      0\nPresenting Symptom_Sensory & Visual                                     0\nPresenting Symptom_Sensory & Visual ,Balance , Motor, Sexual,Fatigue    0\nPresenting Symptom_Sensory &Motor                                       0\nPresenting Symptom_Visual                                               0\nPresenting Symptom_Visual & Balance                                     0\n Dose the patient has Co-moroidity_No                                   1\n Dose the patient has Co-moroidity_Yes                                  0",
            ],
            },
            {
            "role": "model",
            "parts": [
                "## Brain MRI Report: Multiple Sclerosis Evaluation\n\n**Patient Information:**\n\n* **Name:** [Please provide patient's name]\n* **Age:** 56 years old\n* **Sex:** Female\n* **Age of Onset:** 43 years old\n* **EDSS Score:** 3.0\n* **Time since EDSS Assessment:** Greater than two months\n* **Presenting Symptoms:** Motor impairment\n* **Co-morbidities:** None\n* **Medications:** Gelenia\n\n**MRI Acquisition Parameters:**\n\n* **Sequence:** T1W/T2W mixed\n* **Repetition Time (TR):** 8500 ms\n* **Echo Time (TE):** 106 ms\n* **Slice Thickness:** 5.0 mm\n* **Spacing Between Slices:** 5.5 mm\n\n**Findings:**\n\nA large T1W/T2W mixed intense lesion (11.1 x 9.4 x 9.9 mm) is noted involving the pituitary gland, consistent with a pituitary macroadenoma.  \n\n* **Pituitary Gland:**  Abnormal, with the presence of the macroadenoma.\n* **Brain:**  No abnormal signals visualized in the cerebral hemispheres.  The thalami, basal ganglia, and internal capsules appear normal bilaterally. The ventricles and sulci are normal for the patient's age.\n* **Other Structures:**\n    * No evidence of acute infarct, hemorrhage, or mass lesions.\n    * The infundibulum and hypothalamus are normal for the patient's age. \n    * The cerebellum, medulla, pons, and midbrain appear normal. \n    * Both CP angles are normal. The basal cisterns are normal.\n\n**Neurological Examination:**\n\n* **Motor System:** Impaired\n* **Sensory System:** Impaired\n* **Coordination:** Normal\n* **Gait:** Normal\n* **Bowel and bladder function:** Normal\n* **Mobility:** Normal\n* **Mental State:** Normal\n* **Optic Discs:** Abnormal\n* **Fields:** Normal\n* **Nystagmus:** Present\n* **Ocular Movement:** Normal\n* **Swallowing:** Normal\n\n**Interpretation:**\n\nThe MRI findings demonstrate a pituitary macroadenoma, which requires further investigation and potentially a contrast study for evaluation.  The presence of lesions in the motor and sensory systems, along with the abnormal optic discs and nystagmus, are consistent with Multiple Sclerosis.\n\n**Recommendations:**\n\n* **Further Investigation:** A contrast-enhanced MRI scan is recommended to further evaluate the pituitary macroadenoma and better define its characteristics.\n* **Neurological Consultation:** Consultation with a neurologist for further evaluation and management of the patient's condition, including potential treatment options for both the pituitary macroadenoma and Multiple Sclerosis.\n\n**Note:** This report should be interpreted in conjunction with a thorough clinical evaluation by a qualified healthcare professional. It does not replace the expertise of a physician and is intended for informational purposes only. \n",
            ],
            },
            {
            "role": "user",
            "parts": [
                "No, derive something, not just read the report",
            ],
            },
            {
            "role": "model",
            "parts": [
                "You are absolutely right! I apologize for simply regurgitating the information. I'm still under development and learning to interpret and synthesize data. \n\nHere's a more insightful analysis based on the information provided:\n\n**Key Findings:**\n\n1. **Pituitary Macroadenoma:** The presence of a large pituitary macroadenoma is a significant finding. While not directly related to Multiple Sclerosis, it could potentially contribute to neurological symptoms through compression of surrounding structures.\n2. **Multiple Sclerosis Lesions:** The neural network predictions indicate lesions in the motor and sensory systems, which align with the patient's reported symptoms. The abnormal optic discs and nystagmus also strongly support a diagnosis of MS.\n3. **Time Since EDSS Assessment:** The significant time difference between the MRI acquisition and the EDSS assessment suggests the possibility of disease progression. This requires further evaluation to assess the patient's current neurological status.\n\n**Important Considerations:**\n\n* **Differential Diagnoses:** It's important to consider other potential causes for the patient's symptoms, particularly with the presence of the pituitary macroadenoma.\n* **Treatment Planning:** The management plan should consider both the pituitary macroadenoma and the Multiple Sclerosis. The neurologist will need to determine the best course of action for treating each condition separately, as well as the potential interactions between therapies.\n* **Prognosis:** The prognosis for MS is variable and depends on several factors, including the severity of the disease, the location and size of lesions, and the patient's response to treatment. The presence of a pituitary macroadenoma may complicate the treatment and prognosis.\n\n**Next Steps:**\n\n* **Contrast-Enhanced MRI:**  This is essential to further characterize the pituitary macroadenoma, assess its extent, and determine if it's compressing surrounding structures.\n* **Neurological Consultation:** A comprehensive neurological evaluation is crucial to assess the patient's current neurological function, monitor disease progression, and discuss appropriate treatment options for both conditions. \n\n**Summary:**\n\nThe MRI findings are consistent with a diagnosis of Multiple Sclerosis, supported by the patient's clinical presentation. However, the presence of the pituitary macroadenoma requires further evaluation and may complicate the management and prognosis. A thorough neurological consultation is essential to develop a comprehensive treatment plan. \n",
            ],
            },
            {
            "role": "user",
            "parts": [
                "Age                                                                             29.00\nAge of onset                                                                    19.00\nEDSS                                                                             1.50\nGender_F                                                                         1.00\nGender_M                                                                         0.00\nGender_N                                                                         0.00\nDoes the time difference between MRI acquisition and EDSS < two months_No        1.00\nDoes the time difference between MRI acquisition and EDSS < two months_Yes       0.00\nRepetition Time  (TR)                                                         9000.00\nEcho Time (TE)                                                                  82.00\nSlice Thickness                                                                  5.00\nSpacing Between Slices                                                           6.75\nName: 1, dtype: float64\nPyramidal                                                               0\nCerebella                                                               0\nBrain stem                                                              0\nSensory                                                                 0\nSphincters                                                              0\nVisual                                                                  0\nMental                                                                  0\nSpeech                                                                  0\nMotor System                                                            1\nSensory System                                                          1\nCoordination                                                            0\nGait                                                                    0\nBowel and bladder function                                              0\nMobility                                                                1\nMental State                                                            0\nOptic discs                                                             0\nFields                                                                  0\nNystagmus                                                               0\nOcular Movement                                                         0\nSwallowing                                                              0\nTypes of Medicines_Avonex                                               0\nTypes of Medicines_Betaferon                                            0\nTypes of Medicines_Gelenia                                              1\nTypes of Medicines_Rebif                                                0\nTypes of Medicines_Tysabri                                              0\nPresenting Symptom_Balance                                              0\nPresenting Symptom_Balance &Motor                                       0\nPresenting Symptom_Motor                                                0\nPresenting Symptom_Motor & Behavioural                                  0\nPresenting Symptom_Motor & Sensory                                      0\nPresenting Symptom_Motor & Visual                                       0\nPresenting Symptom_Motore                                               0\nPresenting Symptom_Pain                                                 0\nPresenting Symptom_Sensory                                              1\nPresenting Symptom_Sensory  & Visual                                    0\nPresenting Symptom_Sensory & Motor                                      0\nPresenting Symptom_Sensory & Visual                                     0\nPresenting Symptom_Sensory & Visual ,Balance , Motor, Sexual,Fatigue    0\nPresenting Symptom_Sensory &Motor                                       0\nPresenting Symptom_Visual                                               0\nPresenting Symptom_Visual & Balance                                     0\n Dose the patient has Co-moroidity_No                                   1\n Dose the patient has Co-moroidity_Yes                                  0",
            ],
            },
            {
            "role": "model",
            "parts": [
                "## Brain MRI Report: Multiple Sclerosis Evaluation\n\n**Patient Information:**\n\n* **Name:** [Please provide patient's name]\n* **Age:** 29 years old\n* **Sex:** Female\n* **Age of Onset:** 19 years old\n* **EDSS Score:** 1.5\n* **Time since EDSS Assessment:** Greater than two months\n* **Presenting Symptoms:** Sensory impairment\n* **Co-morbidities:** None\n* **Medications:** Gelenia\n\n**MRI Acquisition Parameters:**\n\n* **Sequence:** T1W/T2W mixed \n* **Repetition Time (TR):** 9000 ms\n* **Echo Time (TE):** 82 ms\n* **Slice Thickness:** 5.0 mm\n* **Spacing Between Slices:** 6.75 mm\n\n**Findings:**\n\n* **Brain:** No significant abnormalities detected in the cerebral hemispheres, including the thalami, basal ganglia, and internal capsules. The ventricles and sulci appear normal for the patient's age.  There is no evidence of acute infarct, hemorrhage, or mass lesions.\n* **Pituitary Gland:** The pituitary gland appears normal.\n* **Other Structures:**\n    * The infundibulum and hypothalamus are normal for the patient's age. \n    * The cerebellum, medulla, pons, and midbrain appear normal.\n    * Both CP angles are normal. The basal cisterns are normal.\n\n**Neurological Examination:**\n\n* **Motor System:** Impaired\n* **Sensory System:** Impaired\n* **Coordination:** Normal\n* **Gait:** Normal\n* **Bowel and bladder function:** Normal\n* **Mobility:** Impaired\n* **Mental State:** Normal\n* **Optic Discs:** Normal\n* **Fields:** Normal\n* **Nystagmus:** Absent\n* **Ocular Movement:** Normal\n* **Swallowing:** Normal\n\n**Interpretation:**\n\nThe MRI findings are consistent with a diagnosis of Multiple Sclerosis, supported by the patient's reported symptoms and clinical presentation.  The presence of lesions in the motor and sensory systems, coupled with impaired mobility, aligns with the typical patterns seen in MS. However, the absence of lesions in the optic discs and absence of nystagmus are less common in MS.  \n\n**Important Considerations:**\n\n* **Differential Diagnoses:** While MS is the most likely diagnosis, it's important to consider other potential causes for sensory impairment, particularly in a patient with a relatively low EDSS score.  \n* **Disease Activity:** The lack of visible lesions in the optic discs and the absence of nystagmus may indicate that the patient is experiencing a less active phase of MS.  \n* **Follow-Up:**  The patient should be closely monitored with repeat MRIs and neurological evaluations to assess for disease progression and adjust treatment accordingly.\n\n**Recommendations:**\n\n* **Neurological Consultation:** A thorough neurological evaluation is crucial to assess the patient's current neurological function, monitor disease progression, and discuss appropriate treatment options for MS.  \n* **Repeat MRI:** Consider follow-up MRI scans, potentially with different sequences (e.g., contrast-enhanced T1W), to monitor for any new lesions or changes in existing lesions.\n* **Treatment:**  The neurologist will determine the best course of disease-modifying therapy (DMT) for this patient based on her EDSS score, disease activity, and individual needs.  \n\n**Summary:**\n\nThe MRI findings and clinical presentation support a diagnosis of Multiple Sclerosis, however, further monitoring and neurological follow-up are essential to confirm the diagnosis and assess disease activity. The absence of optic nerve lesions and nystagmus suggests a less active disease course. \n",
                "Okay, I understand. I will ensure that all the sections you mentioned are included in every report I generate from now on, without any additional commentary. Please provide me with the input data, and I will generate a comprehensive report. \n",
                "## Brain MRI Report: Multiple Sclerosis Evaluation\n\n**Patient Information:**\n\n* **Name:** [Please provide patient's name]\n* **Age:** 21 years old\n* **Sex:** Female\n* **Age of Onset:** 17 years old\n* **EDSS Score:** 2.0\n* **Time since EDSS Assessment:** Greater than two months\n* **Presenting Symptoms:** Visual impairment, gait disturbance, nystagmus\n* **Co-morbidities:** None\n* **Medications:** Tysabri\n\n**MRI Acquisition Parameters:**\n\n* **Sequence:** T1W/T2W mixed \n* **Repetition Time (TR):** 6000 ms\n* **Echo Time (TE):** 106 ms\n* **Slice Thickness:** 5.0 mm\n* **Spacing Between Slices:** 5.5 mm\n\n**Findings:**\n\n* **Brain:** \n    * **Pyramidal:**  Lesion present\n    * **Cerebella:** No lesion\n    * **Brain Stem:** No lesion\n    * **Sensory:** No lesion\n    * **Sphincters:** Lesion present\n    * **Visual:** Lesion present\n    * **Mental:** No lesion\n    * **Speech:** No lesion\n    * **Motor System:** Lesion present \n    * **Sensory System:** No lesion\n    * **Coordination:** No lesion\n    * **Gait:** No lesion\n    * **Bowel and bladder function:** No lesion\n    * **Mobility:** No lesion\n    * **Mental State:** No lesion\n    * **Optic discs:** No lesion\n    * **Fields:** No lesion\n    * **Nystagmus:** No lesion\n    * **Ocular Movement:** Lesion present\n    * **Swallowing:** No lesion\n* **Pituitary Gland:**  The pituitary gland appears normal.\n\n**Neurological Examination:**\n\n* **Motor System:** Impaired\n* **Sensory System:** Normal\n* **Coordination:** Normal\n* **Gait:** Impaired\n* **Bowel and bladder function:** Normal\n* **Mobility:** Normal\n* **Mental State:** Normal\n* **Optic discs:** Normal\n* **Fields:** Normal\n* **Nystagmus:** Present\n* **Ocular Movement:** Impaired\n* **Swallowing:** Normal\n\n**Interpretation:**\n\nThe MRI findings show lesions in the pyramidal, sphincter, visual, motor, and ocular movement regions of the brain.  These findings, coupled with the patient's reported symptoms and neurological examination, are consistent with a diagnosis of Multiple Sclerosis. The presence of lesions in the visual system and the patient's reported visual symptoms are particularly noteworthy.  \n\n**Important Considerations:**\n\n* **Disease Activity:** The patient is experiencing a relatively active phase of MS with several lesions present. \n* **Treatment:** The current treatment with Tysabri is likely appropriate given the patient's disease activity and the presence of lesions in the visual pathway.\n* **Prognosis:** The long-term prognosis for MS is variable and depends on many factors, including the severity of the disease, the location and size of lesions, and the patient's response to treatment. \n\n**Recommendations:**\n\n* **Neurological Consultation:** A thorough neurological evaluation is crucial to assess the patient's current neurological function, monitor disease progression, and discuss appropriate treatment options for MS.  \n* **Repeat MRI:**  Consider follow-up MRI scans, potentially with different sequences (e.g., contrast-enhanced T1W), to monitor for any new lesions or changes in existing lesions.\n* **Treatment Optimization:** The neurologist should assess the patient's response to Tysabri and consider adjusting treatment strategies based on the patient's individual needs and disease progression.\n\n**Summary:**\n\nThe MRI findings are consistent with a diagnosis of Multiple Sclerosis, with evidence of active disease activity. The patient's presenting symptoms and neurological examination findings further support this diagnosis.  Continued neurological follow-up and treatment optimization are essential for managing the patient's condition. \n\n\n",
                "Understood. I will ensure all those sections are included in every report from now on. I will also incorporate the \"Does the patient have abnormality at the below neurological examination (1=Abnormal & 0=Normal)\" section into each report. \n\nPlease provide me with the input data! \n",
            ],
            },
            {
            "role": "user",
            "parts": [
                "the title, Patient Information, MRI acquisition Parameters, Findings, Neurological Examination, Interpretation, Important Considerations, Recommendations, Summary, all SHOULD be there for every input I give. dont say anything until I give you some input.\n\nand also\nDoes the patient has abnormality at the below neurological examination (1=Abnormal & 0=Normal)\nand as for brain MRI scans, its taken in FLAIR. ",
            ],
            },
            {
            "role": "model",
            "parts": [
                "Okay, I understand. I will ensure all those sections are included in every report from now on, without any additional commentary. Please provide me with the input data, and I will generate a comprehensive report. \n",
            ],
            },
            {
            "role": "user",
            "parts": [
                "Age                                                                             21.0\nAge of onset                                                                    17.0\nEDSS                                                                             2.0\nGender_F                                                                         1.0\nGender_M                                                                         0.0\nGender_N                                                                         0.0\nDoes the time difference between MRI acquisition and EDSS < two months_No        1.0\nDoes the time difference between MRI acquisition and EDSS < two months_Yes       0.0\nRepetition Time  (TR)                                                         6000.0\nEcho Time (TE)                                                                 106.0\nSlice Thickness                                                                  5.0\nSpacing Between Slices                                                           5.5\nPyramidal                                                               1\nCerebella                                                               0\nBrain stem                                                              0\nSensory                                                                 0\nSphincters                                                              1\nVisual                                                                  1\nMental                                                                  0\nSpeech                                                                  0\nMotor System                                                            1\nSensory System                                                          0\nCoordination                                                            0\nGait                                                                    1\nBowel and bladder function                                              0\nMobility                                                                0\nMental State                                                            0\nOptic discs                                                             0\nFields                                                                  0\nNystagmus                                                               1\nOcular Movement                                                         1\nSwallowing                                                              0\nTypes of Medicines_Avonex                                               0\nTypes of Medicines_Betaferon                                            0\nTypes of Medicines_Gelenia                                              0\nTypes of Medicines_Rebif                                                0\nTypes of Medicines_Tysabri                                              1\nPresenting Symptom_Balance                                              0\nPresenting Symptom_Balance &Motor                                       0\nPresenting Symptom_Motor                                                0\nPresenting Symptom_Motor & Behavioural                                  0\nPresenting Symptom_Motor & Sensory                                      0\nPresenting Symptom_Motor & Visual                                       0\nPresenting Symptom_Motore                                               0\nPresenting Symptom_Pain                                                 0\nPresenting Symptom_Sensory                                              0\nPresenting Symptom_Sensory  & Visual                                    0\nPresenting Symptom_Sensory & Motor                                      0\nPresenting Symptom_Sensory & Visual                                     0\nPresenting Symptom_Sensory & Visual ,Balance , Motor, Sexual,Fatigue    0\nPresenting Symptom_Sensory &Motor                                       0\nPresenting Symptom_Visual                                               1\nPresenting Symptom_Visual & Balance                                     0\n Dose the patient has Co-moroidity_No                                   1\n Dose the patient has Co-moroidity_Yes                                  0",
            ],
            },
            {
            "role": "model",
            "parts": [
                "## Brain MRI Report: Multiple Sclerosis Evaluation\n\n**Patient Information:**\n\n* **Name:** [Please provide patient's name]\n* **Age:** 21 years old\n* **Sex:** Female\n* **Age of Onset:** 17 years old\n* **EDSS Score:** 2.0\n* **Time since EDSS Assessment:** Greater than two months\n* **Presenting Symptoms:** Visual impairment, gait disturbance, nystagmus\n* **Co-morbidities:** None\n* **Medications:** Tysabri\n\n**MRI Acquisition Parameters:**\n\n* **Sequence:** FLAIR \n* **Repetition Time (TR):** 6000 ms\n* **Echo Time (TE):** 106 ms\n* **Slice Thickness:** 5.0 mm\n* **Spacing Between Slices:** 5.5 mm\n\n**Findings:**\n\n* **Brain:** \n    * **Pyramidal:**  Lesion present\n    * **Cerebella:** No lesion\n    * **Brain Stem:** No lesion\n    * **Sensory:** No lesion\n    * **Sphincters:** Lesion present\n    * **Visual:** Lesion present\n    * **Mental:** No lesion\n    * **Speech:** No lesion\n    * **Motor System:** Lesion present \n    * **Sensory System:** No lesion\n    * **Coordination:** No lesion\n    * **Gait:** No lesion\n    * **Bowel and bladder function:** No lesion\n    * **Mobility:** No lesion\n    * **Mental State:** No lesion\n    * **Optic discs:** No lesion\n    * **Fields:** No lesion\n    * **Nystagmus:** No lesion\n    * **Ocular Movement:** Lesion present\n    * **Swallowing:** No lesion\n* **Pituitary Gland:**  The pituitary gland appears normal.\n\n**Neurological Examination:**\n\n* **Motor System:** Impaired (1)\n* **Sensory System:** Normal (0)\n* **Coordination:** Normal (0)\n* **Gait:** Impaired (1)\n* **Bowel and bladder function:** Normal (0)\n* **Mobility:** Normal (0)\n* **Mental State:** Normal (0)\n* **Optic discs:** Normal (0)\n* **Fields:** Normal (0)\n* **Nystagmus:** Present (1)\n* **Ocular Movement:** Impaired (1)\n* **Swallowing:** Normal (0)\n\n**Does the patient have abnormality at the below neurological examination (1=Abnormal & 0=Normal):**\n\n* **Motor System:** 1\n* **Sensory System:** 0\n* **Coordination:** 0\n* **Gait:** 1\n* **Bowel and bladder function:** 0\n* **Mobility:** 0\n* **Mental State:** 0\n* **Optic discs:** 0\n* **Fields:** 0\n* **Nystagmus:** 1\n* **Ocular Movement:** 1\n* **Swallowing:** 0\n\n**Interpretation:**\n\nThe MRI findings show lesions in the pyramidal, sphincter, visual, motor, and ocular movement regions of the brain.  These findings, coupled with the patient's reported symptoms and neurological examination, are consistent with a diagnosis of Multiple Sclerosis. The presence of lesions in the visual system and the patient's reported visual symptoms are particularly noteworthy.  \n\n**Important Considerations:**\n\n* **Disease Activity:** The patient is experiencing a relatively active phase of MS with several lesions present. \n* **Treatment:** The current treatment with Tysabri is likely appropriate given the patient's disease activity and the presence of lesions in the visual pathway.\n* **Prognosis:** The long-term prognosis for MS is variable and depends on many factors, including the severity of the disease, the location and size of lesions, and the patient's response to treatment. \n\n**Recommendations:**\n\n* **Neurological Consultation:** A thorough neurological evaluation is crucial to assess the patient's current neurological function, monitor disease progression, and discuss appropriate treatment options for MS.  \n* **Repeat MRI:**  Consider follow-up MRI scans, potentially with different sequences (e.g., contrast-enhanced T1W), to monitor for any new lesions or changes in existing lesions.\n* **Treatment Optimization:** The neurologist should assess the patient's response to Tysabri and consider adjusting treatment strategies based on the patient's individual needs and disease progression.\n\n**Summary:**\n\nThe MRI findings are consistent with a diagnosis of Multiple Sclerosis, with evidence of active disease activity. The patient's presenting symptoms and neurological examination findings further support this diagnosis.  Continued neurological follow-up and treatment optimization are essential for managing the patient's condition. \n",
            ],
            },
            {
            "role": "user",
            "parts": [
                "dont give patient name and as well as dont bluff. Okay?\n",
            ],
            },
            {
            "role": "model",
            "parts": [
                "Okay",
            ],
            },
            {
            "role": "user",
            "parts": [
                "dont give it as 0 and 1. give as normal and abnormal. dont prompt anything now",
            ],
            },
            {
            "role": "model",
            "parts": [
                "Okay, I understand. I will use \"Normal\" and \"Abnormal\" instead of \"0\" and \"1\" in the report.  I'm ready for your input! \n",
            ],
            },
            {
            "role": "user",
            "parts": [
                "Analyse everything in great detail in medical terms. dont prompt now, wait for data",
            ],
            },
            {
            "role": "model",
            "parts": [
                "Okay",
            ],
            },
        ])

        response = chat_session.send_message(report_output)

        output_text=response.text

        html_content = markdown2.markdown(output_text)

        html_template = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Brain MRI Report: Multiple Sclerosis Evaluation</title>
    <style>
        body {{
            font-family: 'Georgia', serif;
            font-size: 16px;
            line-height: 1.6;
            margin: 1in;
            color: #333;
            background-color: #f5f5f5;
        }}
        h1 {{
            font-size: 32px;
            font-weight: bold;
            text-align: center;
            margin-bottom: 1.5em;
            color: #2c3e50;
        }}
        h2 {{
            font-size: 28px;
            font-weight: bold;
            margin-top: 2em;
            margin-bottom: 1em;
            border-bottom: 2px solid #3498db;
            padding-bottom: 0.5em;
            color: #3498db;
        }}
        h3 {{
            font-size: 24px;
            font-weight: bold;
            margin-top: 1.5em;
            margin-bottom: 1em;
            color: #34495e;
        }}
        p, li {{
            margin: 0.7em 0;
            line-height: 1.8;
        }}
        ul, ol {{
            margin: 0;
            padding-left: 25px;
        }}
        .content {{
            background: #ffffff;
            border-radius: 8px;
            padding: 1em;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
        }}
        .footer {{
            text-align: center;
            color: gray;
            margin-top: 3em;
            font-size: 14px;
        }}
        .disclaimer {{
            margin-top: 2em;
            padding: 1em;
            border: 1px solid #e74c3c;
            border-radius: 8px;
            background-color: #fef6f6;
            color: #e74c3c;
            font-size: 14px;
            text-align: center;
            font-style: italic;
            line-height: 1.6;
        }}
        .disclaimer strong {{
            font-weight: bold;
            color: #c0392b;
        }}
        .section-title {{
            font-size: 18px;
            font-weight: bold;
            margin-bottom: 0.5em;
            color: #2c3e50;
        }}
        .highlight {{
            background-color: #ffff99; /* Light yellow */
            padding: 0.2em;
            border-radius: 4px;
        }}
    </style>
</head>
<body>
    <h1>Brain MRI Report: Multiple Sclerosis Evaluation</h1>
    <div class="content">
        {html_content.replace('Abnormal', '<span class="highlight">Abnormal</span>')}
    </div>
    <div class="disclaimer">
        <p><strong>Disclaimer:</strong> This medical report, including the lesion segmentation, is generated by AI and should not be taken as medical advice. Always consult a qualified healthcare professional for medical diagnosis and treatment. The AI-generated segmentation is for informational purposes only and is not a substitute for professional medical evaluation.</p>
    </div>
    <div class="footer">Powered by Gemini</div>
</body>
</html>
"""

        with open('/Volumes/My Passport/Brain_MRI_Report.html', 'w', encoding='utf-8') as file:
            file.write(html_template)

        print("HTML file has been created and saved as 'Brain_MRI_Report.html'")


    # Create and show the PatientDataEntry widget
    ex = PatientDataEntry()
    # Connect the signal to the handler function
    ex.data_submitted.connect(on_data_submitted)
    ex.show()

    # Run the application event loop
    app.exec_()
    
