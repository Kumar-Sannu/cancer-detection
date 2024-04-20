#Data download
from tcia_utils import nbia



# Print information about the downloaded series

datadir = '/home/group15/demoproject/axial_data/kidney'
cart_name = "nbia-34101712928777604 "
cart_data = nbia.getSharedCart(cart_name)
df = nbia.downloadSeries(cart_data, format="df", path = datadir)
print(df)


#use of segmentator model
import os
import shutil
import torch
from totalsegmentator.python_api import totalsegmentator

torch.cuda.set_device(0)

def segment_subfolder(subfolder_path, output_folder):
    """
    Function to perform segmentation on a subfolder and save results.
    """
    try:
        # Create a temporary directory for segmented output
        temp_output_folder = os.path.join(output_folder, os.path.basename(subfolder_path) + "_temp")
        os.makedirs(temp_output_folder, exist_ok=True)
        
        # Perform segmentation using totalsegmentator
        segmentation = totalsegmentator(subfolder_path, temp_output_folder, roi_subset=["pancreas", "kidney_right", "kidney_left", "lung_upper_lobe_left", "lung_lower_lobe_left", "lung_upper_lobe_right", "lung_middle_lobe_right", "lung_lower_lobe_right"], device='cuda')

        # Move the segmentation output from the temporary directory to the final output folder
        output_filepath = os.path.join(output_folder, os.path.basename(subfolder_path) + ".nii.gz")
        shutil.move(os.path.join(temp_output_folder, "segmentation.nii.gz"), output_filepath)

        print(f"Segmentation saved to: {output_filepath}")
    except Exception as e:
        print(f"Error processing {subfolder_path}: {str(e)}")

def process_all_subfolders(input_folder, output_folder):
    """
    Function to process all subfolders in the input folder.
    """
    # Iterate over subfolders in the input folder
    for subfolder_name in os.listdir(input_folder):
        subfolder_path = os.path.join(input_folder, subfolder_name)
        
        # Check if the item in the input folder is a subfolder
        if os.path.isdir(subfolder_path):
            # Apply segmentation function to each subfolder
            segment_subfolder(subfolder_path, output_folder)

if __name__ == "__main__":
    # Specify input and output directories
    input_folder = "/home/group15/demoproject/Kartika/ultidataset/kidney"
    output_folder = "/home/group15/demoproject/Kartika/allorgseg/kidney"

    # Run the segmentation process for all subfolders
    process_all_subfolders(input_folder, output_folder)


#Data preprocessing
import os
import numpy as np
import nibabel as nib
from tqdm import tqdm
import multiprocessing
import statistics
import torch
import torch.nn.functional as F

import os
import SimpleITK as sitk
import numpy as np
import nibabel as nib
from tqdm import tqdm
import multiprocessing
import statistics

import os
import SimpleITK as sitk
import numpy as np
import nibabel as nib
from tqdm import tqdm
import multiprocessing
import statistics

def calculate_slice_counts(input_folders):
    slice_counts = []
    for input_folder in input_folders:
        for root, dirs, files in os.walk(input_folder):
            for file in files:
                if file.endswith(".nii") or file.endswith(".nii.gz"):
                    file_path = os.path.join(root, file)
                    try:
                        if os.path.getsize(file_path) == 0:
                            continue  # Skip empty files
                        nii_image = nib.load(file_path)
                        num_slices = nii_image.shape[-1]  # Assuming the slices are along the last dimension
                        slice_counts.append(num_slices)
                    except nib.filebasedimages.ImageFileError as e:
                        print(f"Skipping empty file: {file_path}")
    return slice_counts

# Rest of the code remains the same...


# Rest of the code remains the same...


def calculate_target_shape(slice_counts):
    if not slice_counts:
        raise ValueError("Slice counts list is empty")
    mean_count = statistics.mean(slice_counts)
    std_count = statistics.stdev(slice_counts)
    # Assuming the target shape will be the mean +/- one standard deviation
    target_shape = (int(mean_count), int(mean_count + std_count), int(mean_count - std_count))
    return target_shape

def normalize_image(image):
    # Normalize image to fit [0, 1] range
    minimum = np.min(image)
    maximum = np.max(image)
    return (image - minimum) / (maximum - minimum)

def preprocess_image(args):
    input_path, output_path, target_shape = args

    # Load NIfTI image using nibabel
    nii_image = nib.load(input_path)
    np_image = np.array(nii_image.dataobj)

    # Resize the image to the target shape
    resized_image = F.interpolate(torch.from_numpy(np_image).unsqueeze(0).unsqueeze(0).float(), size=target_shape, mode='trilinear', align_corners=True)

    # Check if the shape matches the target shape
    if resized_image.shape[2] != target_shape[0] or resized_image.shape[3] != target_shape[1] or resized_image.shape[4] != target_shape[2]:
        return  # Skip if shape doesn't match

    # Convert back to numpy array
    np_resized_image = resized_image.squeeze(0).squeeze(0).numpy()

    # Normalize the resized image
    np_normalized_image = normalize_image(np_resized_image)

    # Save the normalized image
    nii_resized_image = nib.Nifti1Image(np_normalized_image, affine=nii_image.affine)
    nib.save(nii_resized_image, output_path)


def preprocess_parallel(input_folder, output_folder, target_shape):
    input_files = []
    output_files = []

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Get a list of input files
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.endswith(".nii") or file.endswith(".nii.gz"):
                input_files.append(os.path.join(root, file))
                output_files.append(os.path.join(output_folder, file))

    # Combine input and output files with target shape into a list of tuples
    file_args = [(input_files[i], output_files[i], target_shape) for i in range(len(input_files))]

    # Process images in parallel
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        list(tqdm(pool.imap(preprocess_image, file_args), total=len(input_files), desc="Processing"))


# Define the input folders
input_folder1 = "/home/group15/demoproject/Kartika/ultisegdata/lungs/cancerous"
input_folder2 = "/home/group15/demoproject/Kartika/ultisegdata/lungs/noncancerous"

# Combine input folders for target shape calculation
input_folders_combined = [input_folder1, input_folder2]

# Calculate target shape
slice_counts = calculate_slice_counts(input_folders_combined)
target_shape = calculate_target_shape(slice_counts)
print("Target shape:", target_shape)

# Define the output folders
output_folder1 = "/home/group15/demoproject/Kartika/ppsegdata/lungs/cancerous"
output_folder2 = "/home/group15/demoproject/Kartika/ppsegdata/lungs/noncancerous"
# Run preprocessing on the input NIfTI images
preprocess_parallel(input_folder1, output_folder1, target_shape)
preprocess_parallel(input_folder2, output_folder2, target_shape)




#Classification model

import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
import os
import nibabel as nib  # Assuming you're using NiBabel to read NIfTI fil



# Function to load NIfTI files from a folder
def load_nifti_files(folder):
    data = []
    for file_name in os.listdir(folder):
        file_path = os.path.join(folder, file_name)
        # Assuming the file is in NIfTI format and can be loaded using nibabel
        img = nib.load(file_path)
        # Assuming you need to preprocess the image data in some way (e.g., normalization)
        # You can add your preprocessing steps here
        # For simplicity, we're just getting the image data as a numpy array
        img_data = img.get_fdata()
        # Flatten the image data
        flattened_data = img_data.flatten()
        data.append(flattened_data)
    return np.array(data)
# Path to your cancerous and non-cancerous folders
cancerous_folder = '/home/group15/demoproject/Kartika/unnecessary/unun/larcan'
non_cancerous_folder = '/home/group15/demoproject/Kartika/unnecessary/unun/larnon'

# Load cancerous and non-cancerous images
cancerous_data = load_nifti_files(cancerous_folder)
non_cancerous_data = load_nifti_files(non_cancerous_folder)

# Label cancerous data as 1 and non-cancerous as 0
cancerous_labels = np.ones(len(cancerous_data))
non_cancerous_labels = np.zeros(len(non_cancerous_data))

# Concatenate data and labels
X = np.concatenate((cancerous_data, non_cancerous_data))
y = np.concatenate((cancerous_labels, non_cancerous_labels))

# Handle NaN values
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize SVM model
clf = svm.SVC(kernel='linear')

# Train the model
clf.fit(X_train, y_train)

# Predictions
y_pred_train = clf.predict(X_train)
y_pred_test = clf.predict(X_test)

# Calculate accuracy
train_accuracy = accuracy_score(y_train, y_pred_train)
test_accuracy = accuracy_score(y_test, y_pred_test)

print(f"Training accuracy: {train_accuracy}")
print(f"Testing accuracy: {test_accuracy}")
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, precision_recall_curve

# Confusion matrix
cm = confusion_matrix(y_test, y_pred_test)
print("Confusion Matrix:")
print(cm)

# Accuracy
accuracy = accuracy_score(y_test, y_pred_test)
print("Accuracy:", accuracy)

# Precision
precision = precision_score(y_test, y_pred_test)
print("Precision:", precision)

# Recall
recall = recall_score(y_test, y_pred_test)
print("Recall:", recall)

# F1 Score
f1 = f1_score(y_test, y_pred_test)
print("F1 Score:", f1)

# ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_test)
roc_auc = auc(fpr, tpr)
print("AUC-ROC:", roc_auc)

# Precision-recall curve
precision, recall, _ = precision_recall_curve(y_test, y_pred_test)
pr_auc = auc(recall, precision)
print("AUC-PR:", pr_auc)
import matplotlib.pyplot as plt

# ROC curve plot
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

# Precision-recall curve plot
plt.figure()
plt.plot(recall, precision, color='blue', lw=2, label='Precision-Recall curve (area = %0.2f)' % pr_auc)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower left")
plt.show()
