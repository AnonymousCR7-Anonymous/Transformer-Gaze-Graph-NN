import os
import cv2
import pandas as pd
import mediapipe as mp

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)

# Define paths
input_base_path = ".../MPIIFaceGaze_Landmarks"
subjects = [f'p{str(i).zfill(2)}' for i in range(15)]

# Function to process images and extract landmarks using MediaPipe
def process_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if not results.multi_face_landmarks:
        return None
    landmarks = results.multi_face_landmarks[0].landmark
    return [(lm.x, lm.y, lm.z) for lm in landmarks]

# Main processing loop
for subject in subjects:
    subject_folder = os.path.join(input_base_path, subject)
    image_files = sorted([f for f in os.listdir(subject_folder) if f.endswith('.jpg')], key=lambda x: int(x.split('_')[-1].split('.')[0]))

    data = []
    failed_images = []

    for image_file in image_files:
        image_path = os.path.join(subject_folder, image_file)
        landmarks = process_image(image_path)
        if landmarks is None:
            failed_images.append(image_file)
        else:
            row = [image_file] + [coord for landmark in landmarks for coord in landmark]
            data.append(row)

    # Save landmarks to CSV
    df = pd.DataFrame(data)
    csv_file_path = os.path.join(subject_folder, f'{subject}_landmarks.csv')
    df.to_csv(csv_file_path, index=False, header=False)

    # Save failed images to TXT
    failed_images_path = os.path.join(subject_folder, f'{subject}_failed_images.txt')
    with open(failed_images_path, 'w') as file:
        for fail in failed_images:
            file.write(f"{fail}\n")

print("Landmark extraction complete.")


# Define paths
base_path = "D:/MPIIFaceGaze/MPIIFaceGaze_normalized/MPIIFaceGaze_Landmarks"
subjects = [f'p{str(i).zfill(2)}' for i in range(15)]


# Function to load and combine data
def combine_landmarks_and_labels(subject_folder, subject):
    # Load landmarks CSV
    landmarks_csv_path = os.path.join(subject_folder, f'{subject}_landmarks.csv')
    landmarks_df = pd.read_csv(landmarks_csv_path, header=None)

    # Load labels CSV
    labels_csv_path = os.path.join(subject_folder, f'{subject}_labels.csv')
    labels_df = pd.read_csv(labels_csv_path, header=None)

    # Ensure the first column is treated as strings
    labels_df[0] = labels_df[0].astype(str)

    # Extract the first two columns (2D labels) from labels CSV
    labels_2d = labels_df.iloc[:, :2]

    # Get the valid image names from the landmarks CSV
    valid_image_names = landmarks_df[0].str.replace('.jpg', '', regex=False).values

    # Initialize a list to store the combined rows
    combined_data = []

    for index, row in landmarks_df.iterrows():
        image_name = row[0].replace('.jpg', '')
        # Find the index of the corresponding row in the labels CSV
        label_index = labels_df.index[labels_df[0].str.contains(image_name)]
        if not label_index.empty:
            label_index = label_index[0]
            # Get the 2D labels
            corresponding_label = labels_df.iloc[label_index, :2].values
            # Append the 2D labels to the landmarks row
            combined_row = list(row) + list(corresponding_label)
            combined_data.append(combined_row)

    # Create a DataFrame for the combined data
    combined_df = pd.DataFrame(combined_data)

    # Save the combined data to a new CSV file
    combined_csv_path = os.path.join(subject_folder, f'{subject}_combined_landmarks_labels.csv')
    combined_df.to_csv(combined_csv_path, index=False, header=False)


# Main processing loop
for subject in subjects:
    subject_folder = os.path.join(base_path, subject)
    combine_landmarks_and_labels(subject_folder, subject)

print("Combined CSV files created successfully.")
