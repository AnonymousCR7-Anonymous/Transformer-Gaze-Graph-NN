import os
import cv2
import json
import pandas as pd
import mediapipe as mp
import time


def save_data(landmarks_data, failed_images, batch_index):
    save_dir = "D:/GazeCapture_Results"
    os.makedirs(save_dir, exist_ok=True)

    for dataset_type in ['train', 'val', 'test_tablet', 'test_phone']:
        df = pd.DataFrame(landmarks_data[dataset_type])
        df.to_csv(f"{save_dir}/{dataset_type}_landmarks_all_batch_{batch_index}.csv", index=False)

        with open(f"{save_dir}/{dataset_type}_failed_images_all_batch_{batch_index}.txt", 'w') as f:
            f.write("\n".join(failed_images[dataset_type]))


def should_process_frame(face_data, left_eye_data, right_eye_data, index):
    return face_data['IsValid'][index] and left_eye_data['IsValid'][index] and right_eye_data['IsValid'][index]


base_dir = ".../GazeCapture"
folders = [os.path.join(base_dir, f) for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]

batch_size = 100
num_batches = len(folders) // batch_size + (1 if len(folders) % batch_size > 0 else 0)

mp_face_mesh = mp.solutions.face_mesh

for batch_count in range(num_batches):
    landmarks_data = {
        'train': [],
        'val': [],
        'test_tablet': [],
        'test_phone': []
    }

    failed_images = {
        'train': [],
        'val': [],
        'test_tablet': [],
        'test_phone': []
    }

    start_time = time.time()

    for folder in folders[batch_count * batch_size: (batch_count + 1) * batch_size]:
        subfolders = [os.path.join(folder, sf) for sf in os.listdir(folder) if os.path.isdir(os.path.join(folder, sf))]
        for subfolder in subfolders:
            folder_path = subfolder

            with open(os.path.join(folder_path, "info.json"), "r") as info_file:
                info = json.load(info_file)
                dataset_type = info['Dataset']

                device_type = 'tablet' if info['DeviceName'].startswith('iPad') else 'phone'
                specific_dataset_type = f'test_{device_type}' if dataset_type == 'test' else dataset_type

            # Load screen.json for orientation values
            with open(os.path.join(folder_path, "screen.json"), "r") as screen_file:
                screen_data = json.load(screen_file)
                orientation_values = screen_data['Orientation']

            with open(os.path.join(folder_path, "appleFace.json"), "r") as face_file, \
                    open(os.path.join(folder_path, "appleLeftEye.json"), "r") as left_eye_file, \
                    open(os.path.join(folder_path, "appleRightEye.json"), "r") as right_eye_file:
                face_data = json.load(face_file)
                left_eye_data = json.load(left_eye_file)
                right_eye_data = json.load(right_eye_file)

            with open(os.path.join(folder_path, "dotInfo.json"), "r") as dot_info_file:
                dot_info = json.load(dot_info_file)
                gaze_labels = list(zip(dot_info['XCam'], dot_info['YCam']))

            frames_path = os.path.join(folder_path, "frames")
            image_files = [os.path.join(frames_path, f) for f in os.listdir(frames_path) if
                           f.endswith('.jpg') or f.endswith('.png')]

            with mp_face_mesh.FaceMesh(
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
            ) as face_mesh:
                for index, image_file in enumerate(image_files):
                    if not should_process_frame(face_data, left_eye_data, right_eye_data, index):
                        continue

                    image = cv2.imread(image_file)
                    if image is None:
                        failed_images[specific_dataset_type].append(image_file)
                        continue

                    rgb_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    results = face_mesh.process(rgb_frame)

                    if results.multi_face_landmarks:
                        landmarks = results.multi_face_landmarks[0].landmark
                        flattened_landmarks = [coord for landmark in landmarks for coord in
                                               (landmark.x, landmark.y, landmark.z)]

                        # Append orientation value for the current index
                        flattened_landmarks.append(orientation_values[index])

                        landmarks_data[specific_dataset_type].append(
                            flattened_landmarks + list(gaze_labels[index]))
                    else:
                        failed_images[specific_dataset_type].append(image_file)

    end_time = time.time()
    save_data(landmarks_data, failed_images, batch_count)

    print(f"Processed batch {batch_count + 1} of {num_batches} in {end_time - start_time:.2f} seconds.")
