import os
from PIL import Image
import mediapipe as mp
import numpy as np
import pandas as pd


def read_txt_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    data = {}
    for i in range(0, len(lines), 1):
        parts = lines[i].strip().split()
        image_path = parts[0]
        gaze_vector = [float(val) for val in parts[1:]]
        data[image_path] = gaze_vector

    return data


def process_data(file_path, source_path):
    data = read_txt_file(file_path)
    mp_face_mesh = mp.solutions.face_mesh

    landmarks_data = {}
    failed_images = []

    with mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
    ) as face_mesh:

        for image_path, gaze_vector in data.items():

                img_pil = Image.open(os.path.join(source_path, image_path))
                image = np.array(img_pil)
                results = face_mesh.process(image)

                if results.multi_face_landmarks:
                    landmarks = results.multi_face_landmarks[0].landmark
                    flattened_landmarks = [coord for landmark in landmarks for coord in
                                           (landmark.x, landmark.y, landmark.z)]
                    landmarks_data[image_path] = gaze_vector + flattened_landmarks
                else:
                    failed_images.append(image_path)

    df = pd.DataFrame.from_dict(landmarks_data, orient='index')
    df.to_csv(file_path.replace('.txt', '_PIL_landmarks.csv'), index_label='image_path')
    with open(file_path.replace('.txt', '_PIL_failed_images.txt'), 'w') as f:
        f.write("\n".join(failed_images))


# Paths
source_path = ".../Gaze360/imgs/"
val_file = "validation.txt"
train_file = "train.txt"
test_file = "test.txt"

## Processing
 process_data(train_file, source_path)
# process_data(val_file, source_path)
# process_data(test_file, source_path)
