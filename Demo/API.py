
import cv2
import torch
import numpy as np
import networkx as nx
import mediapipe as mp
from torch_geometric.data import Data
from moviepy.editor import ImageSequenceClip
from torch_geometric.nn import GATConv, GlobalAttention, TransformerConv
import torch.nn.functional as F
import os
import time


# Define or import the TransformerNet class
class TransformerNet(torch.nn.Module):
    def __init__(self, num_node_features):
        super(TransformerNet, self).__init__()
        head_dim1 = 64
        head_dim2 = 32
        head_dim3 = 16
        head_dim4 = 8
        self.conv1 = TransformerConv(num_node_features, head_dim1 * 8)
        self.conv2 = TransformerConv(head_dim1 * 8, head_dim2 * 8)
        self.conv3 = TransformerConv(head_dim2 * 8, head_dim3 * 4)
        self.conv4 = TransformerConv(head_dim3 * 4, head_dim4 * 4)
        self.att_pool = GlobalAttention(gate_nn=torch.nn.Linear(head_dim4 * 4, 1))
        self.fc = torch.nn.Linear(head_dim4 * 4, 2)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.elu(self.conv1(x, edge_index))
        x = F.elu(self.conv2(x, edge_index))
        x = F.elu(self.conv3(x, edge_index))
        x = F.elu(self.conv4(x, edge_index))
        x = self.att_pool(x, data.batch)
        x = self.fc(x)
        return x


# Load the trained model
model_path = ".../trained_model_No_Or.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TransformerNet(num_node_features=3).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)


# Function to process images and extract landmarks using MediaPipe
def process_image(image):
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if not results.multi_face_landmarks:
        return None
    landmarks = results.multi_face_landmarks[0].landmark
    return [(lm.x, lm.y, lm.z) for lm in landmarks]


# Function to build graph from landmarks
def build_graph_from_landmarks(landmarks, edges):
    G = nx.Graph()
    for i, pos in enumerate(landmarks):
        G.add_node(i, pos=pos)
    G.add_edges_from(edges)
    x = torch.tensor([G.nodes[i]['pos'] for i in G.nodes()], dtype=torch.float)
    edge_index = torch.tensor(list(G.edges)).t().contiguous()
    data = Data(x=x, edge_index=edge_index)
    return data


# Define edges (from provided code)
edges = [(468, node) for node in range(478) if node != 468]
edges += [(473, node) for node in range(478) if node != 473]
edges.extend([(471, 159), (159, 469), (469, 145), (145, 471), (476, 475), (475, 474), (474, 477), (477, 476),
              (1, 33), (1, 173), (1, 162), (1, 263), (1, 398), (1, 368), (33, 246), (146, 161), (161, 160),
              (160, 150), (150, 158), (158, 157), (157, 173), (173, 155), (155, 154), (154, 153), (153, 145),
              (145, 144), (144, 163), (163, 7), (7, 33), (398, 384), (384, 385), (385, 386), (386, 387), (387, 388),
              (388, 263), (263, 249), (249, 390), (390, 373), (373, 374), (374, 380), (380, 381), (381, 382),
              (382, 398)])

# Path to input video
input_video_path = ".../input.mp4"

# Read the video
cap = cv2.VideoCapture(input_video_path)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Process each frame to calculate the gaze vector
gaze_vectors = []
frames = []
landmarks_list = []
frame_count = 0
execution_times = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(frame)

    start_time = time.time()

    landmarks = process_image(frame)
    if landmarks is None:
        continue
    landmarks_list.append(landmarks)
    data = build_graph_from_landmarks(landmarks, edges).to(device)
    with torch.no_grad():
        output = model(data)
        gaze_vectors.append(output.cpu().numpy())

    end_time = time.time()
    execution_times.append(end_time - start_time)

    frame_count += 1

cap.release()

# Calculate average execution time per frame
average_execution_time = sum(execution_times) / len(execution_times)

# Visualize the gaze vectors on the frames with POG display
output_frames = []
for i, frame in enumerate(frames):
    if i >= len(gaze_vectors):
        break
    landmarks = landmarks_list[i]
    gaze_vector = gaze_vectors[i]

    # Scale POG coordinates
    pog_x = int(gaze_vector[0][0] * frame_width)
    pog_y = int(gaze_vector[0][1] * frame_height)

    # Add a small black rectangle on top of the frame
    black_bg = (0, 0, 0)
    text_color = (0, 255, 255)  # Yellow color for the text
    overlay_height = 30
    cv2.rectangle(frame, (0, 0), (frame_width, overlay_height), black_bg, -1)

    # Display POG coordinates on the black rectangle
    text = f"POG: ({pog_x}, {pog_y})"
    cv2.putText(frame, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 3, cv2.LINE_AA)

    # Convert gaze vector to 3D vector with x = -x and y = y
    gaze_vector_3d = np.array([-gaze_vector[0][0], gaze_vector[0][1], 1.0])

    # Draw the gaze vector starting from landmarks 468 and 473
    start_point_468 = (int(landmarks[468][0] * frame_width), int(landmarks[468][1] * frame_height))
    end_point_468 = (
    start_point_468[0] + int(gaze_vector_3d[0] * 100), start_point_468[1] - int(gaze_vector_3d[1] * 100))
    start_point_473 = (int(landmarks[473][0] * frame_width), int(landmarks[473][1] * frame_height))
    end_point_473 = (
    start_point_473[0] + int(gaze_vector_3d[0] * 100), start_point_473[1] - int(gaze_vector_3d[1] * 100))

    cv2.arrowedLine(frame, start_point_468, end_point_468, (0, 0, 255), 4, tipLength=0.2)
    cv2.arrowedLine(frame, start_point_473, end_point_473, (0, 0, 255), 4, tipLength=0.2)

    output_frames.append(frame)

# Save the output video
output_video_path = ".../Output.mp4"
clip = ImageSequenceClip([cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in output_frames], fps=fps)
clip.write_videofile(output_video_path, codec='libx264')

fps = 1 / average_execution_time
print(f"FPS: {fps:.2f}")
print(f"Average execution time per frame: {average_execution_time * 1000:.4f} ms")
