import numpy as np
import pandas as pd
import networkx as nx
import torch
from torch_geometric.nn import GATConv, GlobalAttention, TransformerConv, GCNConv
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
import torch.nn as nn
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import time
import os
from fvcore.nn import FlopCountAnalysis, parameter_count_table

start1 = time.time()

def load_and_preprocess_data(file_name):
    df = pd.read_csv(file_name)

    landmarks_indices = list(range(478))
    indices = [idx for sublist in [[idx * 3 + 1, idx * 3 + 2, idx * 3 + 3] for idx in landmarks_indices] for idx in sublist]
    df_landmarks = df.iloc[:, indices]
    pog_vectors = df.iloc[:, -2:]  # Extract 2D POG labels

    edges = [(468, node) for node in range(478) if node != 468]
    edges += [(473, node) for node in range(478) if node != 473]
    edges.extend([(471, 159), (159, 469), (469, 145), (145, 471),
                  (476, 475), (475, 474), (474, 477), (477, 476),
                  (1, 33), (1, 173), (1, 162), (1, 263), (1, 398), (1, 368),
                  (33, 246), (146, 161), (161, 160), (160, 150), (150, 158), (158, 157),
                  (157, 173), (173, 155), (155, 154), (154, 153), (153, 145), (145, 144),
                  (144, 163), (163, 7), (7, 33),
                  (398, 384), (384, 385), (385, 386), (386, 387), (387, 388),
                  (388, 263), (263, 249), (249, 390), (390, 373), (373, 374),
                  (374, 380), (380, 381), (381, 382), (382, 398)])

    data_list = []
    for idx, row in df_landmarks.iterrows():
        G = nx.Graph()
        landmarks_to_new_indices = {landmark: i for i, landmark in enumerate(landmarks_indices)}
        for i in range(len(landmarks_indices)):
            node_pos = row.iloc[i * 3: i * 3 + 3]
            G.add_node(i, pos=node_pos)

        new_edges = [(landmarks_to_new_indices[i], landmarks_to_new_indices[j]) for i, j in edges]
        G.add_edges_from(new_edges)

        G.graph['target'] = pog_vectors.iloc[idx]

        x = torch.tensor([data['pos'] for _, data in G.nodes(data=True)], dtype=torch.float)
        y = torch.tensor(G.graph['target'].values, dtype=torch.float).unsqueeze(0)
        edge_index = torch.tensor(list(G.edges)).t().contiguous()

        data = Data(x=x, edge_index=edge_index, y=y)
        data_list.append(data)

    return data_list



# Cross-validation setup
base_path = "D:/MPIIFaceGaze/MPIIFaceGaze_normalized/MPIIFaceGaze_Landmarks"
subjects = [f"p{str(i).zfill(2)}" for i in range(15)]


all_data = {}
start_graph_build = time.time()
for subject in subjects:
    subject_path = os.path.join(base_path, subject, f"{subject}_landmarks.csv")
    all_data[subject] = load_and_preprocess_data(subject_path)
end_graph_build = time.time()
print("The time of graph building for all subjects:", (end_graph_build - start_graph_build), "s")

batch_size = 64
num_epochs = 50
criterion = nn.MSELoss()

class GCNNet(torch.nn.Module):
    def __init__(self, num_node_features):
        super(GCNNet, self).__init__()
        # Define the hidden dimension sizes
        head_dim1 = 64
        head_dim2 = 32
        head_dim3 = 16
        head_dim4 = 8

        # GCNConv layers instead of TransformerConv
        self.conv1 = GCNConv(num_node_features, head_dim1 * 8)
        self.conv2 = GCNConv(head_dim1 * 8, head_dim2 * 8)
        self.conv3 = GCNConv(head_dim2 * 8, head_dim3 * 4)
        self.conv4 = GCNConv(head_dim3 * 4, head_dim4 * 4)

        # Global attention pooling (same as original)
        self.att_pool = GlobalAttention(gate_nn=torch.nn.Linear(head_dim4 * 4, 1))

        # Fully connected layer
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

# ---- Dummy Input for FLOPs ----

dummy_data = Data(
    x=torch.randn(478, 3),
    edge_index=torch.randint(0, 478, (2, 996)),
    batch=torch.zeros(478, dtype=torch.long)
)

model = GCNNet(num_node_features=3)
model.eval()

# ---- Compute GFLOPs ----

flops = FlopCountAnalysis(model, dummy_data)
print(parameter_count_table(model))
print(f"Total GFLOPs: {flops.total() / 1e9:.6f} GFLOPs")

all_test_losses = []
all_test_times = []
all_mean_error = []

for leave_out_subject in subjects:
    print(f"Leaving out subject: {leave_out_subject}")

    # Prepare training data by excluding the leave-out subject
    train_data = []
    for subject in subjects:
        if subject != leave_out_subject:
            train_data += all_data[subject]

    # Prepare test data with the leave-out subject
    test_data = all_data[leave_out_subject]

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_node_features = train_data[0].num_node_features
    model = GCNNet(num_node_features).to(device)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Calculate the number of parameters
    num_params = count_parameters(model)
    print(f"Total number of trainable parameters: {num_params}")

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    train_losses = []

    # Training loop
    start_train = time.time()

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()
        model.train()
        train_loss = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, data.y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        epoch_end = time.time()
        print(f"Epoch {epoch}, Train Loss: {train_loss}, Epoch Time: {epoch_end - epoch_start}s")

    end_train = time.time()
    print("The Training time is:", (end_train - start_train), "s")

    # Test phase
    start_test = time.time()

    model.eval()
    test_loss = 0
    total_distance_error = 0
    counter = 0

    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            output = model(data)
            test_loss += criterion(output, data.y).item()

            # Compute the Euclidean distance between the predicted POG and the true POG
            distance_error = torch.sqrt(torch.sum((output - data.y) ** 2, dim=1))
            total_distance_error += torch.mean(distance_error)

            counter += 1

        test_loss /= len(test_loader)
        mean_distance_error = total_distance_error / counter  # mean distance error on test set

    end_test = time.time()
    test_time = end_test - start_test
    print(f"Test Loss for {leave_out_subject}: {test_loss:.4f}")
    print(f"Mean Distance Error for {leave_out_subject}: {mean_distance_error:.4f} ")
    print(f"The Testing time for {leave_out_subject} is:", test_time, "s")

    all_test_losses.append(test_loss)
    all_test_times.append(test_time)
    all_mean_error.append(mean_distance_error)

# Print overall results
print("Cross-validation results:")
for i, subject in enumerate(subjects):
    print(f"Subject {subject}: Test Loss = {all_test_losses[i]:.4f}, Test Time = {all_test_times[i]:.2f}s, Mean Distance Error = {all_mean_error[i]:.4f} ")

for i, subject in enumerate(subjects):
    print(f"Subject {subject}: Test Loss = {all_test_losses[i]:.4f}, Test Time = {all_test_times[i]:.2f}s, Mean Distance Error = {all_mean_error[i]:.4f} ")


#
# import numpy as np
#
# # Given data
# # normalized_POG_errors = [
# #     0.1012, 0.1366, 0.0970, 0.2183, 0.0954, 0.1093, 0.1048,
# #     0.1229, 0.1187, 0.1247, 0.0984, 0.1168, 0.1242, 0.1099, 0.1630
# # ]
#
# normalized_POG_errors = [
#     0.1, 0.17, 0.15, 0.15, 0.028, 0.14, 0.020,
#     0.19, 0.032, 0.19, 0.20, 0.15, 0.022, 0.15, 0.18
# ]
#
# screen_width = 1280
# par_a_w = 0.2238
# par_b_w = -3.2354
# par_a_h = 0.2238
# par_b_h = 0
#
# def calculate_real_world_error(normalized_POG_error, screen_width, par_a_w, par_b_w, par_a_h, par_b_h):
#     # Step 1: Convert Normalized Error to Pixel Error
#     pixel_error = normalized_POG_error * screen_width
#
#     # Step 2: Convert Pixel Error to Real-World Coordinates
#     real_x_error = (pixel_error * par_a_w) + par_b_w
#     real_y_error = (pixel_error * par_a_h) + par_b_h
#
#     # Step 3: Calculate Euclidean Distance
#     euclidean_distance = np.sqrt(real_x_error**2 + real_y_error**2)
#     return euclidean_distance
#
# # Calculate the real-world error for each subject
# real_world_errors = [
#     calculate_real_world_error(error, screen_width, par_a_w, par_b_w, par_a_h, par_b_h)
#     for error in normalized_POG_errors
# ]
#
# # Calculate the average error over all subjects
# average_error = np.mean(real_world_errors)
#
# # Print the results
# for i, (norm_error, real_error) in enumerate(zip(normalized_POG_errors, real_world_errors)):
#     print(f"Subject {i+1} - Normalized POG Error: {norm_error}, Real-World Error: {real_error:.2f} mm")
#
# print(f"\nAverage Real-World Error over all subjects: {average_error:.2f} mm")




