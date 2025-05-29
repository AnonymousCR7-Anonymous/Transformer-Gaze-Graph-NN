import numpy as np
import pandas as pd
import networkx as nx
import torch
from torch_geometric.nn import GATConv, GlobalAttention, TransformerConv
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
import torch.nn as nn
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import time
import os
from thop import profile

start1 = time.time()

def load_and_preprocess_data(file_name):
    df = pd.read_csv(file_name)

    landmarks_indices = list(range(478))
    indices = [idx for sublist in [[idx * 3 + 1, idx * 3 + 2, idx * 3 + 3] for idx in landmarks_indices] for idx in sublist]
    df_landmarks = df.iloc[:, indices]
    gaze_vectors = df.iloc[:, -3:]

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
            node_pos = row[i * 3: i * 3 + 3]
            G.add_node(i, pos=node_pos)

        new_edges = [(landmarks_to_new_indices[i], landmarks_to_new_indices[j]) for i, j in edges]
        G.add_edges_from(new_edges)

        G.graph['target'] = gaze_vectors.iloc[idx]

        x = torch.tensor([data['pos'] for _, data in G.nodes(data=True)], dtype=torch.float)
        y = torch.tensor(G.graph['target'].values, dtype=torch.float).unsqueeze(0)
        edge_index = torch.tensor(list(G.edges)).t().contiguous()

        data = Data(x=x, edge_index=edge_index, y=y)
        data_list.append(data)

    return data_list


class TransformerNet(torch.nn.Module):
    def __init__(self, num_node_features):
        super(TransformerNet, self).__init__()
        # Define the hidden dimension and heads for consistency
        head_dim1 = 64
        head_dim2 = 32
        head_dim3 = 16
        head_dim4 = 8
        # TransformerConv layers
        self.conv1 = TransformerConv(num_node_features, head_dim1 * 8)
        self.conv2 = TransformerConv(head_dim1 * 8, head_dim2 * 8)
        self.conv3 = TransformerConv(head_dim2 * 8, head_dim3 * 4)
        self.conv4 = TransformerConv(head_dim3 * 4, head_dim4 * 4)
        # Global attention pooling
        self.att_pool = GlobalAttention(gate_nn=torch.nn.Linear(head_dim4 * 4, 1))
        # Fully connected layer
        self.fc = torch.nn.Linear(head_dim4 * 4, 3)  # Adjusted to output 3 values

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = F.elu(self.conv1(x, edge_index))
        x = F.elu(self.conv2(x, edge_index))
        x = F.elu(self.conv3(x, edge_index))
        x = F.elu(self.conv4(x, edge_index))

        x = self.att_pool(x, data.batch)
        x = self.fc(x)

        return x


# Load the train, validation, and test datasets
start_graph_build = time.time()
train_data = load_and_preprocess_data('train_landmarks.csv')
val_data = load_and_preprocess_data('validation_landmarks.csv')
test_data = load_and_preprocess_data('test_landmarks.csv')
end_graph_build = time.time()

print("The time of graph building:", (end_graph_build - start_graph_build), "s")

batch_size = 64

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size)
test_loader = DataLoader(test_data, batch_size=batch_size)

end1 = time.time()
print("The time of data loading:", (end1 - start1), "s")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_node_features = train_data[0].num_node_features
model = TransformerNet(num_node_features).to(device)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Calculate the number of parameters
num_params = count_parameters(model)
print(f"Total number of trainable parameters: {num_params}")

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

num_epochs = 50
criterion = nn.MSELoss()

train_losses = []
val_losses = []

# Training loop
start_train = time.time()

for epoch in range(1, num_epochs + 1):
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

    # Evaluate the model
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for data in val_loader:
            data = data.to(device)
            output = model(data)
            val_loss += criterion(output, data.y).item()
        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        print(f"Epoch {epoch}, Train Loss: {train_loss}, Validation Loss: {val_loss}")

end_train = time.time()
print("The Training time is:", (end_train - start_train), "s")

# Save the trained model
model_save_path = "trained_model.pth"
torch.save(model.state_dict(), model_save_path)
print(f"Trained model saved to {model_save_path}")

# Test phase
start_test = time.time()

model.eval()
test_loss = 0
total_angle_error = 0
counter = 0

with torch.no_grad():
    for data in test_loader:
        data = data.to(device)
        output = model(data)
        test_loss += criterion(output, data.y).item()

        # Compute the cosine of the angle between the predicted gaze and the true gaze
        magnitude_labels = torch.sqrt((data.y ** 2).sum(dim=1))
        magnitude_outputs = torch.sqrt((output ** 2).sum(dim=1))
        dot_product = (data.y * output).sum(dim=1)

        cos_theta = dot_product / (magnitude_labels * magnitude_outputs)

        # Clamp values to avoid acos domain errors
        cos_theta = torch.clamp(cos_theta, min=-1.0, max=1.0)

        error_rad = torch.acos(cos_theta)
        error_deg = torch.rad2deg(error_rad)  # Convert to degrees
        total_angle_error += torch.mean(error_deg)

        counter += 1

    test_loss /= len(test_loader)
    mean_angle_error = total_angle_error / counter  # mean angle error on test set

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Mean Angle Error: {mean_angle_error:.4f} degrees")

end_test = time.time()
print("The Testing time is:", (end_test - start_test), "s")

# Calculate FLOPs
dummy_input = torch.randn(1, 478, 3).to(device)
flops, params = profile(model, inputs=(dummy_input,))
print(f"Total FLOPs: {flops}")

# Plotting
plt.plot(train_losses, label='Training loss')
plt.plot(val_losses, label='Validation loss')
plt.legend()
plt.show()
