import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, TransformerConv, GlobalAttention
from torch_geometric.data import Data, DataLoader
import networkx as nx
import time
from torch_scatter import scatter_add
from torch_geometric.utils import softmax
from torch_geometric.nn.aggr.attention import AttentionalAggregation
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
import os

batch_size = 256

# Load and concatenate data
data_dir = "D:/..."    #The direction of landmarks' csv files
file_types = ["train_landmarks_all_batch_", "val_landmarks_all_batch_"]
dfs = {file_type: [] for file_type in file_types}

start_time = time.time()

for file_type in file_types:
    for i in range(15):
        filename = os.path.join(data_dir, file_type + str(i) + ".csv")

        # Check if the file is empty or does not exist
        try:
            df = pd.read_csv(filename)
            if df.empty:
                print(f"File {filename} is empty. Skipping...")
                continue
            dfs[file_type].append(df)
        except FileNotFoundError:
            print(f"File {filename} not found. Skipping...")
        except pd.errors.EmptyDataError:
            print(f"File {filename} is empty. Skipping...")

    dfs[file_type] = pd.concat(dfs[file_type], ignore_index=True)

# Define edges
edges = [(468, node) for node in range(478) if node != 468]  # 468 connects to all
edges += [(473, node) for node in range(478) if node != 473]  # 473 connects to all
edges.extend([(471, 159), (159, 469), (469, 145), (145, 471),
              (476, 475), (475, 474), (474, 477), (477, 476),
              (1, 33), (1, 173), (1, 162), (1, 263), (1, 398), (1, 368),
              (33, 246), (146, 161), (161, 160), (160, 150), (150, 158), (158, 157),
              (157, 173), (173, 155), (155, 154), (154, 153), (153, 145), (145, 144),
              (144, 163), (163, 7), (7, 33),
              (398, 384), (384, 385), (385, 386), (386, 387), (387, 388),
              (388, 263), (263, 249), (249, 390), (390, 373), (373, 374),
              (374, 380), (380, 381), (381, 382), (382, 398)])


def normalize_2d_vector(vec):    #length-based normalizaton
    """
    Function to normalize a 2D vector.
    """
    magnitude = torch.norm(vec, p=2, dim=-1, keepdim=True)
    return vec / magnitude.clamp(min=1e-8)


# Since there is no header in the CSV files, we'll use the last two columns for gaze data
def build_graphs(df_landmarks, edges):
    data_list = []
    for idx, row in df_landmarks.iterrows():
        G = nx.Graph()
        for i in range(478):
            node_pos = row[i * 3: i * 3 + 3]
            G.add_node(i, pos=node_pos)

        G.add_edges_from(edges)

        gaze_vector = row.iloc[-2:]  # Last two columns
        orientation = row.iloc[1434]  # 1435th column

        # Modify gaze_vector based on orientation
        if orientation == 2:
            gaze_vector[1] = -gaze_vector[1]
        elif orientation == 3:
            gaze_vector = [gaze_vector[1], gaze_vector[0]]
        elif orientation == 4:
            gaze_vector = [-gaze_vector[1], -gaze_vector[0]]

        # Normalize the gaze_vector
        gaze_tensor = torch.tensor(gaze_vector, dtype=torch.float)
        gaze_tensor_normalized = normalize_2d_vector(gaze_tensor)

        # Storing the normalized gaze vector in the graph data
        G.graph['target'] = gaze_tensor_normalized

        x = torch.tensor([data['pos'] for _, data in G.nodes(data=True)], dtype=torch.float)
        y = G.graph['target'].unsqueeze(0)
        edge_index = torch.tensor(list(G.edges)).t().contiguous()

        data = Data(x=x, edge_index=edge_index, y=y)
        data_list.append(data)

    return data_list


# Using the function for data list creation
train_data_list = build_graphs(dfs['train_landmarks_all_batch_'], edges)
val_data_list = build_graphs(dfs['val_landmarks_all_batch_'], edges)

end_time = time.time()
print(f"loading time is {end_time - start_time:.2f} seconds.")
print('Graph construction is done!')


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


# Create an instance of the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TransformerNet(num_node_features=3).to(device)

# Count the total number of parameters
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"The model has {total_params:,} trainable parameters.")

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
criterion = torch.nn.MSELoss()
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

train_loader = DataLoader(train_data_list, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data_list, batch_size=batch_size, shuffle=False)

# Training & Validation
start_train_time = time.time()
train_losses = []
val_losses = []

for epoch in range(50):  # adjust epochs if needed
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
    train_losses.append(train_loss / len(train_loader))

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for data in val_loader:
            data = data.to(device)
            output = model(data)
            val_loss += criterion(output, data.y).item()
    val_losses.append(val_loss / len(val_loader))

    print(f"Epoch {epoch + 1}, Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}")

    # Step the scheduler after each epoch
    scheduler.step()

end_train_time = time.time()
print(f"Training time for 50 epochs is {end_train_time - start_train_time:.2f} seconds.")

# Save the trained model
model_save_path = "D:/GazeCapture_Results/trained_model.pt"
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")

# Load the test datasets
file_types_test = ["test_phone_landmarks_all_batch_", "test_tablet_landmarks_all_batch_"]
dfs_test = {file_type: [] for file_type in file_types_test}

start_test_graph_time = time.time()

for file_type in file_types_test:
    for i in range(15):
        filename = os.path.join(data_dir, file_type + str(i) + ".csv")

        # Check if the file is empty or does not exist
        try:
            df = pd.read_csv(filename)
            if df.empty:
                print(f"File {filename} is empty. Skipping...")
                continue
            dfs_test[file_type].append(df)
        except FileNotFoundError:
            print(f"File {filename} not found. Skipping...")
        except pd.errors.EmptyDataError:
            print(f"File {filename} is empty. Skipping...")

    dfs_test[file_type] = pd.concat(dfs_test[file_type], ignore_index=True)

# Build graphs for the test datasets
test_phone_data_list = build_graphs(dfs_test['test_phone_landmarks_all_batch_'], edges)
test_tablet_data_list = build_graphs(dfs_test['test_tablet_landmarks_all_batch_'], edges)
combined_test_data_list = test_phone_data_list + test_tablet_data_list

end_test_graph_time = time.time()
print(f"Building test graphs time: {end_test_graph_time - start_test_graph_time:.2f} seconds.")



# Create DataLoader for the test datasets
test_phone_loader = DataLoader(test_phone_data_list, batch_size=batch_size, shuffle=False)
test_tablet_loader = DataLoader(test_tablet_data_list, batch_size=batch_size, shuffle=False)
combined_test_loader = DataLoader(combined_test_data_list, batch_size=batch_size, shuffle=False)


# Function to compute test metrics
def compute_test_metrics(loader, model):
    model.eval()
    test_loss = 0
    total_euclidean_error = 0
    counter = 0

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data)
            test_loss += criterion(output, data.y).item()

            # Compute the Euclidean distance
            euclidean_error = torch.sqrt(torch.sum((output - data.y) ** 2, dim=1))
            total_euclidean_error += torch.mean(euclidean_error).item()
            counter += 1

    return test_loss / len(loader), total_euclidean_error / counter

# Testing
start_test_time = time.time()

# Compute metrics for each set
phone_loss, phone_error = compute_test_metrics(test_phone_loader, model)
tablet_loss, tablet_error = compute_test_metrics(test_tablet_loader, model)
combined_loss, combined_error = compute_test_metrics(combined_test_loader, model)

print(f"Phone Test Loss: {phone_loss:.4f}, Euclidean Error: {phone_error:.4f}")
print(f"Tablet Test Loss: {tablet_loss:.4f}, Euclidean Error: {tablet_error:.4f}")
print(f"Combined Test Loss: {combined_loss:.4f}, Euclidean Error: {combined_error:.4f}")

end_test_time = time.time()
print(f"Testing time is {end_test_time - start_test_time:.2f} seconds.")

plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss', linewidth=3)  # Increase linewidth
plt.plot(val_losses, label='Validation Loss', linewidth=3)  # Increase linewidth
plt.xlabel('Epoch', fontsize=18)  # Increase font size
plt.ylabel('Loss', fontsize=18)  # Increase font size
plt.title('Training and Validation Loss Over Epochs', fontsize=16)  # Increase font size
plt.legend(fontsize=16)  # Increase font size
plt.grid(True)
plt.xticks(fontsize=16)  # Increase font size
plt.yticks(fontsize=16)  # Increase font size
plt.show()
