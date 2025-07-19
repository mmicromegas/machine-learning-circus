import sys

# disable all warnings
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool

# Generate dataset
def create_dataset():
    # Fixed adjacency matrix (A, B → C → D)
    edge_index = torch.tensor([
        [0, 1, 2, 2],  # Source nodes
        [2, 2, 3, 3]   # Target nodes
    ], dtype=torch.long)

    # All possible input combinations (A, B) and outputs (D)
    samples = []
    for a in [0, 1]:
        for b in [0, 1]:
            for d in [0, 1]:
                # Node features: [is_input, is_gate, is_output, value]
                x = torch.tensor([
                    [1, 0, 0, a],  # A
                    [1, 0, 0, b],  # B
                    [0, 1, 0, 0],  # C (gate)
                    [0, 0, 1, d]   # D
                ], dtype=torch.float)

                # Label: 1 if valid XOR, 0 otherwise
                y = torch.tensor([1 if d == (a ^ b) else 0], dtype=torch.float)
                samples.append(Data(x=x, edge_index=edge_index, y=y))
    return samples



# Prepare data
data = create_dataset()
torch.manual_seed(42)

print('Input Node Features:')
print(data[1].x)
print('**************')
print('Edge Index:')
print(data[1].edge_index)
# from edge_index, calculate adjacency matrix
adjacency_matrix = torch.zeros((4, 4))
for i in range(data[1].edge_index.shape[1]):
    adjacency_matrix[data[1].edge_index[0][i]][data[1].edge_index[1][i]] = 1

print('Adjacency matrix:')
print(adjacency_matrix)

print('**************')
print('Output Label:')
print(data[1].y)
#print(test_data)

# https://rish-16.github.io/posts/gnn-math/

# initialize weights vector of size 4 times 4 with values 1.
# initialize weights w with 1
w_1 = torch.ones((4, 4), requires_grad=True)

# initialize bias b with 0
b = torch.tensor(0.5, requires_grad=True)
# get node features of first graph
x_j = data[0].x

node_A = x_j[0]
node_B = x_j[1]
node_C = x_j[2]
node_D = x_j[3]

# calculate affine transform F for node A
F_A = torch.matmul(w_1, node_A) + b
# calculate affine transform F for node B
F_B = torch.matmul(w_1, node_B) + b
# calculate affine transform F for node C
F_C = torch.matmul(w_1, node_C) + b
# calculate affine transform F for node D
F_D = torch.matmul(w_1, node_D) + b

# print node_A
print('Node_A:')
print(node_A)

# print w_1
print('Weights_1:')
print(w_1)

# print F_A
print('F_A:')
print(F_A)

# print F_C
print('F_C:')
print(F_C)

