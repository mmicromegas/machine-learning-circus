import os
import torch

import torch_geometric

# Load the samples object from the file
nrows = 64721
#nrows = 1000
syf_dataset = torch.load('syf_graph_nrows_{}.pt'.format(nrows),weights_only=False)

syf_graph = syf_dataset

print("Training samples: ", syf_graph.train_mask.sum().item())
print("Validation samples: ", syf_graph.val_mask.sum().item())
print("Test samples: ", syf_graph.test_mask.sum().item())

print(f'Number of nodes: {syf_graph.num_nodes}')
print(f'Number of edges: {syf_graph.num_edges}')
print(f'Average node degree: {syf_graph.num_edges / syf_graph.num_nodes:.2f}')
print(f'Has isolated nodes: {syf_graph.has_isolated_nodes()}')
print(f'Has self-loops: {syf_graph.has_self_loops()}')
print(f'Is undirected: {syf_graph.is_undirected()}')

label_dict = {
    0: "Non-Issue",
    1: "Issue"
}

syf_graph.y[:10]

import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(in_features=input_channels, out_features=hidden_channels),
            nn.ReLU(),
            nn.Linear(in_features=hidden_channels, out_features=output_channels)
        )

    def forward(self, data):
        # only using node features (x)
        x = data.x

        output = self.layers(x)

        return output

    def predict_proba(self, data):
        # forward Method: Computes the raw logits (unscaled scores) for each class.
        # predict_proba Method: Applies the softmax function to the logits to convert them into probabilities.

        # Apply softmax to get probabilities
        logits = self.forward(data)
        probabilities = F.softmax(logits, dim=1)
        return probabilities


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

syf_graph = syf_dataset.to(device)

input_channels = syf_dataset.num_features

hidden_channels = 16

output_channels = 2

model = MLP(
    input_channels = input_channels,
    hidden_channels = hidden_channels,
    output_channels = output_channels).to(device)

print(model)

print("Number of parameters: ", sum(p.numel() for p in model.parameters()))

optimizer = torch.optim.Adam(model.parameters(), lr = 0.01, weight_decay = 5e-4)

criterion = nn.CrossEntropyLoss()

num_epochs = 7000

for epoch in range(num_epochs):

    model.train()
    optimizer.zero_grad()
    out = model(syf_graph)

    loss = criterion(out[syf_graph.train_mask], syf_graph.y[syf_graph.train_mask])
    loss.backward()

    optimizer.step()

    # Get predictions on the training data
    pred_train = out.argmax(dim=1)

    correct_train = (
            pred_train[syf_graph.train_mask] == syf_graph.y[syf_graph.train_mask]
    ).sum()

    acc_train = int(correct_train) / int(syf_graph.train_mask.sum())

    # Get predictions on validation data
    model.eval()

    pred_val = model(syf_graph).argmax(dim=1)

    correct_val = (
            pred_val[syf_graph.val_mask] == syf_graph.y[syf_graph.val_mask]
    ).sum()

    acc_val = int(correct_val) / int(syf_graph.val_mask.sum())

    if (epoch + 1) % 1000 == 0:
        print(f'Epoch: {epoch + 1:03d}, \
               Train Loss: {loss:.3f}, \
               Train Acc: {acc_train:.3f} Val Acc: {acc_val:.3f}')



model.eval()

pred = model(syf_graph).argmax(dim = 1)

correct = (pred[syf_graph.test_mask] == syf_graph.y[syf_graph.test_mask]).sum()

test_acc = int(correct) / int(syf_graph.test_mask.sum())



