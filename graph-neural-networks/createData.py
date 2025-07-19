import os
import torch
import torch_geometric
from torch_geometric.data import Data, DataLoader
import pandas as pd
import itertools

# create syfeatures_nrows dataframe with columns ALERT_SCORE, AML_ATM#SUM, AML_CLF#SUM, ENTITY_ID, FINDING
syfeatures_nrows = pd.DataFrame({
    'ALERT_SCORE': [0.5, 0.6, 0.7, 0.8, 0.9],
    'AML_ATM#SUM': [1, 2, 3, 4, 5],
    'AML_CLF#SUM': [5, 4, 3, 2, 1],
    'ENTITY_ID': ['PTY0', 'PTY1', 'PTY2', 'PTY3', 'PTY0'],
    'FINDING': [1, 1, 1, 0, 0]
})


# create party_account_relation_nrows dataframe with columns account_key, party_key
party_account_relation_nrows = pd.DataFrame({
    'account_key': ['ACCT0', 'ACCT1', 'ACCT2', 'ACCT1', 'ACCT1','ACCT2'],
    'party_key': ['PTY0', 'PTY1', 'PTY2', 'PTY3', 'PTY0','PTY1']
})


# visualize party_account_relation_nrows as a graph
import networkx as nx
import matplotlib.pyplot as plt
G = nx.from_pandas_edgelist(party_account_relation_nrows, 'account_key', 'party_key')

plt.figure(figsize=(10, 6))
pos = nx.spring_layout(G)

# Draw source nodes (account_key) as circles
source_nodes = party_account_relation_nrows['account_key'].unique()
nx.draw_networkx_nodes(G, pos, nodelist=source_nodes, node_size=700, node_color='lightblue', node_shape='o', label='Account Key')

# Draw target nodes (party_key) as squares
target_nodes = party_account_relation_nrows['party_key'].unique()
nx.draw_networkx_nodes(G, pos, nodelist=target_nodes, node_size=700, node_color='lightgreen', node_shape='s', label='Party Key')

# Draw edges
nx.draw_networkx_edges(G, pos)

# Draw labels
nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')

plt.title('Party Account Relation Graph')
#plt.legend()
plt.show(block=False)

syfinal_nrows = pd.merge(syfeatures_nrows, party_account_relation_nrows,left_on='ENTITY_ID',right_on='party_key', how='inner')

#print(syfinal_nrows.columns)

# in syfinal_nrows, coun how many related parties have the same finding
syfinal_nrows['related_parties'] = syfinal_nrows.groupby('party_key')['FINDING'].transform(lambda x: x.sum())
syfinal_nrows['related_parties'] = syfinal_nrows['related_parties'].astype(int)

print(syfinal_nrows)

# Generate dataset
def create_dataset(syfinal_nrows):
    # Group parties by account_key
    grouped = syfinal_nrows.groupby('account_key')['party_key'].apply(list)

    print(grouped)

    # Create edges by generating all possible pairs of parties within each group
    edges = []
    for parties in grouped:
        edges.extend(itertools.combinations(parties, 2))

    # Remove duplicate edges (since the graph is undirected)
    edges = list(set(tuple(sorted(edge)) for edge in edges))

    # Convert party keys to unique numeric indices
    party_to_index = {party: idx for idx, party in enumerate(syfinal_nrows['party_key'].unique())}

    # Map party keys to indices and create edge_index
    edge_index = torch.tensor([[party_to_index[src], party_to_index[dst]] for src, dst in edges], dtype=torch.long).T

    print("Edge index:\n", edge_index)

    # Reorder syfinal_nrows and finding_nrows based on party_to_index
    syfinal_nrows = syfinal_nrows.set_index('party_key').loc[party_to_index.keys()].reset_index()
    finding_nrows = syfinal_nrows['FINDING']

    # Drop unnecessary columns for x
    syfinal_nrows = syfinal_nrows.drop(columns=['ENTITY_ID', 'account_key', 'party_key', 'FINDING'])

    # Create x and y tensors
    x = torch.tensor(syfinal_nrows.values, dtype=torch.float32)
    y = torch.tensor(finding_nrows.values, dtype=torch.long)

    # Create train, val, and test masks
    num_nodes = x.size(0)
    train_size = int(0.6 * num_nodes)
    val_size = int(0.2 * num_nodes)
    test_size = num_nodes - train_size - val_size

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    # Assign nodes to train, validation, and test sets
    train_mask[:train_size] = True
    val_mask[train_size:train_size + val_size] = True
    test_mask[train_size + val_size:] = True

    # Create a PyTorch Geometric Data object
    samples = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)

    return samples

dd = create_dataset(syfinal_nrows)

print(dd.x)
print(dd.y)
print('-----------------------------------')

# Convert to numpy and find unique rows
x_np = dd.x.numpy()
unique_indices = []
seen = set()

for i, row_tuple in enumerate(map(tuple, x_np)):
    if row_tuple not in seen:
        seen.add(row_tuple)
        unique_indices.append(i)

# Create new tensors with unique entries only
unique_indices = torch.tensor(unique_indices, dtype=torch.long)
x_unique = dd.x[unique_indices]
y_unique = dd.y[unique_indices]

# Update the edge_index to reflect new node indices
# Create a mapping from old indices to new indices
idx_map = {old_idx: new_idx for new_idx, old_idx in enumerate(unique_indices.tolist())}
edge_index_list = dd.edge_index.t().tolist()
new_edge_index = []

# Only keep edges where both nodes exist in our unique set
for u, v in edge_index_list:
    if u in idx_map and v in idx_map:
        new_edge_index.append([idx_map[u], idx_map[v]])

# Create new data object
if new_edge_index:
    new_edge_index = torch.tensor(new_edge_index).t()
    dd = Data(x=x_unique, edge_index=new_edge_index, y=y_unique)
else:
    dd = Data(x=x_unique, y=y_unique)

print(f"Original data had {x_np.shape[0]} nodes, after removing duplicates: {x_unique.shape[0]} nodes")
print(dd.x)
print(dd.y)

print(dd)

# split the dd into train, val, test
train_mask = torch.zeros(dd.x.size(0), dtype=torch.bool)
val_mask = torch.zeros(dd.x.size(0), dtype=torch.bool)
test_mask = torch.zeros(dd.x.size(0), dtype=torch.bool)
train_mask[:int(0.6 * dd.x.size(0))] = True
val_mask[int(0.6 * dd.x.size(0)):int(0.8 * dd.x.size(0))] = True
test_mask[int(0.8 * dd.x.size(0)):] = True
dd.train_mask = train_mask
dd.val_mask = val_mask
dd.test_mask = test_mask
print(dd.train_mask)
print(dd.val_mask)
print(dd.test_mask)

print('------------------------------------')
print(dd)

