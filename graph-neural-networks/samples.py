import numpy as np

# Generate dataset
def create_dataset():
    # Fixed adjacency matrix (A, B → C → D)
    adj = np.array([
        [0, 0, 1, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 0]
    ], dtype=np.float32)

    # All possible input combinations (A,B) and outputs (D)
    samples = []
    for a in [0, 1]:
        for b in [0, 1]:
            for d in [0, 1]:
                # Node features: [is_input, is_gate, is_output, value]
                x = np.array([
                    [1, 0, 0, a],  # A
                    [1, 0, 0, b],  # B
                    [0, 1, 0, 0],  # C (gate)
                    [0, 0, 1, d]  # D
                ], dtype=np.float32)

                # Label: 1 if valid XOR, 0 otherwise
                y = np.array([1 if d == (a ^ b) else 0], dtype=np.float32)
                samples.append((x, adj, y))
    return samples


data = create_dataset()

print(data)