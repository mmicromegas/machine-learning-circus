# https://towardsdatascience.com/an-introduction-to-quantum-computers-and-quantum-coding-e5954f5a0415
from qiskit import QuantumCircuit

qc = QuantumCircuit(2, 2)  # initialize a quantum circuit with 2 qubits and 2 classical bits

qc.h(0)  # apply Hadamrd gate to qubit 0, this is Bob's qubit
qc.h(1)  # apply Hadamard gate to qubit 1, this is Alice's qubit

qc.measure(0, 0) # measure Bob's qubit and map it to classical bit 0
qc.measure(1, 1) # measure Alice's qubit and map it to classical bit 1

# print results of the qc
print(qc.draw())  # prints the quantum circuit accociated with

#print(qc) # prints the quantum circuit accociated with this program


