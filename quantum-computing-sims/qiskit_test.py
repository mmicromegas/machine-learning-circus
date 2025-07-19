# https://www.youtube.com/watch?v=RrUTwq5jKM4

from qiskit import *
import matplotlib.pyplot as plt

qr = QuantumRegister(2)

cr = ClassicalRegister(2)

circuit = QuantumCircuit(qr, cr)

# draw the circuit
circuit.draw()


