# https://www.youtube.com/watch?v=vwveMKyzmMM

import numpy

from qiskit import QuantumCircuit
from qiskit import Aer, transpile
import math
import qiskit
from qiskit.tools.visualization import plot_histogram


circ = QuantumCircuit(1)
#initial_state = [0,1]
#initial_state = [math.sqrt(1/2),math.sqrt(1/2)]
initial_state = [math.sqrt(3/4),math.sqrt(1/4)]
circ.initialize(initial_state,0)
circ.measure_all()

# circ.draw('mpl')

simulator = Aer.get_backend("aer_simulator")
circ = transpile(circ,simulator)

result = simulator.run(circ).result()
count = result.get_counts(circ)

print(count)

#plot_histogram(count)



