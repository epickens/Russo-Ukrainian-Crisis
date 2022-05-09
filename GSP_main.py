import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_graph_and_signals(path1, path2):
    return np.load(path1), np.load(path2)


A, signals = load_graph_and_signals('adjacency/current/A_apr24_new_20.npy', 'signals/probabilities_1.npy')

I = np.identity(500)

d_seq = [np.sum(row) for row in A]

D = I*d_seq

L = D - A

eigs, evecs = np.linalg.eigh(L)

plt.plot(eigs)
plt.show()
print(eigs)
H_l = np.matmul(np.linalg.inv(I+0.5*L), signals[1:501,:])

# print(H_l)
