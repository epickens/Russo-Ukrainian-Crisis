import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, SpectralClustering
from tm_gen import load_data

def load_graph_and_signals(path1, path2):
    return np.load(path1), np.load(path2)


A, signals = load_graph_and_signals('adjacency/current/A_apr24_new_20.npy', 'signals/probabilities_1.npy')


count = 0
for i in range(500):
    temp = (sorted(signals[i], reverse=True))
    if temp[0] < 0.025:
        count += 1

print(count)

I = np.identity(500)

d_seq = [np.sum(row) for row in A]

D = I*d_seq

L = D - A

eigs, evecs = np.linalg.eigh(L)

# plt.plot(eigs)
# plt.show()
# print(eigs)

H_l = np.matmul(np.linalg.inv(I+L), signals[1:501,:])

# print(H_l)
kmeans = KMeans(n_clusters=2, random_state=0).fit(H_l)

labs = kmeans.predict(H_l)
print(labs)

docs = load_data('data/apr24_rr_20.csv')

for i in range(len(labs)):
    if labs[i] > 0:
        print('\n')
        print(docs[i+1])
