import networkx as nx 
import numpy as np


A = np.load("adjacency/A_test2.npy")
# print(A.shape)
G = nx.from_numpy_array(A, create_using=nx.DiGraph) #, create_using=nx.MultiGraph

print("Saving output as edgelist...")
with open("graphs/apr24_test2.edgelist.csv", "wb") as fh:
	#nx.write_edgelist(G, fh)
	nx.write_edgelist(G, delimiter=',', path=fh)
	
print("Done.")

