import networkx as nx 
import numpy as np


print("Loading bipartite adjacency matrix...")
B = np.load("adjacency/apr24_adj_rr.npy")
#reduce size if desired
B = B[0:500, :]
#B[:, 0] = 0
#B = B[950:, :]
#print(B[:, 0])

print("Projecting matrix...")
A = np.zeros((500,500))

#Project B -> A
for i in range(500):
	for j in range(500):
		#ccount = 0 
		for k in range(B.shape[1]):
			if B[i,k] == 1 and B[j,k] == 1:
				#ccount += 1
				A[i,j] = 1
				A[j,i] = 1
				break
			#else:
			#	A[i,j] = 0
		#if ccount >= 4:
		#	A[i,j] = 1
		#	A[j,i] = 1
			

#print(A[0])		
G = nx.from_numpy_array(A)

print("Saving output as edgelist...")
with open("graphs/apr24_rr_end.edgelist.csv", "wb") as fh:
	#nx.write_edgelist(G, fh)
	nx.write_edgelist(G, delimiter=',', path=fh)
	
print("Done.")

