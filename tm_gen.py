import numpy as np
import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN


print("Loading data...")
df = pd.read_csv('data/apr24_rr.csv')
#df = df[0:500]
docs = df.full_text.to_list()


print("Setting up model...")
umap_model = UMAP(n_neighbors=15, n_components=10,
                  min_dist=0.0, metric='cosine', random_state=42)
hdbscan_model = HDBSCAN(min_cluster_size=15, metric='euclidean', prediction_data=True, min_samples=5)
sentence_model = SentenceTransformer("all-mpnet-base-v2", device="cuda") #"all-MiniLM-L12-v2"

model = BERTopic(embedding_model=sentence_model, 
                 verbose=True, nr_topics='auto', 
                 calculate_probabilities=True,	
                 hdbscan_model=hdbscan_model,
                 umap_model=umap_model
                 )	

print("Training model...")
topic_model = model.fit(docs)

print("Saving model...")
topic_model.save("models/apr24_rr")


print("Making predictions...")
topics, probabilities = topic_model.transform(docs)

print("Building adjacency matrix...")
n = len(probabilities) #df.shape[0]
m = len(probabilities[0])

A = np.zeros((n,m))

for i in range(n):
    A[i][np.argsort(probabilities[i])[10:]] = 1

print("Saving adjacency matrix...")
with open('adjacency/apr24_adj_rr.npy', 'wb') as f:
    np.save(f, A)

print("Done.")
