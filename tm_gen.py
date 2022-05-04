import numpy as np
import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
import torch
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
import gc


def load_data(path, limit=False, stop=20000):
    print("Loading data...")
    df = pd.read_csv(path)
    if limit:
        df = df[0:stop]
    docs = df.full_text.to_list()
    return docs


def get_pooler_output(input_ids, model):
    with torch.no_grad():
        features = model(input_ids)
    return features.pooler_output[0]


def get_model(docs, betweet=False):
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print(device)
    gc.collect()
    torch.cuda.empty_cache()
    print("Setting up model...")

    umap_model = UMAP(n_neighbors=15, n_components=10,
                      min_dist=0.0, metric='cosine', random_state=42)
    hdbscan_model = HDBSCAN(min_cluster_size=15, metric='euclidean', prediction_data=True, min_samples=5)

    if betweet:
        sentence_model = SentenceTransformer("vinai/bertweet-large", device="cuda")
    else:
        sentence_model = SentenceTransformer("all-mpnet-base-v2", device="cuda") #  "all-MiniLM-L12-v2"

    print("Generating embeddings...")
    embeddings = sentence_model.encode(docs, show_progress_bar=True)
    print("Saving embeddings...")
    with open('embeddings/tw_emb_mpnet.npy', 'wb') as fm:
        np.save(fm, np.asarray(embeddings))
    fm.close()

    model = BERTopic(embedding_model=sentence_model,
                     verbose=True, nr_topics='auto',
                     calculate_probabilities=True,
                     hdbscan_model=hdbscan_model,
                     umap_model=umap_model)
    return model, embeddings


def load_model(path, emb_path, embeddings=True):
    print("Loading model from saved state...")
    if embeddings:
        return BERTopic.load(path), np.load(emb_path)
    else:
        return BERTopic.load(path)


docs = load_data('data/apr24_rr_20.csv')
model, embeddings = get_model(docs, False)
# print(embeddings)
print("Training model...")
topic_model = model.fit(docs, embeddings)

print("Saving model...")
topic_model.save("models/apr24_new_20")
num_topics = len(topic_model.get_topics())

print("Making predictions...")
topics, probabilities = topic_model.transform(docs)
with open('signals/probabilities_1.npy', 'wb') as f:
    np.save(f, probabilities)
f.close()
with open('topics/topics_1.npy', 'wb') as f:
    np.save(f, topics)
f.close()

print("Building adjacency matrix...")
n = len(probabilities) #df.shape[0]
m = len(probabilities[0])
cut = num_topics - 3

A = np.zeros((n,m))

for i in range(n):
    A[i][np.argsort(probabilities[i])[cut:]] = 1

print("Saving adjacency matrix...")
with open('adjacency/B_apr24_new_20.npy', 'wb') as f:
    np.save(f, A)
f.close()

print("Done.")
