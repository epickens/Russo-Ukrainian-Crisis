import pandas as pd
import numpy as np
#from bertopic import BERTopic
from sentence_transformers import SentenceTransformer, CrossEncoder
from tqdm import tqdm
import torch
from transformers import AutoModel, AutoTokenizer


print("Loading adjacency matrix...")
topic_A = np.load("adjacency/A_apr24_rr_20.npy")

print("Loading text data...")
df = pd.read_csv('data/apr24_rr_20.csv')
df = df[0:500]
docs = df.full_text.to_list()


def get_pooler_output(input_ids, model):
	with torch.no_grad():
		features = model(input_ids)
	return features.pooler_output[0]


def bertweet_based_adjacency():
	print("Loading models...")
	bertweet = AutoModel.from_pretrained("vinai/bertweet-large")
	tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-large")

	print("Generating embeddings...")
	emb = []
	for i in tqdm(range(500)):
		input_ids = torch.tensor([tokenizer.encode(docs[i])])
		emb.append(get_pooler_output(input_ids, bertweet))

	print("Building adjacency matrix...")
	A = np.zeros((500, 500))

	for i in tqdm(range(500)):
		for j in range(i + 1, 500):
			if A[i, j] == 1:
				continue

			sim = np.dot(emb[i], emb[j])/(np.linalg.norm(emb[i]) * np.linalg.norm(emb[j]))
			A[i, j] = 1/sim
			A[j, i] = 1/sim
			# if sim >= 0.85:
			# 	A[i, j] = 1
			# 	A[j, i] = 1

	return A


def entailment_based_adjacency():
	print("Loading models...")
	entail_model = CrossEncoder('cross-encoder/nli-distilroberta-base')
	para_model = SentenceTransformer('sentence-transformers/paraphrase-mpnet-base-v2')

	print("Generating embeddings...")
	emb = []
	for i in range(500):
		emb.append(para_model.encode(docs[i]))

	print("Building adjacency matrix...")
	A = np.zeros((500, 500))

	label_mapping = [-1, 1, 0]

	for i in tqdm(range(500)):
		for j in range(i+1, 500):
			if A[i,j] == 1:
				continue

			sim = np.dot(emb[i], emb[j])/(np.linalg.norm(emb[i]) * np.linalg.norm(emb[j]))
			if sim >= 0.5:
				A[i,j] = 1
				A[j,i] = 1
			"""
			scores = entail_model.predict([docs[i], docs[j]])

			label = label_mapping[scores.argmax()]
			
			sim = np.dot(emb[i], emb[j])/(np.linalg.norm(emb[i]) * np.linalg.norm(emb[j]))
			
			if label == 1:
				A[i,j] = 1
				A[j,i] = 1
			elif sim >= 0.75:
				A[i,j] = 1
				A[j,i] = 1
			"""

	return A


A = bertweet_based_adjacency()

adj = A * topic_A#np.where(A == topic_A, 1, 0)
print("Saving adjacency matrix...")
with open('adjacency/A_test1.npy', 'wb') as f:
	np.save(f, adj)

print("Done.")
