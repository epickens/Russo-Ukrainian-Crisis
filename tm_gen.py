import numpy as np
import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

df = pd.read_csv('data/test.csv')
docs = df.full_text.to_list()


sentence_model = SentenceTransformer("all-MiniLM-L12-v2", device="cpu")

model = BERTopic(embedding_model=sentence_model, verbose=True, nr_topics=10, calculate_probabilities=True)

topics, probabilities = model.fit_transform(docs)

model.save("test_model_10") 
