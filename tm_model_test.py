import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

model = BERTopic.load("test_model_5")

# model.visualize_topics()
print(model.get_topic_info())