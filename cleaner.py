import pandas as pd
import emoji
from tqdm import tqdm


def preprocess_bertweet(text):
    new_text = []
    for t in text.split(" "):
        t = '@USER' if t.startswith('@') and len(t) > 1 else t
        t = 'HTTPURL' if t.startswith('http') else t
        new_text.append(t)
    return emoji.demojize(" ".join(new_text))


print("Loading data...")
df = pd.read_csv('data/apr24.csv', engine='python')
df = df[['full_text', 'retweet_count', 'favorite_count']]
df = df[df['full_text'].notnull()]

print("Text processing...")
df['full_text'] = df['full_text'].apply(preprocess_bertweet)

df_rr = df.sort_values(by=['retweet_count'], ascending=False)
df_rr = df_rr[0:20000]

path = 'data/apr24_red_20.csv'

print("Saving column reduced dataframe as: " + path)
df.to_csv(path)

path_rr = 'data/apr24_rr_20.csv'

print("Saving column and row reduced dataframe as: " + path_rr)
df_rr.to_csv(path_rr)		
