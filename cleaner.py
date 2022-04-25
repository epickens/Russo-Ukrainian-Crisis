import pandas as pd

print("Loading data...")
df = pd.read_csv('data/apr24.csv', engine='python')

df = df[['full_text', 'retweet_count', 'favorite_count']]

df_rr = df.sort_values(by=['favorite_count'], ascending=False)

df_rr = df_rr[0:10000]

path = 'data/apr24_red.csv'

print("Saving column reduced dataframe as: " + path)
df.to_csv(path)

path_rr = 'data/apr24_rr.csv'

print("Saving column and row reduced dataframe as: " + path_rr)
df_rr.to_csv(path_rr)
