import gensim
import numpy as np
import pandas as pd
import torch
from gensim.models import Word2Vec

# Load pre-trained word embeddings using Gensim
# Replace 'path_to_pretrained_embeddings' with the actual path to your embeddings file
embedding_model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary = True)

# Sample DataFrame with a 'name' column
# data = {'name': ['apple banana cherry', 'dog cat', 'elephant']}
# df = pd.DataFrame(data)
df_file = pd.read_csv("video_games_sales.csv")
print(df_file.head())

df = df_file[['Name']]
print(df)

df = df.dropna()
print(df)

# Tokenize the 'name' column
df['Name'] = df['Name'].apply(lambda x: x.split())

# Pad sequences on the left to ensure they have exactly 5 words
max_seq_length = 5
df['Name'] = df['Name'].apply(lambda x: ['<PAD>'] * (max_seq_length - len(x)) + x[:max_seq_length])

# Apply pre-trained embeddings to tokenized and padded sequences
def word_to_vec(word):
    try:
        return embedding_model[word]
    except KeyError:
        return np.zeros(embedding_model.vector_size)  # Replace with a vector of zeros for out-of-vocabulary words

df['Name'] = df['Name'].apply(lambda x: [word_to_vec(word) for word in x])

# Convert the resulting DataFrame to a torch.Tensor
name_embeddings = torch.tensor(df['Name'].tolist())

print(name_embeddings)
