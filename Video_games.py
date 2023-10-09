import pandas as pd
import gensim
import numpy as np
import torch

df = pd.read_csv("video_games_sales.csv")
print(df.head())

df2 = df[['Name', 'Platform', 'Genre', 'Publisher', 'Critic_Score', 'Global_Sales']]
print(df2)

df2 = df2.dropna()
print(df2)

df2 = pd.get_dummies(df2, columns=['Platform', 'Genre', 'Publisher'])
print(df2)
# https://www.kaggle.com/datasets/leadbest/googlenewsvectorsnegative300
emb_model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary = True)
print(emb_model['the'])
# print(emb_model['the'].shape)
# print(emb_model.vectors.shape)

# def euclid(a, b):
#     return  np.linalg.norm(a-b)
#
# def cos_sim(a,b):
#     return  np.dot(a,b) / (np.linalg.norm(a) * np.linalg.norm(b))
#
# def eucliud_dist_words(w1, w2):
#     return euclid(emb_model[w1], emb_model[w2])
#
# def cos_sin_words(w1, w2):
#     return cos_sim(emb_model[w1], emb_model[w2])

# print(eucliud_dist_words('the', 'the'))
# print(cos_sin_words('the', 'the'))
# # sin is closes when its cloers er to 1
# print(emb_model.most_similar('queen', topn=10))
#
# vec_queen = emb_model['queen']
# vec_king = emb_model['king']
# vec_woman = emb_model['woman']
#
# vec_resluts = vec_queen - vec_woman + vec_king
# print(emb_model.similar_by_vector(vec_resluts))

# tokenized name colunm
# df2['Name'] = df2['Name'].apply(lambda  x: x.lower().split())
# print(df2['Name'])
#
# df2['Name'] = df2['Name'].apply(lambda  x: [emb_model[i] for i in x if i in emb_model.vocab.keys()])
# print(df2['Name'])
# print(len(df2['Name'].iloc[0]))
# print(len(df2['Name'].iloc[1]))
# print(len(df2['Name'].iloc[2]))
# print(emb_model[' '])

# pad_vec = np.zeros(300)
# # pad each vector in name colum to length 3
# df2['Name'] = df2['Name'].apply(lambda  x: + [pad_vec] * (3 - len(x)))
#
# # trim each vector by 3
# df2['Name'] = df2['Name'].apply(lambda  x: x[:3])
# print(df2['Name'])
# # print len of each bname colume
# print(df2['Name'].apply(lambda  x: len(x)))
#
# t_name = torch.tensor(df2['Name'].values.tolist())
# print(t_name.shape)
# t_rest = torch.tensor(df2.drop(columns=['Name', 'Global_Sales']).values.tolist())

# t_input = torch.cat(t_name, t_rest, dim=1)
# print(t_input.shape)
# t_output = torch.tensor(df2['Global_Sales'].values.tolist())
