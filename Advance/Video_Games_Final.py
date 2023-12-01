import pandas as pd
import gensim
import numpy as np
import torch
from gensim.models import Word2Vec
import torch.nn as nn
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import spacy
import matplotlib.pyplot as plt

lr_rate = 0.01
epoch_Value = 100
context_size = 10
# ASK ABOUT THE MIN COUNT ON WEDNSDAY FOR NAMES

# def build_model(): # OLD MODEL
#     return torch.nn.Sequential(
#         torch.nn.Linear(2286, 1),
#         torch.nn.Sigmoid()
#
#     )

# def build_model(): #New model
#     return torch.nn.Sequential(
#         torch.nn.Linear(2286, 100),
#         torch.nn.ReLU(),
#         torch.nn.Linear(100, 1),
#         torch.nn.Sigmoid()
#     )

def build_model(): #New NEW model #smaller with same performance is better
    return torch.nn.Sequential(
        torch.nn.Linear(3286, 20),
        torch.nn.ReLU(),
        torch.nn.Linear(20, 15),
        torch.nn.ReLU(),
        torch.nn.Linear(15, 1)
        ,torch.nn.Sigmoid()
    )


# ---------------------------------------------------------------------------------------------------------------------------

df = pd.read_csv("video_games_sales.csv")
df = df[['Name','Year_of_Release', 'Critic_Score', 'Critic_Count', 'User_Score', 'User_Count', 'Genre', 'Rating', 'Publisher', 'Global_Sales']]

df = df.dropna()
# Drop rows where 'Column2' is not a float
df = df[~df['User_Score'].str.contains('tbd')]
df['User_Score'] = pd.to_numeric(df['User_Score'], errors='coerce')
df = df[df['User_Score'].notnull()]
print(df)

# ---------------------------------------------------------------------------------------------------------------------------

#tokenize text columns
df['Name'] = df['Name'].apply(lambda x: str(x))
df['Name'] = df['Name'].apply(lambda x: gensim.utils.simple_preprocess(x))
print(df['Name'])

# #keep max num of words, adding padding on the right
df['Name'] = df['Name'].apply(lambda x: x[:context_size] + [''] * (context_size - len(x)))
print(df['Name'])
#Get size of all Names df
name_size = int(len(df))
name_df = df[:name_size]

load_embeddings = True #true or false for different models
if load_embeddings:
    word_vector_size = 300
    word_vectors = gensim.models.KeyedVectors.load_word2vec_format(
        'GoogleNews-vectors-negative300.bin',
        binary=True)
else:
    word_vector_size = 200
    # trai word2vec model
    model = gensim.models.Word2Vec(
        name_df['Name'],
        vector_size = word_vector_size,
        max_vocab_size=2000, #2000 unqines wors that appear twice
        window=5, #2 words before and two words after and one in the middle
        min_count=1, #
        workers=10 # how many to do in parrelles
    )
    model.train(name_df['Name'], total_examples=len(name_df['Name']), epochs=10)
    word_vectors = model.wv
#
# #Tokenize Names with padding
name_df['Name'] = name_df['Name'].apply(lambda x: [word_vectors[word] if word in word_vectors else np.zeros(word_vector_size) for word in x])
print(name_df['Name'])

# ---------------------------------------------------------------------------------------------------------------------------

#Tokenize Tensor Names
t_name = torch.tensor(np.array(name_df['Name'].tolist()), dtype=torch.float32).reshape(-1, context_size * word_vector_size)
print(t_name.shape)

#normalize to 0-1 range
t_year = torch.tensor(df['Year_of_Release'].values, dtype=torch.float32).unsqueeze(1) / 10000
t_critScore = torch.tensor(df['Critic_Score'].values, dtype=torch.float32).unsqueeze(1) / 100
t_critCount = torch.tensor(df['Critic_Count'].values, dtype=torch.float32).unsqueeze(1) / 100
t_userScore = torch.tensor(df['User_Score'].values, dtype=torch.float32).unsqueeze(1) / 100
t_userCount = torch.tensor(df['User_Count'].values, dtype=torch.float32).unsqueeze(1) / 1000

# dummies for categorical columns
t_genre = torch.tensor(pd.get_dummies(df['Genre']).values, dtype=torch.float32)
t_rating = torch.tensor(pd.get_dummies(df['Rating']).values, dtype=torch.float32)
t_publisher = torch.tensor(pd.get_dummies(df['Publisher']).values, dtype=torch.float32)

# target is expenses & unsqeeeze to put them all in one little vector
t_globalSales = torch.tensor(df['Global_Sales'].values, dtype=torch.float32).unsqueeze(1) / 100

#concatenate all tensors into one
# print("Name :        " + str(t_name.shape))
print("Year Relase:  " + str(t_year.shape))
print("Crit Score:   " + str(t_critScore.shape))
print("Crit Count:   " + str(t_critCount.shape))
print("User Score:   " + str(t_userScore.shape))
print("User Count:   " + str(t_userCount.shape))
print("Genre:        " + str(t_genre.shape))
print("Rating:       " + str(t_rating.shape))
print("Publisher:    " + str(t_publisher.shape))
print("Global Sales: " + str(t_globalSales.shape))

#combined inputs into one testing data set
t_input = torch.cat([t_name, t_year, t_critScore, t_critCount, t_userScore, t_userCount, t_genre, t_rating, t_publisher], dim=1)
# t_input = torch.cat([t_year, t_critScore, t_critCount, t_userScore, t_userCount, t_genre, t_rating, t_publisher], dim=1)
print("T_input:      " + str(t_input.shape))

# ---------------------------------------------------------------------------------------------------------------------------

kf = KFold(n_splits=5, shuffle=True, random_state=42)

k = 1
for train_index, test_index in kf.split(t_input):
    print(f'FOLD {k}')
    # print(f'Train: {train_index}, Test: {test_index}')
    k = k + 1
    t_train_input = t_input[train_index] #train x
    t_train_target = t_globalSales[train_index] #train true Y
    t_test_input = t_input[test_index] #test X
    t_test_target = t_globalSales[test_index] #Test true Y

    model = build_model() #Building model size based on the input
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.001) #sgd = GRADIANT DESENT
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_rate) #sgd = GRADIANT DESENT
    loss_fn = torch.nn.MSELoss() #mean Square = loss

    for epoch in range(epoch_Value): #change with 100, 500, 1000  (MAKE THE SAME)
        optimizer.zero_grad() #Clear out the memory from previous epoch
        t_train_output = model(t_train_input) #get the input through the model
        loss = loss_fn(t_train_output, t_train_target) # get the loss between target and output which is x and true y
        loss.backward() #adjust weights and send the loss backwards | tell the weights how they contributed to the loss
        optimizer.step() #makes adjustments to the weights based on the loss

    #Validation prints
    t_test_output = model(t_test_input)
    test_loss = loss_fn(t_test_output, t_test_target)
    print("FOLD FINAL LOSS")
    print(f'Test loss: {test_loss}')

# ---------------------------------------------------------------------------------------------------------------------------

# separeate 80/20 train and test
t_train_input, t_test_input, t_train_target, t_test_target = train_test_split(t_input, t_globalSales, test_size=0.2, random_state=42)

model = build_model()
# optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
optimizer = torch.optim.Adam(model.parameters(), lr=lr_rate)
loss_fn = torch.nn.MSELoss()

losses_Graph = []
test_losses_Graph = []

for epoch in range(epoch_Value): #Change with 100, 500, 1000
    optimizer.zero_grad()  # Clear out the memory from previous epoch
    t_train_output = model(t_train_input)  # get the input through the model
    loss = loss_fn(t_train_output, t_train_target)  # get the loss between target and output which is x and true y
    loss.backward()  # adjust weights and send the loss backwards | tell the weights how they contributed to the loss
    optimizer.step()  # makes adjustments to the weights based on the loss

    losses_Graph.append(loss.item())
    t_test_output = model(t_test_input)

    test_loss = loss_fn(t_test_output, t_test_target)
    test_losses_Graph.append(test_loss.item())

#Validation prints
print("FOLD 80/20 LOSS")
t_test_output = model(t_test_input)
test_loss = loss_fn(t_test_output, t_test_target)
print(f'Test loss: {test_loss}')

# ---------------------------------------------------------------------------------------------------------------------------

#Graphing while thinking
plt.plot(losses_Graph, label='TRAIN')
plt.plot(test_losses_Graph, label='TEST')
plt.legend()
plt.show()

# ---------------------------------------------------------------------------------------------------------------------------

#Make Predictions
t_pred = model(t_test_input).detach() * 100 # since the true y is being / by 50000
t_test_target = t_test_target * 100
print(t_pred.shape)

# show predictions cs actual as dataframe
df_pred = pd.DataFrame(torch.cat([t_test_target, t_pred], dim=1).numpy(), columns=['actual', 'predicted'])
print(df_pred)
#compute the differences
df_pred['diff'] = df_pred['predicted'] - df_pred['actual']
print(df_pred)
#compute mean squared error
print(df_pred['diff'].pow(2).mean())
#root mean squared error
print(df_pred['diff'].pow(2).mean() ** 0.5)

# ---------------------------------------------------------------------------------------------------------------------------

# plot predictions vs actual
plt.scatter(t_test_target, t_pred)
plt.xlabel("ACTUAL")
plt.ylabel("PREDICTION")
plt.show()

#plot predictions vs actual as line
plt.plot(t_test_target, label="ACTUAL")
plt.plot(t_pred, label="PREDICTION")
plt.legend()
plt.show()








