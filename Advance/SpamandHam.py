import pandas as pd
import gensim
import numpy as np
import torch
import matplotlib.pyplot as plt


df = pd.read_csv('spam_ham_dataset.csv')
df = df[['text', 'label_num']]
print(df)

#tokenize text column
df['text'] = df['text'].apply(lambda x: gensim.utils.simple_preprocess(x))
print(df['text'])

context_size = 100
#keep max num of words, adding padding on the right
df['text'] = df['text'].apply(lambda x: x[:context_size] + [''] * (context_size - len(x)))
print(df['text'])

#split train and test in 80/20
train_size = int(len(df) * 0.8)
train_df = df[:train_size]
test_df = df[train_size:]

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
        train_df['text'],
        vector_size = word_vector_size,
        max_vocab_size=2000, #2000 unqines wors that appear twice
        window=5, #2 words before and two words after and one in the middle
        min_count=2, #
        workers=10 # how many to do in parrelles
    )
    model.train(train_df['text'], total_examples=len(train_df['text']), epochs=10)
    word_vectors = model.wv

#take each word and tokenize it and get each word from x and take it out of the word vectors in the 200 numerica
train_df['text'] = train_df['text'].apply(lambda x: [word_vectors[word] if word in word_vectors else np.zeros(word_vector_size) for word in x])
test_df['text'] = test_df['text'].apply(lambda x: [word_vectors[word] if word in word_vectors else np.zeros(word_vector_size) for word in x])
print(train_df['text'])

#input tensro
train_input = torch.tensor(np.array(train_df['text'].tolist()), dtype=torch.float32).reshape(-1, context_size * word_vector_size)
train_target = torch.tensor(np.array(train_df['label_num'].tolist()), dtype=torch.float32).unsqueeze(1)
test_input = torch.tensor(np.array(test_df['text'].tolist()), dtype=torch.float32).reshape(-1, context_size * word_vector_size)
test_target = torch.tensor(np.array(test_df['label_num'].tolist()), dtype=torch.float32).unsqueeze(1)

print(train_input.shape)
print(train_target.shape)
print(test_input.shape)
print(test_target.shape)

model = torch.nn.Sequential(
    torch.nn.Linear(context_size * word_vector_size, 100),
    torch.nn.ReLU(),
    torch.nn.Linear(100, 1),
    torch.nn.Sigmoid()
)

loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

losses = []
test_losses = []
for epoch in range(100):
    optimizer.zero_grad()
    train_output = model(train_input)
    train_loss = loss_fn(train_output, train_input)
    train_loss.backward()
    optimizer.step()
    losses.append(train_loss.item())
    test_output = model(test_input)
    test_loss = loss_fn(test_output, test_target)
    test_losses.append(test_loss.item())
    print(f'Epoch {epoch}: train loss {train_loss.item()}, test loss {test_loss.item()}')

#graph epoch
plt.plot(losses, label='train')
plt.plot(test_losses, label='test')
plt.legend()
plt.show()




