import torch
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import KFold
import numpy
import torch.nn as nn
import spacy

# Make a github repo with cavas data
fold = KFold(n_splits=5)
X_Tests = []
Y_Tests = []
Epoch_loss = []

def line(x):
    return 0.5 * x + 1

def mae(true_y, y_pred):
    return numpy.mean((abs(true_y - y_pred)))


def average_All_Folds(list):
    total = 0
    for avge in list:
        total = total + avge
    return total / len(list)

def Folding_Loop(x, y):
    for train_index, test_index in fold.split(x):
        print(train_index, test_index)
        train_x = numpy.array(x)[train_index]
        train_y = numpy.array(y)[train_index]
        test_x = numpy.array(x)[test_index]
        test_y = numpy.array(y)[test_index]

        # neural net code here, get a loss value for this fold
        print(train_x, test_x)
        print(train_y, test_y)

        model = nn.Linear(1, 1)
        # print(model)
        # print(list(model.parameters()))
        loss = nn.MSELoss()
        # print(loss)
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=0.001)  # lr = lreaning weight or how big of a jump[ of data to do
        # print(optimizer)
        x_touch = torch.tensor(train_x, dtype=torch.float32).unsqueeze(1)
        # print(x_touch)# unsqeese seperates the df file
        y_true_touch = torch.tensor(train_y, dtype=torch.float32).unsqueeze(1)
        print_count = 0
        for epoch in range(10):
            y_pred = model(x_touch)
            # print(y_pred)
            epoch_loss = loss(y_pred, y_true_touch)
            # if(print_count == 0 or print_count % 10 == 0):
            print("epoch loss #" + str(print_count) + ": " + str(epoch_loss))
            optimizer.zero_grad()  # restart the starting point in data
            epoch_loss.backward()  # propagate the error backwars (for each weight)
            optimizer.step()  # looking both ways in the data and choose the directioni to move the weight in favor of loss
            print_count = print_count + 1


        # print(list(model.parameters()))  # print acutal line


# df = pd.read_csv("heights_M.csv")
# x1 = df["Age (in months)"].values.tolist()
# x2 = df["3rd Percentile Length (in centimeters)"].values.tolist()
# x3 = df["5th Percentile Length (in centimeters)"].values.tolist()
# x4 = df["10th Percentile Length (in centimeters)"].values.tolist()
# x5 = df["25th Percentile Length (in centimeters)"].values.tolist()
# True_Y = df["50th Percentile Length (in centimeters)"].values.tolist()


df = pd.read_csv("video_games_sales.csv")
True_Y = df["Global_Sales"].values.tolist()

nlp = spacy.load("en_core_web_sm")
# df_crit = df["Critic_Score"].values.tolist() #Average = 69
# df_pub = df["Publisher"].values.tolist()
# df_userscore = df["User_Score"].values.tolist() #Average = 7.1
# df_genre = df["Genre"].values.tolist()
# df_names = df["Name"].values.tolist()
# df_usercount = df["User_Count"].values.tolist() #Average = 162
# df_critcount = df["Critic_Count"].values.tolist() #Average = 26

df['Critic_Score'] = df['Critic_Score'].replace(numpy.nan, 69)
df['User_Score'] = df['User_Score'].replace(numpy.nan, 7.1)
df['Critic_Count'] = df['Critic_Count'].replace(numpy.nan, 26)
df['User_Count'] = df['User_Count'].replace(numpy.nan, 162)

df_cats = df[["Name", "Genre", "Publisher", "Critic_Score", "Critic_Count", "User_Score", "User_Count"]]
df_list = df_cats.values.tolist()
# df_cats = pd.get_dummies(df_cats)
# print(df_cats[0])
# print()
print(pd.get_dummies(df_cats))

print_c = 0
for row in df_list:
    print(row)
    print_c+= 1
    if print_c == 10:
        break

# for token in df_cats:
#     print(token)


# vocabulary = set()
# for genre in genres:
#     doc = nlp(str(genre))
#     for token in doc:
#         print(token.text)
#         vocabulary.add(token.text)
#
# for genre in genres:
#     doc = nlp(str(genre))
#     word_frequency = {}
#     for token in doc:
#         if token.text in word_frequency:
#             word_frequency[token.text] += 1
#         else:
#             word_frequency[token.text] = 1
#     # print(word_frequency)
#     bag = []
#     for w in sorted(vocabulary):
#         if w in word_frequency:
#             bag.append(word_frequency[w])
#         else:
#             bag.append(0)
#     print(bag)
#
#
#     t = torch.tensor(bag)
#     print(t)
#
#     t2 = torch.tensor([2,25,42])
#     t3 = torch.cat([t,t2])
#     print(t3)



print("TRUE Y: 50TH%")
print(True_Y)
# print("\nX1*********")
# print(x1)
# Folding_Loop(x1, True_Y)
#
# print("\nx2*********")
# print(x2)
# Folding_Loop(x2, True_Y)
#
# print("\nx3*********")
# print(x3)
# Folding_Loop(x3, True_Y)
#
# print("\nx4*********")
# print(x4)
# Folding_Loop(x4, True_Y)

#
# print("\nx5*********")
# print(x5)
# Folding_Loop(x5, True_Y)



