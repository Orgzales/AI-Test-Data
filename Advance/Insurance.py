import pandas as pd
import torch
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy

# def build_model(): # OLD MODEL
#     return torch.nn.Sequential(
#         torch.nn.Linear(11, 1)
#     )

# def build_model(): #New model
#     return torch.nn.Sequential(
#         torch.nn.Linear(11, 100),
#         torch.nn.ReLU(),
#         torch.nn.Linear(100, 1)
#     )

def build_model(): #New NEW model #smaller with same performance is better
    return torch.nn.Sequential(
        torch.nn.Linear(11, 70),
        torch.nn.ReLU(),
        torch.nn.Linear(70, 70),
        torch.nn.ReLU(),
        torch.nn.Linear(70, 1)
    )

df = pd.read_csv("insurance.csv")
print(df)

# ummies for categorical columns: sex, smoker, region
t_sex = torch.tensor(pd.get_dummies(df['sex']).values, dtype=torch.float32)
t_smoker = torch.tensor(pd.get_dummies(df['smoker']).values, dtype=torch.float32)
t_region = torch.tensor(pd.get_dummies(df['region']).values, dtype=torch.float32)

#normalize age, bmi, and children to 0-1 range
t_age = torch.tensor(df['age'].values, dtype=torch.float32).unsqueeze(1) / 100
t_bmi = torch.tensor(df['bmi'].values, dtype=torch.float32).unsqueeze(1) / 100
t_children = torch.tensor(df['children'].values, dtype=torch.float32).unsqueeze(1) / 10

# target is expenses & unsqeeeze to put them all in one little vector
t_expenses = torch.tensor(df['expenses'].values, dtype=torch.float32).unsqueeze(1) / 50000

#concatenate all tensors into one
print(t_age.shape)
print(t_bmi.shape)
print(t_children.shape)
print(t_sex.shape)
print(t_smoker.shape)
print(t_region.shape)
print(t_expenses.shape)

t_input = torch.cat([t_age, t_bmi, t_children, t_sex, t_smoker, t_region], dim=1) #combined inputs into one testing data set
print(t_input.shape)

#Folding Buildhere
kf = KFold(n_splits=5, shuffle=True, random_state=42)

k = 1
for train_index, test_index in kf.split(t_input):
    print(f'FOLD {k}')
    # print(f'Train: {train_index}, Test: {test_index}')
    k = k + 1
    t_train_input = t_input[train_index] #train x
    t_train_target = t_expenses[train_index] #train true Y
    t_test_input = t_input[test_index] #test X
    t_test_target = t_expenses[test_index] #Test true Y

    model = build_model() #Building model size based on the input
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.001) #sgd = GRADIANT DESENT
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01) #sgd = GRADIANT DESENT
    loss_fn = torch.nn.MSELoss() #mean Square = loss

    for epoch in range(100): #change with 100, 500, 1000  (MAKE THE SAME)
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

# separeate 80/20 train and test
t_train_input, t_test_input, t_train_target, t_test_target = train_test_split(t_input, t_expenses, test_size=0.2, random_state=42)

model = build_model()

# optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = torch.nn.MSELoss()

losses_Graph = []
test_losses_Graph = []

for epoch in range(100): #Change with 100, 500, 1000
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

#Graphing while thinking
plt.plot(losses_Graph, label='TRAIN')
plt.plot(test_losses_Graph, label='TEST')
plt.legend()
plt.show()

#Make Predictions
t_pred = model(t_test_input).detach() * 50000 # since the true y is being / by 50000
t_test_target = t_test_target * 50000
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

# root mean squared error as percentage of mean
# print(df_pred['diff'].pow(2).mean() ** 0.5 / df_pred['actual'].mean())



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





