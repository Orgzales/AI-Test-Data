import pandas as pd
import torch
import numpy as np
from sklearn.model_selection import KFold
import torch.nn as nn


df = pd.read_csv("garments_worker_productivity.csv")

# df_data = df['department', 'day', 'team', 'no_of_workers']
# print(df_data)

fold = KFold(n_splits = 5)
X_tests = []
Y_tests = []


def Data_pred(x, y):
    for train_index, test_index in fold.split(x):
        train_x = np.array(x)[train_index]
        train_y = np.array(y)[train_index]
        test_x = np.array(x)[test_index]
        test_y = np.array(y)[test_index]

        model = nn.Linear(3,1)
        loss = nn.MSELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.003)

        Colum_torch = torch.tensor(train_x, dtype=torch.float32).unsqueeze(1)
        y_true_torch = torch.tensor(train_y, dtype=torch.float32).unsqueeze(1)

        for epoch in range(10):
            test = model(Colum_torch)

            epoch_lose = loss(test, y_true_torch)
            optimizer.zero_grad()
            epoch_lose.backward()
            optimizer.step()
            print("epoch - " + str(epoch_lose))



True_y = df["actual_productivity"].values.tolist()
df_test = df["department"]
df_x = pd.get_dummies(df_test, columns=['department']).values.tolist()

print(df)
Data_pred(df_x, True_y)


