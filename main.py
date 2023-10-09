import torch
import matplotlib.pyplot as plt
import pandas as pd
import sklearn as sk


def line(x):
    # return x
    return 0.0197 * x + 0.5


#torch pip behavior
x = torch.arange(1, 10, 0.1)
print(line(x))
# print(x)
print(x.shape)
x2 = torch.reshape(x, (10, 9))
print(line(x2))
# print(x2)
print(x2.shape)

#matplotlib pip behavior
plt.plot(x, line(x))
# plt.show()


#https://www.cdc.gov/growthcharts/html_charts/lenageinf.htm#males DATA FROM

#pandas pip behavior
df = pd.read_csv("heights_M.csv")
print(df)
# print(df["50th Percentile Length (in centimeters)"])
x = df["Age (in months)"]
y_true = df["50th Percentile Length (in centimeters)"]
print(x)
print(y_true)

#Mean absolute error
def mae(y_true, y_pred):
    return(y_true - y_pred).abs().mean()

y_pred = line(x)
print(mae(y_true, y_pred))


#Proper Experament
df = pd.read_csv("heights_M.csv")
x1 = df["Age (in months)"].values.tolist()
x2 = df["3rd Percentile Length (in centimeters)"].values.tolist()
x3 = df["5th Percentile Length (in centimeters)"].values.tolist()
x4 = df["10th Percentile Length (in centimeters)"].values.tolist()
x5 = df["25th Percentile Length (in centimeters)"].values.tolist()

# T1 = torch.arange(1, 36, x1)
# list = x1.values.tolist()
print("PROPER EXPEREAMENT")
# print(T1)
print(x1)
print(x2)
print(x3)
print(x4)
print(x5)

# m = torch.tensor(x1.values)
print(m)
# print(list)

# technique for doing an experiament
# note were not traning a model just using the line()
#function above

#steps:
# 1. load data
# 2. make a variable for all x values
# 3. make a varable for all y values
# 4. use a for loop and the Kfold function from sklearn
# 5. each time in the loop you'll have:
#   - X_train, X_test, & Y_train, Y_test
#   - we only need X_test and Y_test
#   - to be sure, y+test are the TRUE y_values
#   - so run your line() functions the X_test inputs
#   - compare the mae() function the predictions with Y_test
# 6. average the mae's across all folds
