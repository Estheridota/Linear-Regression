%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


df = pd.read_csv( 'Boston Housing prices.txt', sep="\s+", names=['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV'])
print(df)


#Working on CRIM

#Scatter Plot
X = df['CRIM']
Y = df['MEDV']

plt.scatter(X,Y,color='blue')
plt.xlabel('Crime rate per town')
plt.ylabel('Value of homes in $1000s')
plt.title('Value of homes in town')
plt.show()


#Finding the coefficients
def mean(values):
    return sum(values) / float(len(values))

# initializing our inputs and outputs
X = df['CRIM'].values
Y = df['MEDV'].values

# mean of our inputs and outputs
x_mean = mean(X)
y_mean = mean(Y)

#total number of values
n = len(X)

# using the formula to calculate the b1 and b0
numerator = 0
denominator = 0
for i in range(n):
    numerator += (X[i] - x_mean) * (Y[i] - y_mean)
    denominator += (X[i] - x_mean) ** 2
    
b1 = numerator / denominator
b0 = y_mean - (b1 * x_mean)

#printing the coefficient
print(b1, b0)


#Making predictions
def predict(x):
    return (b0 + b1 * x)
y_pred = predict(df['CRIM'].values)                      
print(y_pred)



#Final Plot
X = df['CRIM']
Y = df['MEDV']

plt.scatter(X,Y,color='blue')
plt.xlabel('Crime rate per town')
plt.ylabel('Value of homes in $1000s')
plt.title('Value of homes in town')

X = df['CRIM'].values
Y = y_pred
plt.plot(X, Y, color='red')

plt.show()

