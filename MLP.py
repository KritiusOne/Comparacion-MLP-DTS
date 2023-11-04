# print("Soy el perceptron multicapa")
from sklearn.neural_network import MLPClassifier
import pandas as pd

data = pd.read_csv("Metabolic Syndrome.csv", header=None)
x = data.iloc[:, 1:]
y = data.iloc[:, 12]
print(x)
print("----------------------------")
print(y)
