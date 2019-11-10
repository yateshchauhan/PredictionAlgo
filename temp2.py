
#Install the dependencies
import quandl
import numpy as np 
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_linear_regression

#Get the stock data
df = quandl.get("WIKI/AMZN")
print (df)