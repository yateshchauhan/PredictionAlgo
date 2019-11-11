#Reference : https://www.kaggle.com/humamfauzi/multiple-stock-prediction-using-single-nn

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
plt.style.use('seaborn-whitegrid')

from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, GRU, Bidirectional
from keras.optimizers import SGD
import math
from sklearn.metrics import mean_squared_error

# Some functions to help out with
def return_rmse(test,predicted):
    rmse = math.sqrt(mean_squared_error(test, predicted))
    print("The root mean squared error is {}.".format(rmse))
    
    return

#import os
#fileList = os.listdir("../input")
    
# First, we get the data
stockList = ["CSU", "NHC"]
df_ = {}
for i in stockList:
    df_[i] = pd.read_csv("/Users/gauravchauhan/stockDataTest/" + i + ".csv", index_col="Date", parse_dates=["Date"])
    #print (df_[i])

def split(dataframe, border, col):
    return dataframe.loc[:border,col], dataframe.loc[border:,col]

df_new = {}
for i in stockList:
    df_new[i] = {}
    df_new[i]["Train"], df_new[i]["Test"] = split(df_[i], "2015", "Close")
    #print (df_new[i]["Train"])
    #print (df_new[i]["Test"])

def nonPredictGraph():   
    for i in stockList:
        plt.figure(figsize=(14,4))
        plt.plot(df_new[i]["Train"])
        plt.plot(df_new[i]["Test"])
        plt.ylabel("Price")
        plt.xlabel("Date")
        plt.legend(["Training Set", "Test Set"])
        plt.title(i + " Closing Stock Price")
 

nonPredictGraph()    
# Scaling the training set


#transform_train = {}
#transform_test = {}
#scaler = {}
#def transformScaller():
#    for num, i in enumerate(stockList):
#        #sc = MinMaxScaler(feature_range=(0,1), copy=True)
#        sc=MinMaxScaler(copy=True, feature_range=(0, 1))
#        a0 = np.array(df_new[i]["Train"])
#        a1 = np.array(df_new[i]["Test"])
#        a0 = a0.reshape(a0.shape[0],1)
#        a1 = a1.reshape(a1.shape[0],1)
#        transform_train[i] = sc.fit_transform(a0)
#        transform_test[i] = sc.fit_transform(a1)
#        scaler[i] = sc
    
#del a0
#del a1
