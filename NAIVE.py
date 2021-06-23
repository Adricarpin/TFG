import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error

#Naive model where predictions for day t are data values for day t-1

df=pd.read_csv("precios_14_20.txt", sep=",", index_col=0)


def split_data(data):
	train, test = data[0:-8784], data[-8784:]
	train = np.array(np.split(train, len(train)/24))
	test = np.array(np.split(test, len(test)/24))
	return train, test

train, test= split_data(df)

a=train[-1,] #last 24h of train set
a=a.reshape(a.shape[0])

b=test[0:-1] #2020 without last 24h
b=b.reshape(b.shape[0]*b.shape[1])


predicted=np.concatenate((a,b))

actual= test.reshape(test.shape[0]*test.shape[1])

mae = mean_absolute_error(actual, predicted)
rmse = np.sqrt(mean_squared_error(actual, predicted))
mape = mean_absolute_percentage_error(actual, predicted)*100


