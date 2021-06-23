import numpy as np
import pandas as pd
from tensorflow import keras
import tensorflow as tf
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

#split data
def split_dataset(data):
	train, test = data[:-8760], data[-8760:]
	train = np.array(train)
	test = np.array(test)
	return train, test

#data to supervised learning
def to_supervised(train, n_input, n_out=1):
    X, y = list(), list()
    in_start = 0
    for _ in range(len(train)):
        in_end = in_start + n_input
        out_end = in_end[2] + n_out
        if out_end <= len(train):
            x_input = np.stack((np.array(train[(in_end[2]-n_input[0]):(out_end-n_input[0]), 0]),
                               np.array(train[(in_end[2]-n_input[1]):(out_end-n_input[1]), 0]),
                               np.array(train[(in_end[2]-n_input[2]):(out_end-n_input[2]), 0])),axis=1 )
            X.append(x_input)
            y.append(train[in_end[2]:out_end, 0])
        in_start += 1
    return array(X), array(y)

# model
def RNN():
	n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
	model = keras.Sequential()
	model.add(keras.layers.LSTM(10, activation='relu', input_shape=(n_timesteps, n_features)))
	model.add(keras.layers.Dense(4, activation='relu'))
	model.add(keras.layers.Dense(n_outputs))
	opt=keras.optimizers.Adam(learning_rate=0.001)
	model.compile(loss='mae', optimizer=opt)
	history=model.fit(train_x, train_y, epochs=15, batch_size=30, verbose=1, callbacks=[callback,callback2])
	return model, history


# Learning rate schedule
def scheduler(epoch, lr):
  if epoch < 10:
    return lr
  if epoch < 20:
	  return 0.0005
  else:
      return 0.0001


callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

# Early stopping
callback2 = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=6, verbose=1)

# Predictions:
# We add one by one the predictions to data until the end of the day (24 predictions),
# then we replace these predictions by the real values (from the test).
def predictions(n_input):
	data=train
	j=1
	predicciones = np.empty((0,0), int)
	for i in range(len(test)):
		k = len(train) + 24 * j
		test_x = np.stack((np.array(data[-n_input[0], 0]),
                           np.array(data[-n_input[1], 0]),
                           np.array(data[-n_input[2], 0])))
		test_x=test_x.reshape(1,1,3)
		if len(data) < k:
			yhat = model.predict(test_x)
			predicciones=np.append(predicciones,yhat)
			data=np.append(data, yhat, axis=0)
		if len(data) == k:
			data = data[:-24]
			new_test = test[24*(j-1) : 24*j]
			data=np.append(data, new_test, axis=0)
			j += 1
	return predicciones



df = pd.read_csv("precios_luz_14_19.txt", sep=",", index_col=0)

train, test = split_dataset(df)

#Lags selection
n_input=array([1,24,168])

train_x, train_y = to_supervised(train, n_input)

model, history = RNN()

#plot mae per epoch
plt.plot(history.history["loss"])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['loss', 'val_loss'], loc='upper right')
plt.show()


pred = predictions(n_input)

mae=mean_absolute_error(test, pred)

