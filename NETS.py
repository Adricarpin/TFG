import numpy as np
import pandas as pd
from numpy import split
from numpy import array

from tensorflow import keras
import tensorflow as tf
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.regularizers import l2
#from keras.utils.vis_utils import plot_model

from sklearn.preprocessing import MinMaxScaler



# Split data
def split_dataset(data):
	train, test = data[:-8784], data[-8952:]  # train 14-19, test 2020
	scaler = MinMaxScaler(feature_range=(-1, 1)) # rescale data [-1, 1]
	scaler = scaler.fit(train)
	train = scaler.transform(train)
	test = scaler.transform(test)
	train = array(split(train, len(train)/24))
	test = array(split(test, len(test)/24))
	return train, test, scaler

# Data to supervised learning
def to_supervised_3(set, n_input, n_out=24):
	data = set.reshape((set.shape[0]*set.shape[1], set.shape[2]))
	X, y = list(), list()
	in_start = 0
	for _ in range(len(data)):
		in_end = in_start + n_input
		out_end = in_end[2] + n_out
		if out_end <= len(data):
			x_input = np.stack((np.array(data[(in_end[2] - n_input[0]):(out_end - n_input[0]), 0]),
								np.array(data[(in_end[2] - n_input[1]):(out_end - n_input[1]), 0]),
								np.array(data[(in_end[2] - n_input[2]):(out_end - n_input[2]), 0])), axis=1)
			X.append(x_input)
			y.append(data[in_end[2]:out_end, 0])
		in_start += 24
	return array(X), array(y)


# Data to supervised (for exogenous variables)
def to_supervised(set, n_input, n_out=24):
	data = set.reshape((set.shape[0]*set.shape[1], set.shape[2]))
	X = list()
	in_start = 0
	for _ in range(len(data)):
		in_end = in_start + n_input
		out_end = in_end + n_out
		if out_end <= len(data):
			x_input = data[in_start:in_end, 0]
			x_input = x_input.reshape((len(x_input), 1))
			X.append(x_input)
		in_start += 24
	return array(X)

#CNN
def CNN(train_x_final, train_y):
	n_timesteps, n_features, n_outputs = train_x_final.shape[1], train_x_final.shape[2], train_y.shape[1]
	model = keras.Sequential()
	model.add(keras.layers.Conv1D(filters=100, kernel_size=4,kernel_regularizer=l2(0.0004),bias_regularizer=l2(0.0004),
                                  padding="same", activation='relu', input_shape=(n_timesteps, n_features)))
	model.add(keras.layers.MaxPooling1D(pool_size=2))
	model.add(keras.layers.Flatten())
	model.add(keras.layers.Dense(500, activation='relu', kernel_regularizer=l2(0.0004), bias_regularizer=l2(0.0004)))
	model.add(keras.layers.Dense(n_outputs))
#	tf.keras.utils.plot_model(model, to_file='CNN.png', show_shapes=True, show_layer_names=True)
	opt=keras.optimizers.Adam(learning_rate=0.001)
	model.compile(loss='mae', optimizer=opt)
	model.fit(train_x_final, train_y, epochs=40, batch_size=30, verbose=1,callbacks=[callback, callback2])
	return model

#RNN
def RNN(train_x_final, train_y):
	n_timesteps, n_features, n_outputs = train_x_final.shape[1], train_x_final.shape[2], train_y.shape[1]
	model = keras.Sequential()
	model.add(keras.layers.GRU(300, activation='relu', input_shape=(n_timesteps, n_features),
							   kernel_regularizer=l2(0.0002),bias_regularizer=l2(0.0002)))
	model.add(keras.layers.Dense(300, activation='relu', kernel_regularizer=l2(0.0002),bias_regularizer=l2(0.0002)))
	model.add(keras.layers.Dense(n_outputs))
#	tf.keras.utils.plot_model(model, to_file='RNN.png', show_shapes=True, show_layer_names=True)
	opt=keras.optimizers.Adam(learning_rate=0.001)
	model.compile(loss='mae', optimizer=opt)
	model.fit(train_x_final, train_y, epochs=40, batch_size=30, verbose=1,callbacks=[callback, callback2])
	return model


# CNN and RNN
def Hibrid(train_x_final, train_y):
	n_timesteps, n_features, n_outputs = train_x_final.shape[1], train_x_final.shape[2], train_y.shape[1]
	model = keras.Sequential()
	model.add(keras.layers.Conv1D(filters=100, kernel_size=4, kernel_regularizer=l2(0.0004),bias_regularizer=l2(0.0004),
                            padding="same", activation='relu', input_shape=(n_timesteps, n_features)))
	model.add(keras.layers.MaxPooling1D(pool_size=2))
	model.add(keras.layers.Flatten())
	model.add(keras.layers.RepeatVector(n_outputs))
	model.add(keras.layers.GRU(300, activation='relu', return_sequences=True,
							   kernel_regularizer=l2(0.0002), bias_regularizer=l2(0.0002)    ))
	model.add(keras.layers.TimeDistributed(keras.layers.Dense(300, activation='relu',
							   kernel_regularizer=l2(0.0004), bias_regularizer=l2(0.0004)    )))
	model.add(keras.layers.TimeDistributed(keras.layers.Dense(1)))
#	tf.keras.utils.plot_model(model, to_file='CNNandRNN.png', show_shapes=True, show_layer_names=True)
	opt=keras.optimizers.Adam(learning_rate=0.001)
	model.compile(loss='mae', optimizer=opt)
	model.fit(train_x_final, train_y, epochs=40, batch_size=30, verbose=1,callbacks=[callback, callback2])
	return model


# Learning rate schedule
def scheduler(epoch, lr):
  if epoch < 10:
    return lr
  if epoch < 20:
	  return 0.0005
  else:
      return 0.0001

# Early stopping
callback2 = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5, verbose=1)

callback = tf.keras.callbacks.LearningRateScheduler(scheduler)


# Build model, predict and retrain
def built_predict(build_model):
	history_x = train_x_final
	history_y = train_y
	predictions = np.empty((0,0), int)

	for i in range(test_y.shape[0]):
		if i % 92 == 0 : #quadrimestral
			model = build_model(history_x, history_y)
		t_x = test_x_final[i]
		t_x = t_x.reshape(1, t_x.shape[0], t_x.shape[1])
		history_x = np.append(history_x, t_x, axis=0)
		last = history_x[-1,]
		last = last.reshape(1, last.shape[0], last.shape[1])
		yhat=model.predict(last)
		predictions = np.append(predictions, yhat)
		t_y = test_y[i]
		t_y = t_y.reshape(1, t_y.shape[0])
		history_y = np.append(history_y, t_y, axis=0)

	predictions = predictions.reshape(int(predictions.shape[0]/24), 24)
	predictions = scaler1.inverse_transform(predictions)
	return predictions


# Read data
df1=pd.read_csv("precios_14_20.txt", sep=",", index_col=0)
df2=pd.read_csv("eolica_14_20.csv", sep=",", index_col=0)
df3=pd.read_csv("demanda_14_20.csv", sep=",", index_col=0)



train, test, scaler1= split_dataset(df1)

# Lags
n_input=array([24,144,168])


train_x, train_y = to_supervised_3(train, n_input)
test_x, test_y = to_supervised_3(test, n_input)


# Exogenous variables processing
def multi(df):
    train, test, scaler = split_dataset(df)
    input = 24
    train_x = to_supervised(train, input)
    test_x = to_supervised(test, input)
    train_x = train_x[int(n_input[2] / 24 - 1):]
    test_x = test_x[int(n_input[2] / 24 - 1):]
    return train_x, test_x

train_x_df2, test_x_df2 =multi(df2)
train_x_df3, test_x_df3 =multi(df3)


#Final data

# Multivariate model
#train_x_final=np.dstack((train_x,train_x_df2, train_x_df3))
#test_x_final=np.dstack((test_x,test_x_df2, test_x_df3))

# Univariante model
train_x_final=train_x
test_x_final=test_x


# Train model "n_members" times and save predictions
n_members = 10
for i in range(n_members):
	yhats = built_predict(CNN)
	filename = "predictions/um_CNN" + str(i + 1)
	np.save(filename, yhats)
	print('Saved: %s' % filename)



