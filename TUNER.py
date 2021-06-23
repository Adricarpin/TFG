import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
import kerastuner as kt



df=pd.read_csv("precios_14_20.txt", sep=",", index_col=0)

# split data
def split_dataset(data):
	train, test = data[:-17544], data[-17712:-8784] # test 2019
	train = np.array(np.split(train, len(train)/24))
	test = np.array(np.split(test, len(test)/24))
	return train, test



#data to supervised learning
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
	return np.array(X), np.array(y)

# train the model
def build_model(hp):
	n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
	model = keras.Sequential()
	model.add(keras.layers.LSTM(units=hp.Int('units', min_value=50, max_value=500, step=50),input_shape=(n_timesteps, n_features), activation='relu'))
	model.add(keras.layers.Dense(units=hp.Int('units_dense', min_value=0, max_value=400, step=50), activation='relu'))
	model.add(keras.layers.Dense(n_outputs))
	model.compile(loss='mae', optimizer=keras.optimizers.Adam(hp.Choice('learning_rate', [0.001, 0.0001, 0.00001])))
	return model


train, test= split_dataset(df)

#set lags
n_input=np.array([24,144,168])

train_x, train_y = to_supervised_3(train, n_input)
val_x, val_y = to_supervised_3(test, n_input)

callback2 = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, verbose=1)


tuneo = kt.RandomSearch(
    build_model,
    objective='val_loss',
    max_trials=30,
    executions_per_trial=2,
    directory='my_directory',
    project_name='tuner_RNN')


tuneo.search_space_summary()

tuneo.search(train_x, train_y, epochs = 40, batch_size=30, verbose=1, callbacks=[callback2], validation_data = (val_x, val_y))

tuneo.results_summary()



