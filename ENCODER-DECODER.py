import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import mean_absolute_error
from matplotlib import pyplot as plt
import seaborn as sns

#Split data in train and test
def split_dataset(data):
	train, test = data[35064:-8784], data[-8784:]
	train = np.array(np.split(train, len(train)/24))
	test = np.array(np.split(test, len(test)/24))
	return train, test

#Encoder-Decoder model
def build_model():
    n_timesteps, n_features, n_outputs = train.shape[1], train.shape[2], train.shape[1]
    model = keras.Sequential()
    model.add(keras.layers.GRU(200, activation='relu', input_shape=(n_timesteps, n_features), return_sequences=True))
    model.add(keras.layers.GRU(100, activation="relu", return_sequences=False))
    model.add(keras.layers.RepeatVector(n_timesteps))
    model.add(keras.layers.GRU(100, activation='relu', return_sequences=True))
    model.add(keras.layers.GRU(200, activation='relu', return_sequences=True))
    model.add(keras.layers.TimeDistributed(keras.layers.Dense(100, activation="relu")))
    model.add(keras.layers.TimeDistributed(keras.layers.Dense(n_features)))
    opt=keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss='mae', optimizer=opt)
    history=model.fit(train, train, epochs=30, batch_size=30, verbose=1,validation_data=(test, test), callbacks=[callback])
    return model, history

#Learning rate schedule
def scheduler(epoch, lr):
  if epoch < 10:
    return lr
  if epoch < 20:
	  return 0.0005
  else:
    return 0.0001

#Early stopping
callback2 = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3, verbose=1)

#Callback for learning rate schedule
callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

#Read the data
df=pd.read_csv("precios_14_20.txt", sep=",", index_col=0)

#Split in train and test
train, test= split_dataset(df)

#Build autoencoder model
model, history=build_model()



#reconstructed training data
train_reconstructed=model.predict(train)

#flatten the data
train_reconstructed_flat=train_reconstructed.reshape(train_reconstructed.shape[0]*train_reconstructed.shape[1],)
train_flat=train.reshape(train.shape[0]*train.shape[1],)

#MAE representation
train_mae= abs(train_reconstructed_flat - train_flat)
plt.hist(train_mae, bins=300)
plt.xlabel("Train MAE")
plt.ylabel("No of samples")
plt.show()






#Reconstructed test
test_reconstructed = model.predict(test)

#Flatten data
test_reconstructed_flat=test_reconstructed.reshape(test_reconstructed.shape[0]*test_reconstructed.shape[1],)
test_flat=test.reshape(test.shape[0]*test.shape[1],)

#MAE representation
test_mae = abs(test_reconstructed_flat - test_flat)
plt.hist(test_mae, bins=300)
plt.xlabel("test MAE")
plt.ylabel("No of samples")
plt.show()

#set the threshold
threshold = np.std(train_mae)*4

anomalies = test_mae > threshold

#Database with prices, index and anomalies
indice=df[-8784:].index
df_test_flat=pd.DataFrame(data=test_flat, columns=["price"], index=indice)
df_test_flat["anomaly"]=anomalies

#Save database
#df_test_flat.to_csv("df_test_flat.csv")

#Database with reconstructed price
df_test_reconstructed_flat = pd.DataFrame(data=test_reconstructed_flat, columns=["reconstructed"], index=indice)

#Save database
#df_test_reconstructed_flat.to_csv("df_test_reconstructed_flat.csv")

