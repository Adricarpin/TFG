import numpy as np
import pandas as pd


# Load CNN predictions
pred_cnn = np.load("mmcnn.npy")
pred_cnn = pred_cnn.reshape(pred_cnn.shape[0]*pred_cnn.shape[1])

# Load ARIMA predictions
pred_arima = pd.read_csv("ARIMA_FINAL.csv")
pred_arima = pred_arima.to_numpy()
pred_arima = pred_arima.transpose()
pred_arima = pred_arima.reshape(pred_arima.shape[0]*pred_arima.shape[1],)

#SIMPLE AVERAGE

predictions = (pred_cnn*0.5 + pred_arima*0.5)

# Save
np.save("ARIMA-CNN.npy", predictions)
