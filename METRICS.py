import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
pd.options.mode.chained_assignment = None
from datetime import datetime


# Load test data
test = np.load("test2020.npy")
test = test.reshape(test.shape[0]*test.shape[1])

# Load NN univariate o multivariate
pred_cnn = np.load("mmcnn.npy")
pred_cnn = pred_cnn.reshape(pred_cnn.shape[0]*pred_cnn.shape[1])

pred_rnn = np.load("mmrnn.npy")
pred_rnn = pred_rnn.reshape(pred_rnn.shape[0]*pred_rnn.shape[1])

pred_crnn = np.load("mmcnnandrnn.npy")
pred_crnn = pred_crnn.reshape(pred_crnn.shape[0]*pred_crnn.shape[1])

# Load ARIMA or TFM
pred_arima = pd.read_csv("ARIMA_FINAL.csv")
pred_arima = pred_arima.to_numpy()
pred_arima = pred_arima.transpose()
pred_arima = pred_arima.reshape(pred_arima.shape[0]*pred_arima.shape[1],)



#### CREATE DATABASE FOR TABLES

#MAE
mae_cnn = mean_absolute_error(test, pred_cnn)
mae_rnn = mean_absolute_error(test, pred_rnn)
mae_arima = mean_absolute_error(test, pred_arima)
mae_crnn = mean_absolute_error(test, pred_crnn)


#RMSE
def rmse(predictions):
    rmse = np.sqrt(sum((test-predictions)**2)  / len(test))
    return rmse

rmse_cnn = rmse(pred_cnn)
rmse_rnn = rmse(pred_rnn)
rmse_arima = rmse(pred_arima)
rmse_crnn = rmse(pred_crnn)

#MAPE
def mape(predictions):
    mape = np.mean(np.abs((test - predictions)/test))*100
    return mape

mape_cnn = mape(pred_cnn)
mape_rnn = mape(pred_rnn)
mape_arima = mape(pred_arima)
mape_crnn = mape(pred_crnn)

# Create dataframe

df_error=pd.DataFrame(data={"RMSE": [rmse_arima, rmse_cnn, rmse_rnn, rmse_crnn],
                            "MAE": [mae_arima, mae_cnn, mae_rnn, mae_crnn],
                            "MAPE": [mape_arima, mape_cnn, mape_rnn, mape_crnn]}, index = ["ARIMA", "CNN", "RNN", "CNNandRNN"])





#### CREATE DATABASE FOR BARPLOTS

# Database. Columns: Error, day, hour, month
def data_base(error):
    df1=pd.read_csv("precios_14_20.txt", sep=",", index_col=0)
    df=df1[-8784:]
    error = error.tolist()
    df["error"]=error
    df=df.drop("price", axis = 1)
    index = df.index
    day = list()
    hour = list()
    month = list()
    for i in index:
        day.append(i[0:10])
        hour.append(i[11:13])
        month.append(i[5:7])
    day_dt = [datetime.strptime(x, '%Y-%m-%d') for x in day]
    weekday = [x.weekday() for x in day_dt]
    df["day"] = weekday
    df["hour"] = hour
    df["month"] = month
    return df


# MSE for each of the hours
def mse_hours(predictions):
    mse = (test- predictions)**2
    return mse

mse_hours_cnn =   mse_hours(pred_cnn)
mse_hours_rnn =   mse_hours(pred_rnn)
mse_hours_arima = mse_hours(pred_arima)
mse_hours_crnn =  mse_hours(pred_crnn)


df_cnn = data_base(mse_hours_cnn)
df_rnn = data_base(mse_hours_rnn)
df_arima = data_base(mse_hours_arima)
df_crnn = data_base(mse_hours_crnn)


# Group data by day, hour and month. Output: RMSE per day, hour and month
def group_data(df):
    day_error=df.groupby(["day"])["error"].mean()
    hour_error=df.groupby(["hour"])["error"].mean()
    month_error=df.groupby(["month"])["error"].mean()
    day_error = day_error.tolist()
    hour_error = hour_error.tolist()
    month_error = month_error.tolist()
    return np.sqrt(day_error), np.sqrt(hour_error), np.sqrt(month_error)


day_cnn, hour_cnn, month_cnn = group_data(df_cnn)
day_ARIMA, hour_ARIMA, month_ARIMA = group_data(df_arima)
day_rnn, hour_rnn, month_rnn = group_data(df_rnn)
day_crnn, hour_crnn, month_crnn = group_data(df_crnn)




# Hours NEURAL NETWORKS
x = np.arange(24)
width = 0.20
x_labels = ["00", "01", "02","03", "04", "05", "06","07", "08", "09","10","11","12","13","14","15","16","17","18","19",
      "20","21","22","23"]

plt.bar(x -0.2, hour_cnn, width, color = "SkyBlue" , edgecolor="black")
plt.bar(x , hour_rnn, width, edgecolor="black", color = "IndianRed")
plt.bar(x + 0.2 , hour_crnn, width, edgecolor="black", color = "forestgreen")
plt.title("RMSE per hours")
plt.ylabel("RMSE")
plt.xticks(x, x_labels, rotation=0)
plt.legend(["CNN", "RNN", "CNN+RNN"])
plt.show()




# Hours NN VS ARIMA
x = np.arange(24)
width = 0.40
x_labels = ["00", "01", "02","03", "04", "05", "06","07", "08", "09","10","11","12","13","14","15","16","17","18","19",
      "20","21","22","23"]

plt.bar(x - 0.2, hour_cnn, width, color = "SkyBlue" , edgecolor="black")
plt.bar(x + 0.2, hour_ARIMA, width, color = "IndianRed", edgecolor="black")
plt.title("RMSE per hours")
plt.ylabel("RMSE")
plt.xticks(x, x_labels, rotation=0)
plt.legend(["CNN", "ARIMA"])
plt.show()




### Days
x = np.arange(7)
width = 0.40
x_labels = ["Monday", "Tuesday", "Wednesday", "Thrusday", "Friday", "Saturday", "Sunday"]

plt.bar(x - 0.2, day_cnn, width, color = "SkyBlue" , edgecolor="black")
plt.bar(x + 0.2, day_ARIMA, width, color = "IndianRed", edgecolor="black")
plt.title("RMSE per days")
plt.ylabel("RMSE")
plt.xticks(x, x_labels, rotation=45)
plt.legend(["CNN", "ARIMA"])
plt.show()




### Months
x = np.arange(12)
width = 0.40
x_labels = ["January","February","March","April","May","June","July","August","September","October","November",
            "December"]
plt.bar(x - 0.2, month_cnn, width, color = "SkyBlue" , edgecolor="black")
plt.bar(x + 0.2, month_ARIMA, width, color = "IndianRed", edgecolor="black")
plt.title("RMSE per months")
plt.ylabel("RMSE")
plt.xticks(x, x_labels, rotation=45)
plt.legend(["CNN", "ARIMA"])
plt.show()


