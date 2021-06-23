import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

#Load databases
df_test_flat=pd.read_csv("df_test_flat.csv", index_col=0, parse_dates=True, squeeze=True)

df_test_reconstructed_flat=pd.read_csv("df_test_reconstructed_flat.csv", index_col=0, parse_dates=True, squeeze=True)

#Example for January
df_test_flat=df_test_flat[0:744] #January

anomalias_true = df_test_flat[df_test_flat.anomaly == True]

df_test_reconstructed_flat=df_test_reconstructed_flat[0:744]   #January


#Plot price
plt.plot(
  df_test_flat.index,
  df_test_flat.price,
  label='price'
);

#plot reconstructed price
plt.plot(
  df_test_reconstructed_flat.index,
  df_test_reconstructed_flat,
  label='reconstructed price'
);


#plot anomalies
sns.scatterplot(
  anomalias_true.index,
  anomalias_true.price,
  color=sns.color_palette()[3],
  s=52,
  label='anomaly'
)
plt.ylabel('price')
plt.legend();

