import numpy as np
import pandas as pd
import seaborn as sns

df1=pd.read_csv("precios_luz_14_19.txt", sep=",", index_col=0)
df2=pd.read_csv("eolica_14_19.csv", sep=",", index_col=0)
df3=pd.read_csv("demanda_14_19.csv", sep=",", index_col=0)
df4=pd.read_csv("energia_total_14_19.csv", sep=",", index_col=0)
df5=pd.read_csv("exptotales_14_19_abs.csv", sep=",", index_col=0)
df6=pd.read_csv("imptotales_14_19.csv", sep=",", index_col=0)
df7=pd.read_csv("solar_14_19.csv", sep=",", index_col=0)

result = pd.concat([df1,df2,df3,df4, df5, df6, df7], axis=1)

result.corr()

sns.heatmap(result.corr(), annot=True, vmin=-1, vmax=1, cmap="vlag")

