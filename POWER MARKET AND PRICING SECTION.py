import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
pd.options.mode.chained_assignment = None


df=pd.read_csv("power_market.csv", sep=",")


ofertada = df.loc[df["ofertada_casada"] == "O"]
casada = df.loc[df["ofertada_casada"] == "C"]

demanda_normal = ofertada.loc[ofertada["compra_venta"] == "C"]
oferta_normal = ofertada.loc[ofertada["compra_venta"] == "V"]

demanda_casada = casada.loc[casada["compra_venta"] == "C"]
oferta_casada = casada.loc[casada["compra_venta"] == "V"]


def acumulada(tipo):
    tipo=tipo.reset_index()
    cantidad = tipo["cantidad"]
    cant_acu = list()
    for i in range(len(cantidad)):
        if i == 0:
            cant_acu.append(cantidad[i])
        if i > 0:
            cant_acu.append(cantidad[i] + cant_acu[i - 1])

    tipo["cantidad_acumulada"] = cant_acu
    return tipo


demanda_normal = acumulada(demanda_normal)

oferta_normal = acumulada(oferta_normal)

demanda_casada = acumulada(demanda_casada)

oferta_casada = acumulada(oferta_casada)


plt.plot(demanda_normal["cantidad_acumulada"], demanda_normal["precio"])
plt.plot(oferta_normal["cantidad_acumulada"], oferta_normal["precio"])
plt.plot(demanda_casada["cantidad_acumulada"], demanda_casada["precio"])
plt.plot(oferta_casada["cantidad_acumulada"], oferta_casada["precio"])
plt.ylabel('EUR/MWh')
plt.xlabel('Power energy')
plt.legend(['demand', 'supply',"matched demand", "matched supply"])
plt.title('03-16-2021 01:00')


