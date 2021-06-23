import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
pd.options.mode.chained_assignment = None

####
# Code for this section is only a slight variation of the code used for
# graphs in Power market and pricing section.
####


#Data for first graph
df=pd.read_csv("lockdown.csv", sep=",")

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


fig, (ax1, ax2) = plt.subplots(1, 2)

ax1.plot(demanda_normal["cantidad_acumulada"], demanda_normal["precio"])
ax1.plot(oferta_normal["cantidad_acumulada"], oferta_normal["precio"])
ax1.plot(demanda_casada["cantidad_acumulada"], demanda_casada["precio"])
ax1.plot(oferta_casada["cantidad_acumulada"], oferta_casada["precio"])
ax1.legend(['demand', 'supply',"matched demand", "matched supply"], loc= "lower right")
ax1.title.set_text('02-16-2020 16:00')
ax1.grid(linestyle = '--', linewidth = 0.5)
ax1.set_xlim([0,60000])



#Data for second graph
df=pd.read_csv("lockdown-24.csv", sep=",")


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


ax2.plot(demanda_normal["cantidad_acumulada"], demanda_normal["precio"])
ax2.plot(oferta_normal["cantidad_acumulada"], oferta_normal["precio"])
ax2.plot(demanda_casada["cantidad_acumulada"], demanda_casada["precio"])
ax2.plot(oferta_casada["cantidad_acumulada"], oferta_casada["precio"])
ax2.legend(['demand', 'supply',"matched demand", "matched supply"], loc= "lower right")
ax2.title.set_text('02-15-2020 16:00')
ax2.grid(linestyle = '--', linewidth = 0.5)
ax2.set_xlim([0,60000])

