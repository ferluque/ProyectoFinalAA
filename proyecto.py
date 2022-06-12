#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Jose Heredia Muñoz
@author: Fernando Luque de la Torre

"""
###############################################################################
###############################################################################
#BIBLIOTECAS.
###############################################################################
###############################################################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import validation_curve
from sklearn.dummy import DummyClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import seaborn as sns

from collections import OrderedDict #Para obtener la lista de clases
###############################################################################
###############################################################################
#FIN BIBLIOTECAS.
###############################################################################
###############################################################################

###############################################################################
###############################################################################
#FUNCIONES.
###############################################################################
###############################################################################
#LECTURA DE DATOS
def readData(file_x):
	# Leemos los ficheros	
    #Ponemos header a 1
    datax =pd.read_csv(file_x,sep=",")
    datax = datax.to_numpy()
    x = []
    y = []
    
    i = 0
    for i in datax:
        x.append(i[:68])       #Desde la característica 0 hasta la 16 son parámetros
        y.append(i[68:])         #La característica 16 son etiquetas
	
    return x, y


###############################################################################
###############################################################################
#FIN FUNCIONES.
###############################################################################
###############################################################################




###############################################################################
###############################################################################
#OBTENCIÓN Y EXPLORACIÓN DE LOS DATOS.
###############################################################################
###############################################################################

#Lectura y organización de los datos
x,y = readData("./Datos/default_features_1059_tracks.txt")

#Pasamos los datos a formato numpyarray y dataframe
X = np.array(x)
Y = np.array(y)

dfX = pd.DataFrame(X)
dfY = pd.DataFrame(Y)

#Exploramos los vectores de características.


desc = dfX.describe()
#Calculo de media de medias y std
mean = 0
std = 0 
size = len(desc.columns)
for i in range(68):
    mean+=desc[i][1]
    std+=desc[i][2]
print("La media de las medias es: " + str(mean/size))
print("La media de las desviaciones estándar es: " + str(std/size))


#Intento de detección de outliers.
#Lo que hago es obtener el porcentaje de los valores que se quedan
# por encima/debajo de de 1.5*IQR mas/menos q3/q1

resultado = []

for i in range(size):
    Q1 = desc[i][4]
    Q3 = desc[i][6]
    IQR = Q3-Q1
    factor = IQR*1.5 
    upper_bound = Q3 + factor
    lower_bound = Q1 - factor
    porcentaje = ( len(dfX[dfX[i] > upper_bound]) + len(dfX[dfX[i] < lower_bound]) )/len(dfX)*100
    resultado.append(porcentaje)
    
    
#Media de valores fuera del rango acotado
media_outliers  = np.mean(resultado,axis=0)
print(media_outliers)


'''
#Histogramas para cada variable
for column in dfX:
    sns.distplot(dfX[column], hist_kws=dict(color='plum',    edgecolor="k", linewidth=1))
    plt.show()
    # boxplot

#Gráfica de caja para cada variable
for column in dfX:    
    plt.figure(figsize=(12,4))
    sns.boxplot(dfX[column])
    plt.show()

for column in dfX:
    sns.violinplot(x=column, data=dfX, color='salmon', whis=3)
    plt.tight_layout()
    plt.show()
'''
#Exploración de la variable de salida.


#A CONTINUACIÓN SE ESCRIBE EL CÓDIGO NECESARIO PARA HACER UNA CODIFICACIÓN
# DE LA LATITUD Y LONGITUD DE UN PAIS A UN ESCALAR.
#Pasamos de nparay a una conjunto, que no tiene elementos repetidos
conjunto_clases = set(map(tuple, y))
clases = []
for i in conjunto_clases:
    clases.append(np.array(i))

clases = np.array(clases)
#Codificamos los datos de la salida
y_aux =np.array(y)
count = 0
size = len(y_aux)
labels = []
for k in clases:
    labels.append(count)
    for i in range(size):
        if(k == y_aux[i]).all():
            y_aux[i] = count
    count+=1

y_cod  =[]
for i in y_aux:
    y_cod.append(i[0])
#########################################################################
#Una vez realizada la codificación, pintamos un histograma.
y_cod = pd.DataFrame(y_cod)
plt.rcParams["figure.figsize"]=(15,8)
plt.rcParams["figure.dpi"]=150
plt.hist(y_cod,bins=np.arange(34)-0.5,alpha=0.75,stacked=True,rwidth=0.9,density=True)
plt.xticks(range(33))
plt.xlabel("Clase")
plt.ylabel("Densidad")
plt.title("Histograma para las clases") 
plt.show()

###############################################################################
###############################################################################
#FIN OBTENCIÓN Y EXPLORACIÓN DE LOS DATOS.
###############################################################################
###############################################################################



###############################################################################
###############################################################################
#ANÁLISIS DE PRINCIPALES COMPONENTES (PCA)
###############################################################################
###############################################################################
n_features = len(X[0])
explained_var = []
for i in range(n_features):
    pca = PCA(n_components=i)
    pca.fit(X)
    explained_var.append(pca.explained_variance_ratio_.sum())
    
plt.title("Porcentaje de varianza explicado según número de características")
plt.xlabel("Número de características")
plt.ylabel("Porcentaje de varianza")
plt.plot(range(n_features), explained_var,c='r')
plt.show()

###############################################################################
###############################################################################
# FIN DE ANÁLISIS DE PRINCIPALES COMPONENTES (PCA)
###############################################################################
###############################################################################


###############################################################################
###############################################################################
# PRUEBECILLAS CON MODELOS (NADA DEFINITIVO)
###############################################################################
###############################################################################

y_clases = np.array(y_cod).reshape((1,-1))[0]
# print("Todo con hiperparámetros por defecto")
# print("Ecv de SGDClassifier con Log error: ", cross_val_score(SGDClassifier(loss='log',n_jobs=-1),X,y_clases,n_jobs=-1,scoring='accuracy'))
# print("Ecv de SGDClassifier perceptron: ", cross_val_score(SGDClassifier(loss='perceptron',n_jobs=-1),X,y_clases,n_jobs=-1,scoring='accuracy'))
# print("Ecv de MLP: ", cross_val_score(MLPClassifier(learning_rate='adaptive', warm_start=True,max_iter=500),X,y_clases,n_jobs=-1,scoring='accuracy'))
# print("Ecv de SVC kernel polinomial: ", cross_val_score(SVC(kernel='poly'),X,y_clases,n_jobs=-1,scoring='accuracy'))
# print("Ecv de SVC kernel RBF-gaussiano: ", cross_val_score(SVC(kernel='rbf'),X,y_clases,n_jobs=-1,scoring='accuracy'))
# print("Ecv de Boosting: ", cross_val_score(GradientBoostingClassifier(n_estimators=10), X,y_clases,scoring='accuracy', n_jobs=-1))
# print("Ecv de Random Forest: ", cross_val_score(RandomForestClassifier(n_estimators=500), X,y_clases,scoring='accuracy', n_jobs=-1))
# Este último es el que mejor accuracy ha dado con hasta un 50%




###############################################################################
###############################################################################
## Paso a analizar más detalladamente el MLP
###############################################################################
###############################################################################

# Primero veo cuántas iteraciones son realmente necesarias (menos de 500 me dice que no converge)
# Esto tarda un ratillo, tampoco aporta mucho el análisis, a partir de 500 ya converge y el error no depende de esto

# iterations = range(500,800,50)
# results = []
# for i in range(len(iterations)):
#     results.append(cross_val_score(MLPClassifier(max_iter=iterations[i]),X,y_clases,n_jobs=-1,scoring='accuracy').mean())

# plt.title("Porcentaje de acierto en función de iteraciones máximas")
# plt.xlabel("Iteraciones máximas")
# plt.ylabel("Porcentaje de acierto")
# plt.plot(iterations,results,c='b')
# plt.show()

# Hacemos un gridsearch para estimar más o menos qué valores pueden ser buenos y qué resultados podemos tener

# Tarda también lo suyo, no se cuanto pero tarda

# MLP_params = {
#     'hidden_layer_sizes': [[50],[75],[100]],
#     'activation': ['logistic','tanh','relu'],
#     'solver':['lbfgs','adam'],
#     'alpha':[1e-4,1e-3,1e-2]    
#     }

# MLP_GridSearch = GridSearchCV(MLPClassifier(max_iter=500),MLP_params,n_jobs=-1)
# MLP_GridSearch.fit(X,y_clases)

# print(MLP_GridSearch.cv_results_)

# Los mejores parámetros de los estudiados han sido
# {'activation': 'relu', 'alpha': 0.001, 'hidden_layer_sizes': [100], 'solver': 'adam'}
# print(cross_val_score(MLPClassifier(max_iter=500,alpha=1e-3,hidden_layer_sizes=(100,100)),X,y_clases,scoring='accuracy'))

###############################################################################
###############################################################################
## Analizo ahora el RandomForest
###############################################################################
###############################################################################
# RF_params = {
#     'n_estimators':[500,1000,1500],
#     'criterion': ['gini','entropy','log_loss'],
#     'warm_start':[True,False]
#     }

# RF_GridSearch = GridSearchCV(RandomForestClassifier(n_jobs=-1),RF_params, n_jobs=-1)
# RF_GridSearch.fit(X,y_clases)

# print(RF_GridSearch.best_params_)

# Los mejores parámetros son
# {'criterion': 'gini', 'n_estimators': 500, 'warm_start': True}
print(cross_val_score(RandomForestClassifier(criterion='gini',n_estimators=500,warm_start=True),X,y_clases,scoring='accuracy',n_jobs=-1))

