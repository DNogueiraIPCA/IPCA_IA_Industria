# -*- coding: utf-8 -*-
"""
Created on Wed May 17 09:24:50 2023

@author: fonte
"""

import pandas as pd

df = pd.read_csv('./dados/admission_dataset.csv')

y = df['Chance of Admit ']
x = df.drop('Chance of Admit ', axis = 1)

x_treino, x_teste = x[0:300], x[300:]
y_treino, y_teste = y[0:300], y[300:]

from keras.models import Sequential
from keras.layers import Dense

# Criando a arquitetura da rede neural:
modelo = Sequential()
modelo.add(Dense(units=3, activation='relu', input_dim=x_treino.shape[1]))
#modelo.add(Dense(units=6, activation='relu'))
#modelo.add(Dense(units=6, activation='relu'))
#modelo.add(Dense(units=6, activation='relu'))
modelo.add(Dense(units=1, activation='linear'))

# Treinando a rede neural:
modelo.compile(loss='mse', optimizer='adam', metrics=['mse'])
resultado = modelo.fit(x_treino, y_treino, epochs=200, batch_size=32,
                       validation_data=(x_teste, y_teste))

score = modelo.evaluate(x_teste, y_teste, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Plotando gráfico do histórico de treinamento

import matplotlib.pyplot as plt

plt.plot(resultado.history['loss'])
plt.plot(resultado.history['val_loss'])
plt.title('Histórico de Treinamento')
plt.ylabel('Função de custo')
plt.xlabel('Épocas de treinamento')
plt.legend(['Erro treino', 'Erro teste'])
plt.show()

pred = modelo.predict(x_teste)

df_pred = pd.DataFrame(pred, columns=['Pred'])
df_pred['Real'] = y_teste.values

plt.plot(df_pred.Pred)
plt.plot(df_pred.Real)
