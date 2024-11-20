# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 16:07:47 2024

@author: fonte
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np

# Passo 1: Carregar e preparar os dados MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalizar os dados
x_train = x_train / 255.0
x_test = x_test / 255.0

# Converter as labels para categóricas (one-hot encoding)
y_train_cat = to_categorical(y_train, 10)
y_test_cat = to_categorical(y_test, 10)

# Passo 2: Construir o modelo da ANN
model = Sequential([
    Flatten(input_shape=(28, 28)),        # Converter as imagens 2D para 1D
    Dense(128, activation='relu'),        # Camada oculta com 128 neurônios e ReLU
    Dense(64, activation='relu'),         # Outra camada oculta com 64 neurônios e ReLU
    Dense(10, activation='softmax')       # Camada de saída com 10 neurônios e softmax
])

# Passo 3: Compilar o modelo
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Passo 4: Treinar o modelo
history = model.fit(x_train, y_train_cat, epochs=10, validation_split=0.2)

# Passo 5: Avaliar o modelo
test_loss, test_acc = model.evaluate(x_test, y_test_cat)
print(f'Test accuracy: {test_acc}')

# Passo 6: Exibir exemplos de predições
def plot_image(predictions_array, true_label, img):
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel(f"{predicted_label} ({true_label})", color=color)

def plot_value_array(predictions_array, true_label):
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')

# Fazer predições
predictions = model.predict(x_test)

# Plotar algumas imagens com suas predições
num_rows = 5
num_cols = 3
num_images = num_rows * num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_image(predictions[i], y_test[i], x_test[i])
    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plot_value_array(predictions[i], y_test[i])
plt.tight_layout()
plt.show()
