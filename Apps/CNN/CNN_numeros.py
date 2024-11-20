# -*- coding: utf-8 -*-
"""
Created on Mon Jan  2 21:27:00 2023

@author: Daniel Nogueira

#x_train /= 255
#x_test /= 255
#from keras.utils import to_categorical
#y_train = to_categorical(y_train, num_classes)
#y_test = to_categorical(y_test, num_classes)

"""

from keras.datasets import mnist
import matplotlib.pyplot as plt
from keras.models import Sequential#, load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
import numpy as np
import pandas as pd
from keras.utils import to_categorical

##################### DATASET #########################
(x_train, y_train), (x_test, y_test) = mnist.load_data()

image_index = 10
print(y_train[image_index])
plt.imshow(x_train[image_index], cmap='Greys')
plt.show()
print(x_train.shape)
print(x_test.shape)

# save input image dimensions
img_rows, img_cols = x_train[0].shape    #  28, 28

# redimensionamento das imagens para um valor padrão
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

num_classes = len(np.unique(y_train)) #10

##################### MODELO #########################

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
     activation='relu',
     input_shape=(img_rows, img_cols, 1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy',
      optimizer='adam',
      metrics=['accuracy'])


##################### TREINO #########################
batch_size = 128
epochs = 10

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1)#,
          #validation_data=(x_test, y_test))

score = model.evaluate(x_train, y_train, verbose=0)
print('Train loss:', score[0])
print('Train accuracy:', score[1])

##################### TESTE #########################
y_pred = model.predict(x_test)

pred = np.round(y_pred)

ind = np.where(pred[115]==1)[0][0]

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# Comparar predições com os valores reais
num_classes = 10
#y_train = to_categorical(y_train, num_classes)
y_test_cat = to_categorical(y_test, num_classes)

ind_test = np.where(y_test_cat[115]==1)[0][0]

print("The predicted value is %i and the real value is %i"%(ind, ind_test))

model.save("model.h5")
model.save_weights("model_weights.h5")

# Organizar predições
yy_pred = np.argmax(pred, axis=1)
df = pd.DataFrame(y_test, columns=['Real'])
df['Pred'] = yy_pred
df['Comparison'] = df.apply(lambda row: 'OK' if row['Real'] == row['Pred'] else 'NOK', axis=1)

indices_nok = df.index[df['Comparison'] == 'NOK'].tolist()

for image_index in indices_nok[:5]:
    print("Valor Real: %i ------> %i (Predito)"%(y_test[image_index],yy_pred[image_index]))
    plt.imshow(x_test[image_index], cmap='Greys')
    plt.show()
    


