#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
from tensorflow import keras as keras
import numpy as np
from matplotlib import pyplot as plt
from timeit import default_timer as timer


# In[ ]:


# Целевая функция
def fun(t):
    return np.cos(2.5 * t ** 2 - 5 * t)
    # return np.cos(-3 * t ** 2 + 5 * t + 10)


# In[ ]:


# 2.7.0
print("TensorFlow version:", tf.__version__)

t1 = np.linspace(0, 3.4, 200)
y = fun(t1)

# Sequential API более простой чем Functional и позволяет работать только в
# терминах слоёв.
model = keras.models.Sequential()
model.add(keras.layers.Dense(40, input_dim=1, activation='tanh'))
    # kernel_initializer=keras.initializers.RandomNormal(stddev=0.5,mean=0.0),
    # bias_initializer=keras.initializers.RandomNormal(stddev=0.1,mean=0.0)))
    # bias_initializer=keras.initializers.Zeros()))
# model.add(keras.layers.Dropout(0.05, input_shape=(dim,dim)))
model.add(keras.layers.Dense(12, activation='tanh'))
model.add(keras.layers.Dense(1, activation='linear'))
    # kernel_initializer=keras.initializers.RandomNormal(stddev=0.5,mean=0.0),
    # bias_initializer=keras.initializers.RandomNormal(stddev=0.1,mean=0.0)))
    # bias_initializer=keras.initializers.Zeros()))

# Вывести в консоль информацию о построенной модели
model.summary()

batch_size = 10
epochs = 2000

# mse = mean square error, mae = mean absolute error
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
# model.compile(loss='mse', optimizer='adagrad', metrics=['mae'])

start = timer()
hist = model.fit(t1, y, batch_size=batch_size, epochs=epochs)
end = timer()


# In[ ]:


t2 = np.linspace(0, 3.5, 2000)
out = model.predict(t2)

print('Время на обучение:', (int)(end - start), 'сек.', 'Эпох:', epochs, 'Ошибка:', min(hist.history['mae']))

plt.plot(t1, y)
plt.plot(t2, out)

plt.figure()
plt.plot(hist.history['loss'])
plt.plot(hist.history['mae'])


# In[ ]:




