import tensorflow as tf
from tensorflow import keras as keras
import numpy as np
from matplotlib import pyplot as plt
from timeit import default_timer as timer


# Целевая функция
def fun(t):
    return np.cos(2.5 * t ** 2 - 5 * t)
    # return np.cos(-3 * t ** 2 + 5 * t + 10)


if __name__ == '__main__':
    # 2.7.0
    print("TensorFlow version:", tf.__version__)

    t1 = np.linspace(0, 3.5, 250)
    y = fun(t1)

    dim = 48

    # Sequential API более простой чем Functional и позволяет работать только в
    # терминах слоёв.
    model = keras.models.Sequential()
    # model.add(keras.Input(shape=(1,)))
    # model.add(keras.layers.Dropout(0.2, input_shape=(1,)))
    model.add(keras.layers.Dense(dim, input_dim=1, activation='relu'))
    model.add(keras.layers.Dense(dim, activation='relu'))
    model.add(keras.layers.Dense(1, activation='sigmoid'))

    # Вывести в консоль информацию о построенной модели
    model.summary()

    batch_size = 10
    epochs = 1000

    # mse = mean square error, mae = mean absolute error
    model.compile(loss='mse', optimizer='adam', metrics=['mae'])

    start = timer()
    hist = model.fit(t1, y, batch_size=batch_size, epochs=epochs)
    end = timer()
    print('Время на обучение:', end - start)

    # out = model.predict([0.0])
    t2 = np.linspace(0, 3.5, 1000)
    out = model.predict(t2)

    plt.plot(t1, y)
    plt.plot(t2, out)

    plt.figure()
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['mae'])

    x = []
