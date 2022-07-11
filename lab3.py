import tensorflow as tf
from tensorflow import keras as keras
import numpy as np
from matplotlib import pyplot as plt
from timeit import default_timer as timer


# Целевая функция, которую модель обучается предсказывать.
def fun(t):
    return np.cos(2.5 * t ** 2 - 5 * t)
    # return np.cos(-3 * t ** 2 + 5 * t + 10)


def main():
    # Вывести номер версии TensorFlow.
    print("TensorFlow version:", tf.__version__)

    # Sequential API более простой чем Functional и позволяет работать только в терминах слоёв.
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

    # Вывести в консоль информацию о модели.
    model.summary()

    # Скомпилировать модель. Стоит обратить внимание на некоторые параметры. Так параметр loss задаёт функцию потерь
    # которая будет минимизирована в процессе обучения. Параметр metrics в процессе обучения никак не участвует и
    # используется пользователем для оценки качества обученной модели.
    model.compile(loss='mse', optimizer='adam', metrics=['mae'])
    # model.compile(loss='mse', optimizer='adagrad', metrics=['mae'])

    # Создать данные для обучения. Стоит обратить внимание на количество исходных точек - их мало. Предполагается, что
    # обученная модель сможет предсказывать промежуточные значения, которых нет в наборе для обучения.
    t1 = np.linspace(0, 3.4, 50)
    y1 = fun(t1)

    # Задать гиперпараметры обучения. В качестве размера пакета берём 10% от количества данных для обучения.
    batch_size = 5
    epochs = 5000

    # Обучить модель.
    time_start = timer()
    hist = model.fit(t1, y1, batch_size=batch_size, epochs=epochs)
    time_end = timer()

    # Запросить у обученной модели промежуточные точки (точки гораздо меньшего шага).
    t2 = np.linspace(0, 3.5, 2000)
    y2 = model.predict(t2)
    gt = fun(t2)

    # Вывести краткую статистику обучения.
    print('Время обучения:', int(time_end - time_start), 'с.',
          'Количество эпох:', epochs,
          'Функция потерь MSE:', min(hist.history['loss']),
          'Метрика качества MAE:', min(hist.history['mae']))

    # Вывести графики.
    fig, axes = plt.subplots(2, 2)
    axes[0, 0].set_title('Данные для обучения')
    axes[0, 0].plot(t1, y1)
    # axes[0, 0].plot(t2, y2)
    axes[0, 1].set_title('Истинные и предсказанные данные')
    axes[0, 1].plot(t2, gt)
    axes[0, 1].plot(t2, y2, '--')
    axes[1, 0].set_title('Функция потерь')
    axes[1, 0].set_ylabel('MSE')
    axes[1, 0].plot(hist.history['loss'])
    axes[1, 1].set_title('Метрика качества')
    axes[1, 1].set_ylabel('MAE')
    axes[1, 1].plot(hist.history['mae'])
    plt.show()


if __name__ == '__main__':
    main()
