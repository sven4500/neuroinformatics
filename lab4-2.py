import tensorflow as tf
from tensorflow import keras as keras
import numpy as np
from matplotlib import pyplot as plt
from timeit import default_timer as timer
from keras.layers import Layer
from keras import backend as back  # expand_dims, exp

# Задать гиперпараметры обучения. В качестве размера пакета берём 10% от количества данных для обучения.
batch_size = 15
epochs = 175

mu_x, mu_y = [], []  # центры
x_train = [0.0, 3.4]  # отрезок функции для обучения


class RBFLayer(Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(RBFLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # todo: keras рекомендует инициализировать веса в методе call, почему?
        self.mu = self.add_weight(name='mu',
                                  # shape=(input_shape[1], self.output_dim),
                                  shape=(input_shape[1], self.output_dim,),
                                  # initial_value=np.linspace(x_train[0], x_train[1], self.output_dim),
                                  initializer=keras.initializers.RandomUniform(minval=x_train[0], maxval=x_train[1]),
                                  trainable=True)
        self.sigma = self.add_weight(name='sigma',
                                     shape=(self.output_dim,),
                                     initializer=keras.initializers.RandomNormal(stddev=0.05, mean=0.0),
                                     trainable=True)
        super(RBFLayer, self).build(input_shape)

    def call(self, inputs):
        global mu_x
        diff = back.expand_dims(inputs) - self.mu
        output = back.exp(back.sum(diff ** 2, axis=1) * self.sigma)
        print(inputs.shape, diff.shape, output.shape, self.sigma.shape)
        mu_x = back.eval(self.mu)
        return output


# Целевая функция, которую модель обучается предсказывать.
def fun(t):
    # return np.cos(2.5 * t ** 2 - 5 * t)
    # return np.sin(np.sin(t) * t ** 2)
    # return np.cos(-3 * t ** 2 + 5 * t + 10)
    # return np.sin(0.25 * t ** 2 - 5 * t)
    return np.sin(-np.sin(t) * t ** 2 + t)
    # return np.sin(t ** 2 - 6 * t + 3)
    # return t ** 2 - 6 * t + 3


def main():
    # Вывести номер версии TensorFlow.
    print("TensorFlow version:", tf.__version__)

    # Описать модель. Первый слой RBF, второй линейный. Инициализацию весов можно задать словами вместо чисел. По канону
    # выходной слой должен быть линейным. Здесь используем сигмоиду чтобы остаться в диапазоне [0...1].
    model = keras.models.Sequential()
    model.add(RBFLayer(150, input_dim=1))
    model.add(keras.layers.Dense(1, activation='linear',  # linear по канону
                                 kernel_initializer='random_normal',  # нормальное распределение (mean=0.0, stddev=0.05)
                                 bias_initializer='zeros'))  # равномерное распределение (minval=-0.05, maxval=0.05)

    # Вывести в консоль информацию о модели.
    model.summary()

    # Скомпилировать модель. Стоит обратить внимание на некоторые параметры. Так параметр loss задаёт функцию потерь
    # которая будет минимизирована в процессе обучения. Параметр metrics в процессе обучения никак не участвует и
    # используется пользователем для оценки качества обученной модели.
    model.compile(loss='mse', optimizer='adam', metrics=['mae'])

    # Создать данные для обучения. Стоит обратить внимание на количество исходных точек - их мало. Предполагается, что
    # обученная модель сможет предсказывать промежуточные значения, которых нет в наборе для обучения.
    t1 = np.linspace(x_train[0], x_train[1], 150)
    y1 = fun(t1)

    # Обучить модель.
    time_start = timer()
    hist = model.fit(t1, y1, batch_size=batch_size, epochs=epochs, shuffle=True)
    time_end = timer()

    # Запросить у обученной модели промежуточные точки (точки гораздо меньшего шага).
    t2 = np.linspace(x_train[0], x_train[1], 2000)
    y2 = model.predict(t2)
    gt = fun(t2)

    # Вычислить точки соответствующие mu_x.
    global mu_y
    mu_y = model.predict([[x] for x in mu_x])

    # Вывести краткую статистику обучения.
    print('Время обучения:', int(time_end - time_start), 'с.',
          'Количество эпох:', epochs,
          'Функция потерь MSE:', min(hist.history['loss']),
          'Метрика качества MAE:', min(hist.history['mae']))

    # Вывести графики.
    fig, axes = plt.subplots(2, 2)
    fig.tight_layout()

    axes[0, 0].set_title('Функция потерь')
    axes[0, 0].set_ylabel('MSE')
    axes[0, 0].plot(hist.history['loss'])

    axes[0, 1].set_title('Метрика качества')
    axes[0, 1].set_ylabel('MAE')
    axes[0, 1].plot(hist.history['mae'])

    axes[1, 0].set_title('Данные для обучения')
    axes[1, 0].plot(t1, y1, '.')
    # axes[1, 0].plot(t2, y2)

    axes[1, 1].set_title('Истинные и предсказанные данные, центры РБФ')
    axes[1, 1].plot(t2, gt)
    axes[1, 1].plot(t2, y2, '--')
    axes[1, 1].plot(mu_x.flatten(), mu_y.flatten(), '*')

    plt.show()


if __name__ == '__main__':
    main()
