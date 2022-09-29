from tensorflow import keras as keras
import numpy as np
from matplotlib import pyplot as plt
from timeit import default_timer as timer
from keras.layers import Layer


class RBFLayer(Layer):
    def __init__(self, output_dim, sigma=0.5, **kwargs):
        self.output_dim = output_dim
        self.sigma = sigma
        super(RBFLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # todo: keras рекомендует инициализировать веса в методе call, почему?
        self.bias = self.add_weight(name='bias',
                                    shape=(self.output_dim, input_shape[1]),
                                    initializer='uniform',
                                    trainable=True)
        super(RBFLayer, self).build(input_shape)

    def call(self, inputs):
        return inputs - self.bias


# Уравнение эллипса в параметрическом виде.
def ellipse(t, a, b, x0, y0):
    x = x0 + a * np.cos(t)
    y = y0 + b * np.sin(t)
    return x, y


# Уравнение параболы в параметрическом виде.
def parabola(t, p, x0, y0):
    x = x0 + t ** 2 / (2. * p)
    y = y0 + t
    return x, y


# Повернуть фигуру на заданный угол.
def rotate(x, y, alpha):
    xr = x * np.cos(alpha) - y * np.sin(alpha)
    yr = x * np.sin(alpha) + y * np.cos(alpha)
    return xr, yr


def main():
    # Вывести номер версии TensorFlow.
    print("Keras version:", keras.__version__)

    # Гиперпараметры.
    epochs = 10

    # Описать модель. Первый слой RBF, второй линейный.
    # todo: обратить внимание на инициализацию, можно словами вместо чисел
    model = keras.models.Sequential()
    model.add(RBFLayer(10, input_dim=2, sigma=0.1))
    model.add(keras.layers.Dense(3, activation='linear',
                                 kernel_initializer='random_normal',  # нормальное распределение
                                 bias_initializer='uniform'))  # равномерное распределение

    # Вывести в консоль информацию о модели.
    model.summary()

    # Скомпилировать модель.
    model.compile(loss='mse', optimizer='adam', metrics=['mae'])

    # Подготовить данные для обучения. Задать параметр t.
    t = np.linspace(0, 2 * np.pi, 200)

    # Сгенерировать точки для первой фигуры.
    x1, y1 = ellipse(t, 0.4, 0.15, 0, 0)
    x1, y1 = rotate(x1, y1, np.pi / 6.)

    # Сгенерировать точки для второй фигуры.
    x2, y2 = ellipse(t, 0.7, 0.5, 0, 0)
    x2, y2 = rotate(x2, y2, np.pi / 3.)

    # Сгенерировать точки для третьей фигуры.
    # x3, y3 = parabola(t, 1., 0., -0.8)
    x3, y3 = ellipse(t, 1., 1., 0., 0.)
    x3, y3 = rotate(x3, y3, np.pi / 2.)

    # Объединить все точки и преобразовать данные в удобный для TF формат. Стоит отметить что каждая строка таблицы
    # имеет формат входная точка [x, y] и идентификатор класса [a, b, c]. a = 1 означает принадлежность первому классу,
    # b = 1 второму и т.д. У нас строгая принадлежность классу поэтому только один класс ненулевой. Однако это не мешает
    # сети выдавать промежуточные состояния.
    d1 = [[[x, y], [1., 0., 0.]] for x, y in zip(x1, y1)]
    d2 = [[[x, y], [0., 1., 0.]] for x, y in zip(x2, y2)]
    d3 = [[[x, y], [0., 0., 1.]] for x, y in zip(x3, y3)]

    # Объединить и перемешать данные случайным образом.
    dataset = d1 + d2 + d3
    np.random.shuffle(dataset)

    # Разбить набор на обучающую и тестовую выборки. Разбиваем в соотношении 80:20.
    mid = int(len(dataset) * 0.8)
    train_data = dataset[:mid]
    # test_data = dataset[mid:]

    # Сформировать входные и выходные данные для обучения.
    train_input = [x[0] for x in train_data]
    train_output = [x[1] for x in train_data]

    # Задать размер пакета как 10% от размера набора для обучения.
    batch_size = 1  # int(len(train_data) * 0.1)

    # Обучить модель.
    time_start = timer()
    hist = model.fit(train_input, train_output, batch_size=batch_size, epochs=epochs)
    time_end = timer()


if __name__ == '__main__':
    main()
