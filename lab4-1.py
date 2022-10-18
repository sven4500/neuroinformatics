from tensorflow import keras as keras
import numpy as np
from matplotlib import pyplot as plt
from timeit import default_timer as timer
from keras.layers import Layer
from keras import backend as back  # expand_dims, exp


mu = []  # центры


class RBFLayer(Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(RBFLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # todo: keras рекомендует инициализировать веса в методе call, почему?
        self.mu = self.add_weight(name='mu',
                                  # shape=(input_shape[1], self.output_dim),
                                  shape=(input_shape[1], self.output_dim),
                                  initializer='uniform',
                                  trainable=True)
        self.sigma = self.add_weight(name='sigma',
                                     shape=(self.output_dim,),
                                     initializer='uniform',
                                     trainable=True)
        super(RBFLayer, self).build(input_shape)

    def call(self, inputs):
        global mu
        diff = back.expand_dims(inputs) - self.mu
        output = back.exp(back.sum(diff ** 2, axis=1) * self.sigma)
        mu = back.eval(self.mu)
        return output

    # def compute_output_shape(self, input_shape):
        # return (self.output_dim, input_shape[0])


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
    epochs = 500

    # Описать модель. Первый слой RBF, второй линейный. Инициализацию весов можно задать словами вместо чисел. По канону
    # выходной слой должен быть линейным. Здесь используем сигмоиду чтобы остаться в диапазоне [0...1].
    model = keras.models.Sequential()
    model.add(RBFLayer(10, input_dim=2))
    model.add(keras.layers.Dense(3, activation='sigmoid',  # linear по канону
                                 kernel_initializer='random_normal',  # нормальное распределение (mean=0.0, stddev=0.05)
                                 bias_initializer='uniform'))  # равномерное распределение (minval=-0.05, maxval=0.05)

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
    batch_size = int(len(train_data) * 0.1)

    # Обучить модель.
    time_start = timer()
    hist = model.fit(train_input, train_output, batch_size=batch_size, epochs=epochs)
    time_end = timer()

    # Вывести краткую статистику обучения.
    print('Время обучения:', int(time_end - time_start), 'с.',
          'Количество эпох:', epochs,
          'Функция потерь MSE:', min(hist.history['loss']),
          'Метрика качества MAE:', min(hist.history['mae']))

    x = np.linspace(-1, 1, 200)
    y = np.linspace(-1, 1, 200)
    xy = [[a, b] for a in x for b in y]

    z = model.predict(xy)
    z = z.reshape((200, 200, 3))

    fig, axes = plt.subplots(2, 2)
    fig.tight_layout()

    axes[0, 0].set_title('Функция потерь')
    axes[0, 0].set_xlabel('Эпоха')
    axes[0, 0].set_ylabel('MSE')
    axes[0, 0].plot(hist.history['loss'])

    axes[0, 1].set_title('Метрика качества')
    axes[0, 1].set_xlabel('Эпоха')
    axes[0, 1].set_ylabel('MAE')
    axes[0, 1].plot(hist.history['mae'])

    axes[1, 0].plot(x1, y1)
    axes[1, 0].plot(x2, y2)
    axes[1, 0].plot(x3, y3)
    axes[1, 0].plot(mu[0], mu[1], '*')
    axes[1, 0].set_aspect(1)

    axes[1, 1].set_title('Скалярное поле')
    axes[1, 1].set_xlabel('x')
    axes[1, 1].set_ylabel('y')
    axes[1, 1].get_xaxis().set_ticks([])
    axes[1, 1].get_yaxis().set_ticks([])
    # axes[1].pcolormesh(X, Y, Z) #, cmap=cm.gray)
    axes[1, 1].imshow(z)
    # axes[1, 1].invert_xaxis()
    axes[1, 1].invert_yaxis()

    plt.show()


if __name__ == '__main__':
    main()
