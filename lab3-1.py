import tensorflow as tf
from tensorflow import keras as keras
import numpy as np
from matplotlib import pyplot as plt
from timeit import default_timer as timer


# Гиперпараметры.
epochs = 1000


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
    print("TensorFlow version:", tf.__version__)

    # Подготовить данные для обучения.
    # Задать параметр t.
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

    # Разбить набор на обучающую и тестовую выборки. Разбиваем в соотношении 80%.
    mid = int(len(dataset) * 0.8)
    train_data = dataset[:mid]
    test_data = dataset[mid:]

    train_input = [x[0] for x in train_data]
    train_output = [x[1] for x in train_data]

    # Описать модель. На вход подаём двухмерную точку поэтому input_dim равен 2.
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(40, input_dim=2, activation='tanh'))
    model.add(keras.layers.Dense(12, activation='tanh'))
    model.add(keras.layers.Dense(3, activation='sigmoid'))

    # Вывести в консоль информацию о модели.
    model.summary()

    # Скомпилировать модель.
    model.compile(loss='mse', optimizer='adam', metrics=['mae'])

    batch_size = int(len(train_data) * 0.1)

    # Обучить модель.
    time_start = timer()
    hist = model.fit(train_input, train_output, batch_size=batch_size, epochs=epochs)
    time_end = timer()

    x = np.linspace(-1, 1, 200)
    y = np.linspace(-1, 1, 200)
    xy = [[a, b] for a in x for b in y]

    z = model.predict(xy)
    z = z.reshape((200, 200, 3))

    # Вывести краткую статистику обучения.
    print('Время обучения:', int(time_end - time_start), 'с.',
          'Количество эпох:', epochs,
          'Функция потерь MSE:', min(hist.history['loss']),
          'Метрика качества MAE:', min(hist.history['mae']))

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
