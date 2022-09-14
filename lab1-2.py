from tensorflow import keras as keras
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from timeit import default_timer as timer


# Гиперпараметры.
epochs = 1000


def main():
    # Вывести номер версии Keras API.
    print("Keras version:", keras.__version__)

    # Описать модель. Персептрон является одним нейроном поэтому слой в модели тоже всего один. На вход подаём
    # двухмерную точку поэтому input_dim равен 2.
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(2, input_dim=2, activation='sigmoid',
                                 kernel_initializer=keras.initializers.RandomNormal(stddev=0.5,mean=0.0),
                                 bias_initializer=keras.initializers.Zeros()))

    # Вывести в консоль информацию о модели.
    model.summary()

    # Скомпилировать модель. Стоит обратить внимание что в TensorFlow нет функции Хевисайда, которая также не
    # дифференцируемая. Используется сигмоида значение который на выходе округляется в большую или меньшую сторону.
    # todo: про функцию ошибки
    model.compile(loss='mse', optimizer='adam', metrics=['mae'])

    # Подготовить данные для обучения.
    xy = [[-0.5, -2.6], [2.8, 3.8], [4.1, 0.5], [0.9, -3.6], [3.9, -0.5], [-3.0, 3.9], [2.6, 3.8], [-2.2, 1.7]]
    labels = [[0., 0.], [1., 1.], [0., 1.], [0., 0.], [0., 1.], [1., 0.], [1., 1.], [1., 0.]]

    assert len(xy) == len(labels)

    # Преобразовать данные в удобный для вывода формат. В один массив складываем точки относящиеся к одному классу, в
    # другой к другому. Это позволит в частности каждому набору точек придать собственное цветовое значение.
    xy1 = [xy[i] for i in range(len(xy)) if labels[i] == [0., 0.]]
    xy2 = [xy[i] for i in range(len(xy)) if labels[i] == [0., 1.]]
    xy3 = [xy[i] for i in range(len(xy)) if labels[i] == [1., 0.]]
    xy4 = [xy[i] for i in range(len(xy)) if labels[i] == [1., 1.]]

    # Обучить модель.
    time_start = timer()
    hist = model.fit(xy, labels, batch_size=1, epochs=epochs)
    time_end = timer()

    # Получить веса и смещение после обучения которые потребуются для рисования дискриминанты.
    weights = model.layers[0].get_weights()

    # Вычислить дискриминанту исходя из полученных весов. Всего у нас 3 параметра: 2 весовых коэффициента и
    # 1 коэффициент смещения.
    # todo: про размерности массива weights
    w11, w12, w21, w22 = weights[0][0][0], weights[0][1][0], weights[0][0][1], weights[0][1][1]
    b1, b2 = weights[1][0], weights[1][1]

    # Подготовить данные для отрисовки дискриминанты. Нужно решить линейное уравнение w1*x + w2*y - b = 0. Для отрисовки
    # линии достаточно двух точек. Берём минимальное и максимальное значение по оси X.
    xs = np.array([-5, 5])
    ys1 = -w11 * xs / w12 + b1 / w12
    ys2 = -w21 * xs / w22 + b2 / w22

    # Построить скалярное поле: опрашиваем каждую точку двумерного пространство на принадлежность конкретному классу.
    x_mesh, y_mesh = np.linspace(-5, 5, 200), np.linspace(-5, 5, 200)
    z_mesh = [[y, x] for x in x_mesh for y in y_mesh]
    z_mesh = model.predict(z_mesh)
    z_mesh = np.asarray(z_mesh).reshape((len(x_mesh), len(y_mesh), 2))
    z_mesh = [[0.9 * z_mesh[i][j][0] + 0.1 * z_mesh[i][j][1], 0.1 * z_mesh[i][j][0] + 0.9 * z_mesh[i][j][1], 0.5 * z_mesh[i][j][0] + 0.5 * z_mesh[i][j][1]]
              for i in range(len(x_mesh)) for j in range(len(y_mesh))]
    z_mesh = np.asarray(z_mesh).reshape((len(x_mesh), len(y_mesh), 3))

    # Вывести краткую статистику обучения.
    print('Время обучения:', int(time_end - time_start), 'с.',
          'Количество эпох:', epochs,
          'Функция потерь MSE:', min(hist.history['loss']),
          'Метрика качества MAE:', min(hist.history['mae']))

    # Вывести графики на экран
    fig, axes = plt.subplots(2, 2)

    axes[0, 0].set_title('Функция потерь')
    axes[0, 0].set_xlabel('Эпоха')
    axes[0, 0].set_ylabel('MSE')
    axes[0, 0].plot(hist.history['loss'])

    axes[0, 1].set_title('Метрика качества')
    axes[0, 1].set_xlabel('Эпоха')
    axes[0, 1].set_ylabel('MAE')
    axes[0, 1].plot(hist.history['mae'])

    axes[1, 0].set_xlabel('x')
    axes[1, 0].set_ylabel('y')
    axes[1, 0].set_aspect(1)
    axes[1, 0].set_xlim(-5, 5)
    axes[1, 0].set_ylim(-5, 5)
    # todo: здесь стоит обратить внимание на использование *
    axes[1, 0].plot(*zip(*xy1), 'ro')
    axes[1, 0].plot(*zip(*xy2), 'bo')
    axes[1, 0].plot(*zip(*xy3), 'go')
    axes[1, 0].plot(*zip(*xy4), 'mo')
    axes[1, 0].plot(xs, ys1, 'g')
    axes[1, 0].plot(xs, ys2, 'r')

    axes[1, 1].set_title('Скалярное поле')
    axes[1, 1].set_xlabel('x')
    axes[1, 1].set_ylabel('y')
    axes[1, 1].get_xaxis().set_ticks([])
    axes[1, 1].get_yaxis().set_ticks([])
    axes[1, 1].imshow(z_mesh)
    axes[1, 1].invert_xaxis()
    # axes[1, 1].invert_yaxis()

    plt.show()


if __name__ == '__main__':
    main()
