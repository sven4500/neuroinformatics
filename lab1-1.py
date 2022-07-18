from tensorflow import keras as keras
import numpy as np
from matplotlib import pyplot as plt


def main():
    # Вывести номер версии Keras API.
    print("Keras version:", keras.__version__)

    # Описать модель. Персептрон является одним нейроном поэтому слой в модели тоже всего один. На вход подаём
    # двухмерную точку поэтому input_dim равен 2.
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(1, input_dim=2, activation='sigmoid',
                                 kernel_initializer=keras.initializers.RandomNormal(stddev=0.5,mean=0.0),
                                 bias_initializer=keras.initializers.Zeros()))

    # Вывести в консоль информацию о модели.
    model.summary()

    # Скомпилировать модель. Стоит обратить внимание что в TensorFlow нет функции Хевисайда, которая также не
    # дифференцируемая. Используется сигмоида значение который на выходе округляется в большую или меньшую сторону.
    # todo: про функцию ошибки
    model.compile(loss='mse', optimizer='adam', metrics=['mae'])

    # Подготовить данные для обучения.
    xy = [[1.1, -0.3], [-1.5, 3.3], [0.8, 0.4], [4.1, -2.2], [2.5, 2.5], [-1.2, 0.6]]
    c = [1., 0., 1., 1., 0., 1.]

    # Обучить модель.
    hist = model.fit(xy, c, batch_size=1, epochs=500)

    # Получить веса и смещение после обучения которые потребуются для рисования дискриминанты.
    weights = model.layers[0].get_weights()

    # Вычислить дискриминанту исходя из полученных весов. Всего у нас 3 параметра: 2 весовых коэффициента и
    # 1 коэффициент смещения.
    # todo: про размерности массива weights
    w1, w2 = weights[0][0][0], weights[0][1][0]
    b = weights[1][0]

    # Подготовить данные для отрисовки дискриминанты. Нужно решить линейное уравнение w1*x + w2*y - b = 0. Для отрисовки
    # линии достаточно двух точек. Берём минимальное и максимальное значение по оси X.
    xs = np.array([-5, 5])
    # ys = (-w1 / w2 * xs) + (b / w2)
    ys = (w1 / -w2 * xs) + (b / -w2)

    # Преобразовать данные в удобный для вывода формат. В один массив складываем точки относящиеся к одному классу, в
    # другой к другому. Это позволит в частности каждому набору точек придать собственное цветовое значение.
    xy1 = [xy[i] for i in range(6) if c[i] > 0.5]
    xy2 = [xy[i] for i in range(6) if c[i] < 0.5]

    # todo: здесь стоит обратить внимание на использование *
    plt.plot(*zip(*xy1), 'ro')
    plt.plot(*zip(*xy2), 'bo')
    plt.plot(xs, ys, 'g')
    # todo: скалярное поле классов
    plt.show()


if __name__ == '__main__':
    main()
