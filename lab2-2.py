from tensorflow import keras as keras
import numpy as np
from matplotlib import pyplot as plt
from timeit import default_timer as timer

epochs = 50
D = 4


def noised_signal(t):
    return (1 / 7) * np.sin(-5 * t ** 2 + 10 * t - 5 - np.pi)
    # return 0.5 * np.cos(t ** 2 + 2 * np.pi)


def true_signal(t):
    return np.sin(-5 * t ** 2 + 10 * t - 5)
    # return np.cos(t ** 2)


def main():
    # Вывести номер версии Keras API.
    print("Keras version:", keras.__version__)

    # Описать модель. Моделируем модель ADALINE (adaptive linear neuron). По сути модель представляет персептрон за
    # исключением функции активации: здесь она линейная.
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(1, input_dim=D, activation='linear',
                                 kernel_initializer=keras.initializers.RandomNormal(stddev=0.5,mean=0.0),
                                 bias_initializer=keras.initializers.RandomNormal(stddev=0.5,mean=0.0)))

    # Вывести в консоль информацию о модели.
    model.summary()

    # Скомпилировать модель.
    # todo: важно отметить связь с SGD
    optimizer = keras.optimizers.SGD(learning_rate=0.01)
    model.compile(loss='mse', optimizer=optimizer, metrics=['mae'])

    # Сгенерировать функции зашумлённого сигнала и истинного.
    t = np.arange(0, 2.5, 0.01)
    yn = noised_signal(t).tolist()
    yt = true_signal(t).tolist()

    # Сгенерировать данные для обучения. Входными данными являются последовательности из D дискрет зашумлённого сигнала,
    # а выходными данными соответствующая дискрета истинного сигнала.
    sequence = [yn[i:i + D] for i in range(0, len(yn) - D)]
    predict_gt = [yt[i] for i in range(D, len(yt))]

    # Проверить соответствие размеров. Каждому окну должно соответствовать выходное значение.
    assert len(sequence) == len(predict_gt)

    # Обучить модель.
    time_start = timer()
    hist = model.fit(sequence, predict_gt, batch_size=1, epochs=epochs, shuffle=True)
    time_end = timer()

    # Предсказать истинный сигнал.
    predict = model.predict(sequence).flatten().tolist()  # predict возвращает список списков поэтому делаем flatten

    # Посчитать ошибки предсказания.
    yp = yt[:D] + predict
    errors = [i - j for i, j in zip(predict, predict_gt)]

    # Вывести краткую статистику обучения.
    print('Время обучения:', int(time_end - time_start), 'с.',
          'Количество эпох:', epochs,
          'Функция потерь MSE:', min(hist.history['loss']),
          'Метрика качества MAE:', min(hist.history['mae']))

    # Вывести графики на экран
    fig, axes = plt.subplots(2, 2)
    fig.tight_layout()  # избавиться от перекрывающихся текстов на графиках
    # plt.subplots_adjust(top=0.9, bottom=0.1)

    axes[0, 0].set_title('Функция потерь')
    axes[0, 0].set_xlabel('Эпоха')
    axes[0, 0].set_ylabel('MSE')
    axes[0, 0].plot(hist.history['loss'])

    axes[0, 1].set_title('Метрика качества')
    axes[0, 1].set_xlabel('Эпоха')
    axes[0, 1].set_ylabel('MAE')
    axes[0, 1].plot(hist.history['mae'])

    axes[1, 0].set_xlabel('t')
    axes[1, 0].set_ylabel('y')
    axes[1, 0].plot(t, yn)  # зашумлённый сигнал
    axes[1, 0].plot(t, yt)  # истинный сигнал
    axes[1, 0].plot(t, yp)  # предсказание (фильтрация) ADALINE (должен совпадать с истинным)

    axes[1, 1].set_title('Ошибка фильтрации')
    axes[1, 1].set_xlabel('t')
    axes[1, 1].set_ylabel('y')
    axes[1, 1].plot(errors)

    plt.show()


if __name__ == '__main__':
    main()
