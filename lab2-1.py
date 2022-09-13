from tensorflow import keras as keras
import numpy as np
from matplotlib import pyplot as plt
from timeit import default_timer as timer


epochs = 50
D = 5
is_recurrent = False


def signal(t):
    return np.sin(-2 * t**2 + 7*t)


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
    # optimizer = keras.optimizers.Adam(learning_rate=0.01)
    optimizer = keras.optimizers.SGD(learning_rate=0.01)
    model.compile(loss='mse', optimizer=optimizer, metrics=['mae'])

    # Сгенерировать функцию.
    t = np.arange(0, 5, 0.025)
    y = signal(t).tolist()

    # Сгенерировать данные для обучения. Берём последовательные окна из D дискрет в качестве входных данных и следующую
    # за окном дискреты в качестве желаемого результата. Учим сеть как бы предсказывать временной ряд. Получаем сеть из
    # D входов и 1 выходом.
    sequence = [y[i:i+D] for i in range(0, len(y) - D)]
    predict_gt = [y[i] for i in range(D, len(y))]

    # Проверить соответствие размеров. Каждому окну должно соответствовать выходное значение.
    assert len(sequence) == len(predict_gt)

    # Обучить модель.
    time_start = timer()
    hist = model.fit(sequence, predict_gt, batch_size=1, epochs=epochs, shuffle=True)
    time_end = timer()

    yp = y[:D]
    errors = []

    # Предсказать последовательность.
    for i in range(0, len(predict_gt)):
        predict = model.predict([yp[-D:]]).item() if is_recurrent else model.predict([sequence[i]]).item()
        yp += [predict]
        errors += [predict - predict_gt[i]]

    # Вывести краткую статистику обучения.
    print('Время обучения:', int(time_end - time_start), 'с.',
          'Количество эпох:', epochs,
          'Функция потерь MSE:', min(hist.history['loss']),
          'Метрика качества MAE:', min(hist.history['mae']))

    # Вывести графики на экран
    fig, axes = plt.subplots(2, 2)
    fig.tight_layout()  # Чтобы избавиться от перекрывающихся названия заголовка и подписи оси
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
    axes[1, 0].plot(t, y)
    axes[1, 0].plot(t, yp)

    axes[1, 1].set_title('Ошибка предсказания')
    axes[1, 1].set_xlabel('t')
    axes[1, 1].set_ylabel('y')
    axes[1, 1].plot(errors)

    plt.show()


if __name__ == '__main__':
    main()
