import tensorflow as tf
import random  # random
import time  # time()
import numpy as np
from tensorflow import keras as keras
from keras import backend as back
from timeit import default_timer as timer
from matplotlib import pyplot as plt
from keras.layers import Layer


class ElmanLayer(Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        self.prev = tf.Variable(tf.zeros((1, output_dim)))
        super(ElmanLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.w1 = self.add_weight(name='w1',
                                  shape=(input_shape[1], self.output_dim),
                                  initializer='uniform',
                                  trainable=True)
        self.w2 = self.add_weight(name='w2',
                                  shape=(self.output_dim, self.output_dim),
                                  initializer='uniform',
                                  trainable=True)
        self.b = self.add_weight(name='b',
                                 shape=(self.output_dim, ),
                                 initializer='uniform',
                                 trainable=True)
        super(ElmanLayer, self).build(input_shape)

    def call(self, inputs):
        out = back.dot(inputs, self.w1)
        out = tf.add(out, back.dot(self.prev, self.w2))
        out = back.bias_add(out, self.b)
        out = tf.keras.activations.tanh(out)
        self.prev.assign(out)
        return out

    def clear(self):
        self.prev.assign(tf.zeros((1, self.output_dim)))
        return


def make_signal(r1=1, r2=1, r3=1):
    k1, k2 = np.arange(0, 1, 0.025), np.arange(0.62, 3.14, 0.025)
    p1, p2 = np.sin(4 * np.pi * k1), np.sin(-3 * k2**2 + 10 * k2 - 5)
    t1, t2 = -1 * np.ones(len(p1)), np.ones(len(p2))

    assert len(k1) == len(p1) and len(k1) == len(t1)
    assert len(k2) == len(p2) and len(k2) == len(t2)

    signal = np.concatenate((np.tile(p1, r1), p2, np.tile(p1, r2), p2, np.tile(p1, r3), p2))
    labels = np.concatenate((np.tile(t1, r1), t2, np.tile(t1, r2), t2, np.tile(t1, r3), t2))

    return signal, labels


def main():
    # Вывести номер версии PyTorch.
    print('Keras version:', keras.__version__)

    # Гиперпараметры.
    epochs = 50
    window = 10

    # Задать новое зерно.
    seed = time.time()
    random.seed(seed)

    # Создать слой Элмана.
    elman = ElmanLayer(output_dim=8)

    # Создать модель.
    model = keras.models.Sequential()
    model.add(elman)
    model.add(keras.layers.Dense(window,
                                 activation='linear',
                                 kernel_initializer='random_normal',
                                 bias_initializer='zeros'))

    # Скомпилировать модель.
    model.compile(loss='mse', optimizer='adam', metrics=['mae'])

    # Вывести в консоль информацию о модели.
    # model.summary()

    # Сгенерировать данные для обучения.
    signal, labels = make_signal(r1=1, r2=3, r3=2)

    signal_seq = [np.array(signal[i:i + window], dtype=np.float32).tolist() for i in range(0, len(signal) - window)]
    labels_seq = [np.array(labels[i:i + window], dtype=np.float32).tolist() for i in range(0, len(labels) - window)]

    train_loss, train_mae = [], []

    # Обучить модель. На каждой итерации сбрасываем память Элмана.
    time_start = timer()

    for _ in range(epochs):
        elman.clear()
        hist = model.fit(signal_seq, labels_seq, batch_size=1, epochs=1, shuffle=False)
        train_loss += [hist.history['loss'][0]]
        train_mae += [hist.history['mae'][0]]

    time_end = timer()

    # Вывести краткую статистику обучения.
    print('Время обучения:', int(time_end - time_start), 'с.',
          'Количество эпох:', epochs,
          'Функция потерь MSE:', min(train_loss),
          'Метрика качества MAE:', min(train_mae),
    )

    # Сбросить память слоя Элмана.
    elman.clear()

    # Обработать сигнал на обученной модели.
    predict = []
    for seq in signal_seq:
        predict += [model.predict([seq])[0][0]]

    predict = np.array(predict)
    predict[predict > 0] = 1
    predict[predict < 0] = -1

    # Вывести графики на экран.
    fig, axes = plt.subplots(2, 2)
    fig.tight_layout()

    axes[0, 0].set_title('Функция потерь')
    axes[0, 0].set_xlabel('Эпоха')
    axes[0, 0].set_ylabel('MSE')
    axes[0, 0].plot(train_loss)

    axes[0, 1].set_title('Метрика качества')
    axes[0, 1].set_xlabel('Эпоха')
    axes[0, 1].set_ylabel('MAE')
    axes[0, 1].plot(train_mae)

    axes[1, 0].set_title('Маркировка сигнала')
    axes[1, 0].plot(signal)
    axes[1, 0].plot(labels)

    axes[1, 1].set_title('Распознавание сигнала')
    axes[1, 1].plot(signal)
    axes[1, 1].plot(predict)

    plt.show()


if __name__ == '__main__':
    main()
