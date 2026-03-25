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
    # Print the Keras API version number.
    print("Keras version:", keras.__version__)

    # Define the model. We model an ADALINE (adaptive linear neuron). Essentially it is a perceptron
    # except for the activation function: here it is linear.
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(1, input_dim=D, activation='linear',
                                 kernel_initializer=keras.initializers.RandomNormal(stddev=0.5,mean=0.0),
                                 bias_initializer=keras.initializers.RandomNormal(stddev=0.5,mean=0.0)))

    # Print model information to the console.
    model.summary()

    # Compile the model.
    # todo: note the connection with SGD
    optimizer = keras.optimizers.SGD(learning_rate=0.01)
    model.compile(loss='mse', optimizer=optimizer, metrics=['mae'])

    # Generate the noisy and true signal functions.
    t = np.arange(0, 2.5, 0.01)
    yn = noised_signal(t).tolist()
    yt = true_signal(t).tolist()

    # Generate training data. The inputs are sequences of D samples of the noisy signal,
    # and the outputs are the corresponding samples of the true signal.
    sequence = [yn[i:i + D] for i in range(0, len(yn) - D)]
    predict_gt = [yt[i] for i in range(D, len(yt))]

    # Verify size consistency. Each window must correspond to one output value.
    assert len(sequence) == len(predict_gt)

    # Train the model.
    time_start = timer()
    hist = model.fit(sequence, predict_gt, batch_size=1, epochs=epochs, shuffle=True)
    time_end = timer()

    # Predict the true signal.
    predict = model.predict(sequence).flatten().tolist()  # predict returns a list of lists, so we flatten it

    # Calculate prediction errors.
    yp = yt[:D] + predict
    errors = [i - j for i, j in zip(predict, predict_gt)]

    # Print brief training statistics.
    print('Training time:', int(time_end - time_start), 's.',
          'Epochs:', epochs,
          'Loss MSE:', min(hist.history['loss']),
          'Metric MAE:', min(hist.history['mae']))

    # Display plots
    fig, axes = plt.subplots(2, 2)
    fig.tight_layout()  # avoid overlapping text on plots
    # plt.subplots_adjust(top=0.9, bottom=0.1)

    axes[0, 0].set_title('Loss function')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('MSE')
    axes[0, 0].plot(hist.history['loss'])

    axes[0, 1].set_title('Quality metric')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('MAE')
    axes[0, 1].plot(hist.history['mae'])

    axes[1, 0].set_xlabel('t')
    axes[1, 0].set_ylabel('y')
    axes[1, 0].plot(t, yn)  # noisy signal
    axes[1, 0].plot(t, yt)  # true signal
    axes[1, 0].plot(t, yp)  # ADALINE prediction (filtering) — should match the true signal

    axes[1, 1].set_title('Filtering error')
    axes[1, 1].set_xlabel('t')
    axes[1, 1].set_ylabel('y')
    axes[1, 1].plot(errors)

    plt.show()


if __name__ == '__main__':
    main()
