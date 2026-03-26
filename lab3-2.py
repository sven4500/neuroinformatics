import tensorflow as tf
from tensorflow import keras as keras
import numpy as np
from matplotlib import pyplot as plt
from timeit import default_timer as timer


# Set training hyperparameters. We use 10% of the training data as the batch size.
batch_size = 10
epochs = 2500


# Target function that the model learns to predict.
def fun(t):
    # return np.cos(2.5 * t ** 2 - 5 * t)
    return np.sin(np.sin(t) * t ** 2)
    # return np.cos(-3 * t ** 2 + 5 * t + 10)


def main():
    # Print the TensorFlow version number.
    print("TensorFlow version:", tf.__version__)

    # The Sequential API is simpler than the Functional API and works only in terms of layers.
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(40, input_dim=1, activation='tanh'))
    # kernel_initializer=keras.initializers.RandomNormal(stddev=0.5,mean=0.0),
    # bias_initializer=keras.initializers.RandomNormal(stddev=0.1,mean=0.0)))
    # bias_initializer=keras.initializers.Zeros()))
    # model.add(keras.layers.Dropout(0.05, input_shape=(dim,dim)))
    model.add(keras.layers.Dense(12, activation='tanh'))
    model.add(keras.layers.Dense(1, activation='linear'))
    # kernel_initializer=keras.initializers.RandomNormal(stddev=0.5,mean=0.0),
    # bias_initializer=keras.initializers.RandomNormal(stddev=0.1,mean=0.0)))
    # bias_initializer=keras.initializers.Zeros()))

    # Print model information to the console.
    model.summary()

    # Compile the model. Note some of the parameters: the loss parameter sets the loss function
    # that will be minimised during training. The metrics parameter is not used during training and
    # is only used by the user to evaluate the quality of the trained model.
    model.compile(loss='mse', optimizer='adam', metrics=['mae'])
    # model.compile(loss='mse', optimizer='adagrad', metrics=['mae'])

    # Create training data. Note the small number of source points. It is assumed that
    # the trained model will be able to predict intermediate values not present in the training set.
    t1 = np.linspace(0, 3.4, 150)
    y1 = fun(t1)

    # Train the model.
    time_start = timer()
    hist = model.fit(t1, y1, batch_size=batch_size, epochs=epochs)
    time_end = timer()

    # Request intermediate points from the trained model (at a much finer step).
    t2 = np.linspace(0, 3.5, 2000)
    y2 = model.predict(t2)
    gt = fun(t2)

    # Print brief training statistics.
    print('Training time:', int(time_end - time_start), 's.',
          'Epochs:', epochs,
          'Loss MSE:', min(hist.history['loss']),
          'Metric MAE:', min(hist.history['mae']))

    # Display plots.
    fig, axes = plt.subplots(2, 2)
    fig.tight_layout()

    axes[0, 0].set_title('Loss function')
    axes[0, 0].set_ylabel('MSE')
    axes[0, 0].plot(hist.history['loss'])

    axes[0, 1].set_title('Quality metric')
    axes[0, 1].set_ylabel('MAE')
    axes[0, 1].plot(hist.history['mae'])

    axes[1, 0].set_title('Training data')
    axes[1, 0].plot(t1, y1, '.')
    # axes[1, 0].plot(t2, y2)

    axes[1, 1].set_title('True and predicted data')
    axes[1, 1].plot(t2, gt)
    axes[1, 1].plot(t2, y2, '--')

    plt.show()


if __name__ == '__main__':
    main()
