import tensorflow as tf
from tensorflow import keras as keras
import numpy as np
from matplotlib import pyplot as plt
from timeit import default_timer as timer
from keras.layers import Layer
from keras import backend as back  # expand_dims, exp

# Set training hyperparameters. We use 10% of the training data as the batch size.
batch_size = 15
epochs = 5000

mu_x, mu_y = [], []  # centers
x_train = [0.0, 3.4]  # function interval for training


class RBFLayer(Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(RBFLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # todo: keras recommends initialising weights in the call method, why?
        self.mu = self.add_weight(name='mu',
                                  shape=(input_shape[1], self.output_dim,),
                                  initializer=keras.initializers.RandomUniform(minval=x_train[0], maxval=x_train[1]),
                                  trainable=True)
        self.sigma = self.add_weight(name='sigma',
                                     shape=(self.output_dim,),
                                     initializer='random_normal',
                                     trainable=True)
        self.sw = self.add_weight(name='sw',
                                  shape=(self.output_dim,),
                                  initializer='random_normal',
                                  trainable=True)
        super(RBFLayer, self).build(input_shape)

    def call(self, inputs):
        global mu_x
        diff = back.expand_dims(inputs) - self.mu
        output = back.exp(back.sum(diff ** 2, axis=1) * self.sigma)
        output = output * self.sw
        mu_x = back.eval(self.mu)
        return output


# Target function that the model learns to predict.
def fun(t):
    # return np.cos(2.5 * t ** 2 - 5 * t)
    return np.sin(np.sin(t) * t ** 2)
    # return np.cos(-3 * t ** 2 + 5 * t + 10)
    # return np.sin(0.25 * t ** 2 - 5 * t)
    # return np.sin(-np.sin(t) * t ** 2 + t)
    # return np.sin(t ** 2 - 6 * t + 3)
    # return t ** 2 - 6 * t + 3


def main():
    # Print the TensorFlow version number.
    print("TensorFlow version:", tf.__version__)

    # Define the model. First layer is RBF, second is linear. Weight initialisation can be specified
    # by name. Canonically the output layer should be linear; here we use sigmoid to stay in [0..1].
    model = keras.models.Sequential()
    model.add(RBFLayer(15, input_dim=1))
    model.add(keras.layers.Dense(1, activation='linear',  # linear by convention
                                 kernel_initializer='random_normal',  # normal distribution (mean=0.0, stddev=0.05)
                                 bias_initializer='zeros'))  # uniform distribution (minval=-0.05, maxval=0.05)

    # Print model information to the console.
    model.summary()

    # Compile the model. Note some of the parameters: the loss parameter sets the loss function
    # that will be minimised during training. The metrics parameter is not used during training and
    # is only used by the user to evaluate the quality of the trained model.
    model.compile(loss='mse', optimizer='adam', metrics=['mae'])

    # Create training data. Note the small number of source points. It is assumed that
    # the trained model will be able to predict intermediate values not present in the training set.
    t1 = np.linspace(x_train[0], x_train[1], 150)
    y1 = fun(t1)

    # Train the model.
    time_start = timer()
    hist = model.fit(t1, y1, batch_size=batch_size, epochs=epochs, shuffle=True)
    time_end = timer()

    # Request intermediate points from the trained model (at a much finer step).
    t2 = np.linspace(x_train[0], x_train[1], 2000)
    y2 = model.predict(t2)
    gt = fun(t2)

    # Calculate the points corresponding to mu_x.
    global mu_y
    mu_y = model.predict([[x] for x in mu_x])

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

    axes[1, 1].set_title('True and predicted data, RBF centers')
    axes[1, 1].plot(t2, gt)
    axes[1, 1].plot(t2, y2, '--')
    axes[1, 1].plot(mu_x.flatten(), mu_y.flatten(), '*')

    plt.show()


if __name__ == '__main__':
    main()
