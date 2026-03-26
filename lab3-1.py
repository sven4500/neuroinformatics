import tensorflow as tf
from tensorflow import keras as keras
import numpy as np
from matplotlib import pyplot as plt
from timeit import default_timer as timer


# Hyperparameters.
epochs = 1000


# Ellipse equation in parametric form.
def ellipse(t, a, b, x0, y0):
    x = x0 + a * np.cos(t)
    y = y0 + b * np.sin(t)
    return x, y


# Parabola equation in parametric form.
def parabola(t, p, x0, y0):
    x = x0 + t ** 2 / (2. * p)
    y = y0 + t
    return x, y


# Rotate the figure by the given angle.
def rotate(x, y, alpha):
    xr = x * np.cos(alpha) - y * np.sin(alpha)
    yr = x * np.sin(alpha) + y * np.cos(alpha)
    return xr, yr


def main():
    # Print the TensorFlow version number.
    print("TensorFlow version:", tf.__version__)

    # Prepare training data.
    # Set the parameter t.
    t = np.linspace(0, 2 * np.pi, 200)

    # Generate points for the first figure.
    x1, y1 = ellipse(t, 0.4, 0.15, 0, 0)
    x1, y1 = rotate(x1, y1, np.pi / 6.)

    # Generate points for the second figure.
    x2, y2 = ellipse(t, 0.7, 0.5, 0, 0)
    x2, y2 = rotate(x2, y2, np.pi / 3.)

    # Generate points for the third figure.
    # x3, y3 = parabola(t, 1., 0., -0.8)
    x3, y3 = ellipse(t, 1., 1., 0., 0.)
    x3, y3 = rotate(x3, y3, np.pi / 2.)

    # Combine all points and convert data to a TF-friendly format. Note that each row of the table
    # has the format: input point [x, y] and class identifier [a, b, c]. a=1 means membership in
    # class 1, b=1 in class 2, etc. We have strict class membership so only one class is non-zero,
    # but this does not prevent the network from outputting intermediate states.
    d1 = [[[x, y], [1., 0., 0.]] for x, y in zip(x1, y1)]
    d2 = [[[x, y], [0., 1., 0.]] for x, y in zip(x2, y2)]
    d3 = [[[x, y], [0., 0., 1.]] for x, y in zip(x3, y3)]

    # Combine and shuffle data randomly.
    dataset = d1 + d2 + d3
    np.random.shuffle(dataset)

    # Split the dataset into training and test sets. Split ratio is 80%.
    mid = int(len(dataset) * 0.8)
    train_data = dataset[:mid]
    test_data = dataset[mid:]

    train_input = [x[0] for x in train_data]
    train_output = [x[1] for x in train_data]

    # Define the model. We feed a two-dimensional point as input, so input_dim equals 2.
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(40, input_dim=2, activation='tanh'))
    model.add(keras.layers.Dense(12, activation='tanh'))
    model.add(keras.layers.Dense(3, activation='sigmoid'))

    # Print model information to the console.
    model.summary()

    # Compile the model.
    model.compile(loss='mse', optimizer='adam', metrics=['mae'])

    batch_size = int(len(train_data) * 0.1)

    # Train the model.
    time_start = timer()
    hist = model.fit(train_input, train_output, batch_size=batch_size, epochs=epochs)
    time_end = timer()

    x = np.linspace(-1, 1, 200)
    y = np.linspace(-1, 1, 200)
    xy = [[a, b] for a in x for b in y]

    z = model.predict(xy)
    z = z.reshape((200, 200, 3))

    # Print brief training statistics.
    print('Training time:', int(time_end - time_start), 's.',
          'Epochs:', epochs,
          'Loss MSE:', min(hist.history['loss']),
          'Metric MAE:', min(hist.history['mae']))

    fig, axes = plt.subplots(2, 2)
    fig.tight_layout()

    axes[0, 0].set_title('Loss function')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('MSE')
    axes[0, 0].plot(hist.history['loss'])

    axes[0, 1].set_title('Quality metric')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('MAE')
    axes[0, 1].plot(hist.history['mae'])

    axes[1, 0].plot(x1, y1)
    axes[1, 0].plot(x2, y2)
    axes[1, 0].plot(x3, y3)
    axes[1, 0].set_aspect(1)

    axes[1, 1].set_title('Scalar field')
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
