from tensorflow import keras as keras
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from timeit import default_timer as timer


# Hyperparameters.
epochs = 1000


def main():
    # Print the Keras API version number.
    print("Keras version:", keras.__version__)

    # Define the model. The perceptron is a single neuron, so there is only one layer in the model. We feed
    # a two-dimensional point as input, so input_dim equals 2.
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(2, input_dim=2, activation='sigmoid',
                                 kernel_initializer=keras.initializers.RandomNormal(stddev=0.5,mean=0.0),
                                 bias_initializer=keras.initializers.Zeros()))

    # Print model information to the console.
    model.summary()

    # Compile the model. Note that TensorFlow has no Heaviside function, which is also not
    # differentiable. A sigmoid is used whose output value is rounded up or down.
    # todo: about the loss function
    model.compile(loss='mse', optimizer='adam', metrics=['mae'])

    # Prepare training data.
    xy = [[-0.5, -2.6], [2.8, 3.8], [4.1, 0.5], [0.9, -3.6], [3.9, -0.5], [-3.0, 3.9], [2.6, 3.8], [-2.2, 1.7]]
    labels = [[0., 0.], [1., 1.], [0., 1.], [0., 0.], [0., 1.], [1., 0.], [1., 1.], [1., 0.]]

    assert len(xy) == len(labels)

    # Convert data to a convenient format for visualization. Points belonging to one class are stored
    # in one array, and points of the other class in another. This gives each set of points its own color.
    xy1 = [xy[i] for i in range(len(xy)) if labels[i] == [0., 0.]]
    xy2 = [xy[i] for i in range(len(xy)) if labels[i] == [0., 1.]]
    xy3 = [xy[i] for i in range(len(xy)) if labels[i] == [1., 0.]]
    xy4 = [xy[i] for i in range(len(xy)) if labels[i] == [1., 1.]]

    # Train the model.
    time_start = timer()
    hist = model.fit(xy, labels, batch_size=1, epochs=epochs)
    time_end = timer()

    # Get the weights and bias after training, needed for drawing the discriminant.
    weights = model.layers[0].get_weights()

    # Calculate the discriminant from the obtained weights. We have 3 parameters in total: 2 weight
    # coefficients and 1 bias coefficient.
    # todo: about the dimensions of the weights array
    w11, w12, w21, w22 = weights[0][0][0], weights[0][1][0], weights[0][0][1], weights[0][1][1]
    b1, b2 = weights[1][0], weights[1][1]

    # Prepare data for drawing the discriminant. Solve the linear equation w1*x + w2*y - b = 0.
    # Two points are sufficient to draw the line. We take minimum and maximum values along the X axis.
    xs = np.array([-5, 5])
    ys1 = -w11 * xs / w12 + b1 / w12
    ys2 = -w21 * xs / w22 + b2 / w22

    # Build a scalar field: query each point in the 2D space for its class membership.
    x_mesh, y_mesh = np.linspace(-5, 5, 200), np.linspace(-5, 5, 200)
    z_mesh = [[y, x] for x in x_mesh for y in y_mesh]
    z_mesh = model.predict(z_mesh)
    z_mesh = np.asarray(z_mesh).reshape((len(x_mesh), len(y_mesh), 2))
    z_mesh = [[0.9 * z_mesh[i][j][0] + 0.1 * z_mesh[i][j][1], 0.1 * z_mesh[i][j][0] + 0.9 * z_mesh[i][j][1], 0.5 * z_mesh[i][j][0] + 0.5 * z_mesh[i][j][1]]
              for i in range(len(x_mesh)) for j in range(len(y_mesh))]
    z_mesh = np.asarray(z_mesh).reshape((len(x_mesh), len(y_mesh), 3))

    # Print brief training statistics.
    print('Training time:', int(time_end - time_start), 's.',
          'Epochs:', epochs,
          'Loss MSE:', min(hist.history['loss']),
          'Metric MAE:', min(hist.history['mae']))

    # Display plots
    fig, axes = plt.subplots(2, 2)

    axes[0, 0].set_title('Loss function')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('MSE')
    axes[0, 0].plot(hist.history['loss'])

    axes[0, 1].set_title('Quality metric')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('MAE')
    axes[0, 1].plot(hist.history['mae'])

    axes[1, 0].set_xlabel('x')
    axes[1, 0].set_ylabel('y')
    axes[1, 0].set_aspect(1)
    axes[1, 0].set_xlim(-5, 5)
    axes[1, 0].set_ylim(-5, 5)
    # todo: note the use of * here
    axes[1, 0].plot(*zip(*xy1), 'ro')
    axes[1, 0].plot(*zip(*xy2), 'bo')
    axes[1, 0].plot(*zip(*xy3), 'go')
    axes[1, 0].plot(*zip(*xy4), 'mo')
    axes[1, 0].plot(xs, ys1, 'g')
    axes[1, 0].plot(xs, ys2, 'r')

    axes[1, 1].set_title('Scalar field')
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
