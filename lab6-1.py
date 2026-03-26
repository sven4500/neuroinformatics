import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm


# SOM (self organised map) - Kohonen map.
class SOM():
    def __init__(self, in_features, width, height):
        self.nodes = np.random.randn(width*height, in_features)
        self.indices = np.array([[x, y] for x in range(0, height) for y in range(0, width)])
        # self.indices = np.array([np.arange(0, width)] * height)
        return

    def update(self, input, radius, lr):
        # Find the BMU (best matching unit) for the input, i.e. the node index
        dist = np.linalg.norm(self.nodes - input, axis=1)
        i_min = np.argmin(dist)

        output = dist[i_min]

        # Find the distance from the BMU to all other nodes.
        dist = np.linalg.norm(self.indices - self.indices[i_min], axis=1)

        # Update only the nodes that are within the influence radius.
        for d, node in zip(dist, self.nodes):
            if d < radius:
                influence = np.exp(-d / (2 * radius))
                node += lr * influence * (input - node)

        return output


def update_fig(*args):
    global im_axes
    im_axes.set_array(frames[update_fig.counter])
    update_fig.counter = (update_fig.counter + 1) % len(frames)
    return im_axes,


frames = []
im_axes = 0
update_fig.counter = 0


def generate_train_data(num_samples):
    # train_data = np.random.rand(num_samples, 3)
    train_data = []
    train_data += [[1., 0., 0.]]  # red
    train_data += [[1., 0.5, 0.]]  # orange
    train_data += [[0., 1., 0.]]  # green
    train_data += [[0., 0., 1.]]  # blue
    return train_data


def main():
    num_epochs = 200
    num_samples = 10

    width, height = 64, 48

    radius = start_radius = max(width, height) // 2
    lr = start_lr = 1

    # Create the model object.
    model = SOM(in_features=3, width=width, height=height)

    train_data = generate_train_data(num_samples)

    global frames, im_axes
    learn_radius = []
    learn_lr = []

    for i in range(1, num_epochs):
        np.random.shuffle(train_data)

        # Create a tqdm object to display additional text.
        pbar = tqdm(enumerate(train_data))

        for j, input in pbar:
            # Update the map.
            model.update(input, radius, lr)

        frames += [np.copy(model.nodes).reshape(height, width, 3)]

        learn_radius += [radius]
        learn_lr += [lr]

        # Update radius and learning rate at the end of the epoch.
        radius = start_radius * np.exp(-i / (num_epochs / np.log(start_radius)))
        lr = start_lr * np.exp(-i / num_epochs)

        # pbar.set_description('radius: %f, lr: %f' % (radius, lr))

    fig, axes = plt.subplots(2, 2)
    fig.tight_layout()

    axes[0, 0].set_title('Radius')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].plot(learn_radius)

    axes[0, 1].set_title('Learning rate')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].plot(learn_lr)

    axes[1, 0].set_title('')
    im_axes = axes[1, 0].imshow(frames[0], animated=True)

    axes[1, 1].axis('off')

    anim = animation.FuncAnimation(fig, update_fig, interval=100, blit=True)
    plt.show()


if __name__ == '__main__':
    main()
