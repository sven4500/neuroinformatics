import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from timeit import default_timer as timer
from matplotlib import pyplot as plt
from tqdm import tqdm
from matplotlib.widgets import Slider, Button


core_data = None


# https://www.cs.toronto.edu/~kriz/cifar.html
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        # https://stackoverflow.com/questions/37512290/reading-cifar10-dataset-in-batches
        dict = pickle.load(fo, encoding='latin1')
        # dict = pickle.load(fo, encoding='bytes')
    return dict


def load_train_data(path, requested_label):
    datadict = unpickle(path)

    inputs = datadict['data']
    labels = datadict['labels']

    dataset = []
    for image, label in zip(inputs, labels):
        if label == requested_label:
            image = np.asarray(image, dtype=np.float32)
            image = (image - 127.5) / 127.5
            dataset += [(image, image)]

    return dataset


def plain_to_image(image, width=32, height=32):
    image = (image + 1) / 2  # normalize to range [0..1]
    image = np.reshape(image, (3, height, width))  # reshape
    image = np.transpose(image, [1, 2, 0])  # move dimensions to their correct positions
    return image


def on_slider_update(axes, decoder, feature, val):
    global core_data
    core_data[feature] = val
    image = decoder(torch.from_numpy(core_data)).detach().numpy()
    image = plain_to_image(image)
    axes.set_array(image)


def on_button_click(axes_gt, axes_out, axes_mod, encoder, decoder, data):
    global core_data
    image = data[np.random.randint(low=0, high=len(data))][0]
    enc_out = encoder(torch.from_numpy(image)).detach().numpy()
    dec_out = decoder(torch.from_numpy(enc_out)).detach().numpy()
    core_data = np.copy(enc_out)
    axes_gt.set_array(plain_to_image(image))
    axes_out.set_array(plain_to_image(dec_out))
    axes_mod.set_array(plain_to_image(dec_out))
    plt.draw()


def main():
    epochs = 50
    width, height = 32, 32  # matches CIFAR-10 image size
    dim_1 = width * height * 3  # 3 colour components
    dim_2, dim_3 = int(dim_1 * 1.5), int(dim_1 / 32.0)

    encoder = nn.Sequential(
        nn.Linear(in_features=dim_1, out_features=dim_2),
        # nn.Tanh(),
        nn.Linear(in_features=dim_2, out_features=dim_3),
        nn.Tanh(),
    )

    decoder = nn.Sequential(
        nn.Linear(in_features=dim_3, out_features=dim_2),
        # nn.Tanh(),
        nn.Linear(in_features=dim_2, out_features=dim_1),
        nn.Tanh(),
    )

    # Set the optimizer.
    optimizer_enc = optim.Adam(encoder.parameters(), lr=1e-5)
    optimizer_dec = optim.Adam(decoder.parameters(), lr=1e-5)

    # Classes have the following identifiers: 0 - planes, 1 - cars, 2 - birds, 3 - cats, 4 - deer, 5 - dogs,
    # 6 - frogs, 7 - horses, 8 - ships, 9 - trucks
    train_data = []
    train_data += load_train_data(path='cifar-10-batches-py/data_batch_1', requested_label=0)
    # train_data += load_train_data(path='cifar-10-batches-py/data_batch_1', requested_label=2)
    # train_data += load_train_data(path='cifar-10-batches-py/data_batch_1', requested_label=3)
    np.random.shuffle(train_data)

    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=32, shuffle=True)

    # Print brief statistics about the model and training data.
    print('dim_1: %d, dim_2: %d, dim_3: %d, samples: %d' % (dim_1, dim_2, dim_3, len(train_data)))

    # Set the model to training mode.
    encoder.train()
    decoder.train()

    train_loss = []
    time_start = timer()

    # Train the model.
    for i in range(epochs):
        # Create a tqdm object to display custom text.
        pbar = tqdm(enumerate(train_loader))

        for j, (input, output_gt) in pbar:
            #
            output_enc = encoder(input)
            output_dec = decoder(output_enc)

            # Calculate the loss.
            crit = nn.MSELoss()
            loss = torch.sqrt(crit(output_gt, output_dec))
            train_loss += [loss.item()]

            # Update the weights.
            optimizer_enc.zero_grad()
            optimizer_dec.zero_grad()

            loss.backward()

            optimizer_enc.step()
            optimizer_dec.step()

            # Display the loss.
            pbar.set_description('%d. loss: %f' % (i + 1, train_loss[-1]))

    time_end = timer()

    # Print brief training statistics.
    print('Training time:', int(time_end - time_start), 's.',
          'Epochs:', epochs,
    )

    # Set the model to evaluation mode.
    encoder.eval()
    decoder.eval()

    fig, axes = plt.subplots(2, 2)
    fig.tight_layout()

    axes[0, 0].set_title('Loss function')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('MSE')
    axes[0, 0].plot(train_loss)

    axes[0, 1].set_title('Input')
    axes[0, 1].set_aspect(1)
    ax_gt = axes[0, 1].imshow([[0]])

    axes[1, 0].set_title('Mod. output')
    axes[1, 0].set_aspect(1)
    ax_mod = axes[1, 0].imshow([[0]])

    axes[1, 1].set_title('Output')
    axes[1, 1].set_aspect(1)
    ax_out = axes[1, 1].imshow([[0]])

    features = np.random.randint(low=0, high=dim_3, size=3)

    # Add three sliders for modifying three random components of the latent code.
    axs_1 = fig.add_axes([0.85, 0.25, 0.0225, 0.65])
    slid_1 = Slider(ax=axs_1, label='#1', valinit=0, valmin=-1.0, valmax=1.0, orientation='vertical')
    slid_1.on_changed(lambda val : on_slider_update(ax_mod, decoder, features[0], val))

    axs_2 = fig.add_axes([0.9, 0.25, 0.0225, 0.65])
    slid_2 = Slider(ax=axs_2, label='#2', valinit=0, valmin=-1.0, valmax=1.0, orientation='vertical')
    slid_2.on_changed(lambda val : on_slider_update(ax_mod, decoder, features[1], val))

    axs_3 = fig.add_axes([0.95, 0.25, 0.0225, 0.65])
    slid_3 = Slider(ax=axs_3, label='#3', valinit=0, valmin=-1.0, valmax=1.0, orientation='vertical')
    slid_3.on_changed(lambda val : on_slider_update(ax_mod, decoder, features[2], val))

    axs_4 = fig.add_axes([0.8625, 0.1, 0.1, 0.075])
    btn = Button(axs_4, 'Rand.')
    btn.on_clicked(lambda event : on_button_click(ax_gt, ax_out, ax_mod, encoder, decoder, train_data))
    on_button_click(ax_gt, ax_out, ax_mod, encoder, decoder, train_data)

    plt.subplots_adjust(right=0.8)  # free up some space on the plot for the UI controls
    plt.show()


if __name__ == '__main__':
    main()
