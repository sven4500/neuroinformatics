import torch
import random  # random
import time  # time()
import torch.nn as nn
import torch.optim as optim  # Adam
import torch.nn.functional as F
import numpy as np
import matplotlib.animation as animation
from timeit import default_timer as timer
from matplotlib import pyplot as plt
from tqdm import tqdm
from PIL import Image


class HopfieldLayer(nn.Module):
    def __init__(self, in_features):
        super(HopfieldLayer, self).__init__()
        self.w = torch.nn.Parameter(torch.zeros(in_features, in_features))
        self.b = torch.nn.Parameter(torch.zeros(in_features))
        self.prev = torch.tensor(torch.zeros(in_features, in_features))

    def set_init_value(self, value):
        self.prev = torch.tensor(value)

    def forward(self, input=0):
        out = torch.matmul(self.prev, self.w)
        out = torch.add(out, self.b)
        out = torch.clamp(out, min=-1, max=1)
        self.prev = torch.tensor(out)
        return out


def load_image(path, width=320, height=240):
    image = Image.open(path)
    image = image.convert('RGB')  # удалить альфа канал, иногда он может присутствовать!
    image = image.resize((width, height), Image.ANTIALIAS)
    image = np.asarray(image, dtype=np.float32)
    image = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.float32)  # получить float32 вместо double
    image = (image - 127.5) / 127.5  # нормализовать [-1..1]
    # np.reshape(image, width * height)
    return image.flatten()


def load_train_data(width=320, height=240):
    output = []
    for i in range(1, 4):
        output += [(load_image('./images/{0}.jpg'.format(i), width, height),
                    load_image('./images/{0}.jpg'.format(i), width, height))]
    return output


def update_fig(*args):
    global im_axes
    if update_fig.counter >= len(frames):
        update_fig.counter = 0
    im_axes.set_array(frames[update_fig.counter])
    update_fig.counter += 1
    return im_axes,


frames = []
im_axes = 0
update_fig.counter = 0


def main():
    # Вывести номер версии PyTorch.
    print('PyTorch version:', torch.__version__)

    # Гиперпараметры.
    epochs = 50
    width, height = 64, 64

    # Задать новое зерно.
    seed = time.time()
    random.seed(seed)
    torch.manual_seed(seed)

    # Создать слой Хопфилда. Слой один поэтому Sequential не нужен.
    model = HopfieldLayer(in_features=width*height)

    # Задать оптимизатор.
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    # optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # train_data = [(load_image('input_image.jpg', width, height), load_image('input_image.jpg', width, height))]
    train_data = load_train_data(width, height)
    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=1, shuffle=True)

    # Перевести модель в состояние обучения.
    model.train()

    #
    train_loss = []

    # Обучить модель.
    time_start = timer()

    for i in range(epochs):
        # create tqdm object to add arbitrary text
        pbar = tqdm(enumerate(train_loader))

        for j, (input, output_gt) in pbar:

            model.set_init_value(input)

            #
            output = model()

            #
            # crit = nn.L1Loss()
            crit = nn.MSELoss()
            loss = crit(output_gt, output)
            train_loss += [loss.item()]

            # propagate error
            optimizer.zero_grad()
            loss.backward()  # retain_graph=True
            optimizer.step()

            pbar.set_description('loss: %f' % (train_loss[-1]))

    time_end = timer()

    # Перевести модель в рабочее состояние
    model.eval()

    global im_axes, frames

    # Задать ожидаемое и зашумлённое входные значения.
    output_gt = train_data[2][0]
    initial = np.copy(output_gt)
    # initial += np.random.normal(0, 0.15, initial.shape)
    initial = (initial + 1) / 2

    model.set_init_value(initial)

    for i in range(10):
        output = model()
        output = output.detach().numpy()  # .flatten()
        output = (output + 1) / 2
        output = np.reshape(output, (height, width))
        frames += [output]

    output_gt = np.reshape(output_gt, (height, width))
    initial = np.reshape(initial, (height, width))

    # Вывести краткую статистику обучения.
    print('Время обучения:', int(time_end - time_start), 'с.',
          'Количество эпох:', epochs,
    )

    fig, axes = plt.subplots(2, 2)
    fig.tight_layout()

    axes[0, 0].set_title('Функция потерь')
    axes[0, 0].set_xlabel('Эпоха')
    axes[0, 0].set_ylabel('MSE')
    axes[0, 0].plot(train_loss)

    axes[0, 1].set_title('Ожидаемое значение')
    axes[0, 1].set_aspect(1)
    axes[0, 1].imshow(output_gt)

    axes[1, 0].set_title('Первоначальное значение')
    axes[1, 0].set_aspect(1)
    axes[1, 0].imshow(initial)

    # https://matplotlib.org/2.1.0/gallery/animation/dynamic_image.html
    axes[1, 1].set_title('Текущее значение')
    axes[1, 1].set_aspect(1)
    im_axes = axes[1, 1].imshow(frames[0], animated=True)

    anim = animation.FuncAnimation(fig, update_fig, interval=50, blit=True)
    plt.show()


if __name__ == '__main__':
   main()
