import torch
import random  # random
import time  # time()
import torch.nn as nn
import torch.optim as optim  # Adam
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from timeit import default_timer as timer
from matplotlib import pyplot as plt
from tqdm import tqdm
from PIL import Image


class HopfieldLayer(nn.Module):
    def __init__(self, in_features):
        super(HopfieldLayer, self).__init__()
        self.w = torch.nn.Parameter(torch.randn(in_features, in_features))
        self.b = torch.nn.Parameter(torch.zeros(in_features))

    def set_init_value(self, value):
        self.prev = torch.tensor(value)

    def forward(self, input=0):
        out = torch.matmul(self.prev, self.w)
        out = torch.add(out, self.b)
        out = torch.clamp(out, min=-1, max=1)
        self.prev = torch.tensor(out)
        return out


zero = [
    -1, -1, -1,  1,  1,  1,  1, -1, -1, -1,
    -1, -1,  1,  1,  1,  1,  1,  1, -1, -1,
    -1, -1,  1, -1, -1, -1, -1,  1, -1, -1,
    -1, -1,  1, -1, -1, -1, -1,  1, -1, -1,
    -1, -1,  1, -1, -1, -1, -1,  1, -1, -1,
    -1, -1,  1, -1, -1, -1, -1,  1, -1, -1,
    -1, -1,  1, -1, -1, -1, -1,  1, -1, -1,
    -1, -1,  1, -1, -1, -1, -1,  1, -1, -1,
    -1, -1,  1, -1, -1, -1, -1,  1, -1, -1,
    -1, -1,  1, -1, -1, -1, -1,  1, -1, -1,
    -1, -1,  1,  1,  1,  1,  1,  1, -1, -1,
    -1, -1,  -1, 1,  1,  1,  1, -1, -1, -1,
]


one = [
    -1, -1, -1, -1,  1,  1, -1, -1, -1, -1,
    -1, -1, -1,  1,  1,  1, -1, -1, -1, -1,
    -1, -1,  1,  1,  1,  1, -1, -1, -1, -1,
    -1, -1,  1, -1,  1,  1, -1, -1, -1, -1,
    -1, -1, -1, -1,  1,  1, -1, -1, -1, -1,
    -1, -1, -1, -1,  1,  1, -1, -1, -1, -1,
    -1, -1, -1, -1,  1,  1, -1, -1, -1, -1,
    -1, -1, -1, -1,  1,  1, -1, -1, -1, -1,
    -1, -1, -1, -1,  1,  1, -1, -1, -1, -1,
    -1, -1, -1, -1,  1,  1, -1, -1, -1, -1,
    -1, -1, -1, -1,  1,  1, -1, -1, -1, -1,
    -1, -1, -1,  1,  1,  1,  1, -1, -1, -1,
]


two = [
    -1, -1, -1,  1,  1,  1,  1, -1, -1, -1,
    -1, -1,  1,  1, -1, -1,  1, -1, -1, -1,
    -1, -1,  1, -1, -1, -1,  1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1,  1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1,  1, -1, -1, -1,
    -1, -1, -1, -1, -1,  1, -1, -1, -1, -1,
    -1, -1, -1,  1,  1,  1, -1, -1, -1, -1,
    -1, -1,  1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1,  1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1,  1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1,  1, -1, -1, -1,  1, -1, -1, -1,
    -1, -1,  1,  1,  1,  1,  1, -1, -1, -1,
]


def load_train_data():
    output = []
    output += [(np.array(zero, dtype=np.float32), np.array(zero, dtype=np.float32))]
    # output += [(np.array(one, dtype=np.float32), np.array(one, dtype=np.float32))]
    output += [(np.array(two, dtype=np.float32), np.array(two, dtype=np.float32))]
    return output


def main():
    # Вывести номер версии PyTorch.
    print('PyTorch version:', torch.__version__)

    # Гиперпараметры.
    epochs = 150
    width, height = 10, 12

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
    train_data = load_train_data()
    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=1, shuffle=True)

    # Перевести модель в состояние обучения.
    model.train()

    #
    train_loss = []

    # Обучить модель.
    time_start = timer()

    for i in range(epochs // 2):
        # create tqdm object to add arbitrary text
        pbar = tqdm(enumerate(train_loader))

        for j, (input, output_gt) in pbar:
            model.set_init_value(input)

            for i in range(epochs):
                #
                output = model()

                #
                crit = nn.L1Loss()
                # crit = nn.CrossEntropyLoss()
                # crit = nn.MSELoss()
                loss = crit(output_gt, output)
                train_loss += [loss.item()]

                # propagate error
                optimizer.zero_grad()
                loss.backward()  # retain_graph=True
                optimizer.step()

                with torch.no_grad():
                    model.w[model.w > 0] = 1
                    model.w[model.w < 0] = -1

                pbar.set_description('loss: %f' % (train_loss[-1]))

    time_end = timer()

    # Перевести модель в рабочее состояние
    model.eval()

    output = np.array(two, dtype=np.float32)
    model.set_init_value(output)

    for i in range(35):
        output = model()

    output = output.detach().numpy()  # .flatten()
    output = (output + 1) / 2
    output = np.reshape(output, (height, width))

    output_gt = np.array(two, dtype=np.float32).flatten()
    output_gt = np.reshape(output_gt, (height, width))

    # Вывести краткую статистику обучения.
    print('Время обучения:', int(time_end - time_start), 'с.',
          'Количество эпох:', epochs,
          # 'Функция потерь MSE:', min(hist.history['loss']),
          # 'Метрика качества MAE:', min(hist.history['mae'])
    )

    fig, axes = plt.subplots(2, 2)
    fig.tight_layout()

    axes[0, 0].set_title('Функция потерь')
    axes[0, 0].set_xlabel('Эпоха')
    axes[0, 0].set_ylabel('MSE')
    axes[0, 0].plot(train_loss)

    axes[0, 1].set_title('Метрика качества')
    axes[0, 1].set_xlabel('Эпоха')
    axes[0, 1].set_ylabel('MAE')
    # axes[0, 1].plot(hist.history['mae'])

    # axes[1, 0].plot(x1, y1)
    axes[1, 0].set_aspect(1)
    axes[1, 0].imshow(output_gt)

    axes[1, 1].set_title('')
    axes[1, 1].set_aspect(1)
    axes[1, 1].imshow(output)
    # axes[1, 1].get_xaxis().set_ticks([])
    # axes[1, 1].get_yaxis().set_ticks([])

    plt.show()


if __name__ == '__main__':
    main()
