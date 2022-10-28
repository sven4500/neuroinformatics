import torch
import random  # random
import time  # time()
import torch.nn as nn
import torch.optim as optim  # Adam
import numpy as np
from timeit import default_timer as timer
from matplotlib import pyplot as plt
from tqdm import tqdm


class ElmanLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(ElmanLayer, self).__init__()
        self.w1 = torch.nn.Parameter(torch.randn(in_features, out_features))
        self.w2 = torch.nn.Parameter(torch.randn(out_features, out_features))
        self.bias = torch.nn.Parameter(torch.randn(out_features))

    def clear_memory(self):
        if hasattr(self, 'prev'):
            delattr(self, 'prev')

    def forward(self, input):
        out = torch.matmul(input, self.w1)
        if hasattr(self, 'prev'):
            d = torch.matmul(self.prev, self.w2)
            out = torch.add(out, self.bias)
            out = torch.add(out, d)
        else:
            out = torch.add(out, self.bias)
        out = torch.tanh(out)
        self.prev = torch.tensor(out)
        return out


def make_signal(r1=1, r2=1, r3=1):
    k1, k2 = np.arange(0, 1, 0.025), np.arange(0.62, 3.14, 0.025)
    p1, p2 = np.sin(4 * np.pi * k1), np.sin(-3 * k2**2 + 10 * k2 - 5)
    t1, t2 = -1 * np.ones(len(p1)), np.ones(len(p2))

    assert len(k1) == len(p1) and len(k1) == len(t1)
    assert len(k2) == len(p2) and len(k2) == len(t2)

    signal = np.concatenate((np.tile(p1, r1), p2, np.tile(p1, r2), p2, np.tile(p1, r3), p2))
    labels = np.concatenate((np.tile(t1, r1), t2, np.tile(t1, r2), t2, np.tile(t1, r3), t2))

    return signal, labels


def get_train_data(signal, labels, window=1):
    signal_seq = [np.array(signal[i:i+window], dtype=np.float32) for i in range(0, len(signal) - window)]
    labels_seq = [np.array(labels[i:i+window], dtype=np.float32) for i in range(0, len(labels) - window)]

    output = [(x, y) for x, y in zip(signal_seq, labels_seq)]
    return output


def main():
    # Вывести номер версии PyTorch.
    print('PyTorch version:', torch.__version__)

    # Гиперпараметры.
    epochs = 100
    window = 5

    # Задать новое зерно.
    seed = time.time()
    random.seed(seed)
    torch.manual_seed(seed)

    # Создать слои.
    elman = ElmanLayer(in_features=window, out_features=8)
    linear = nn.Linear(in_features=8, out_features=window)

    # Объединить слои в сеть.
    model = nn.Sequential(
        elman,
        linear
    )

    # Задать оптимизатор.
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Сгенерировать данные для обучения.
    signal, labels = make_signal(r1=1, r2=3, r3=2)
    train_dataset = get_train_data(signal, labels, window=window)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=1, shuffle=False)

    # Перевести модель в состояние обучения.
    model.train()

    train_loss = []

    # Обучить модель.
    time_start = timer()

    for i in range(epochs):
        # создать объект tqdm, чтобы вывести собственный текст
        pbar = tqdm(enumerate(train_loader))

        # Очистить память модели
        elman.clear_memory()
        last_loss = []

        for j, (input, output_gt) in pbar:
            #
            output = model(input)

            #
            crit = nn.MSELoss()
            loss = torch.sqrt(crit(output_gt, output))
            last_loss += [loss.item()]

            # propagate error
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss += [np.mean(last_loss)]
        # pbar.set_description('loss: %f' % train_loss[-1])

    time_end = timer()

    # Перевести модель в рабочее состояние.
    model.eval()

    # Сбросить память слоя Элмана.
    elman.clear_memory()

    # Обработать сигнал на обученной модели.
    predict = []
    for x, y in train_dataset:
        predict += [model(torch.tensor(x)).detach().numpy().item(0)]

    predict = np.array(predict)
    predict[predict > 0] = 1
    predict[predict < 0] = -1

    # Вывести краткую статистику обучения.
    print('Время обучения:', int(time_end - time_start), 'с.',
          'Количество эпох:', epochs,
    )

    # Вывести графики на экран.
    fig, axes = plt.subplots(2, 2)
    fig.tight_layout()

    axes[0, 0].set_title('Функция потерь')
    axes[0, 0].set_xlabel('Эпоха')
    axes[0, 0].set_ylabel('MSE')
    axes[0, 0].plot(train_loss)

    axes[1, 0].set_title('Маркировка сигнала')
    axes[1, 0].plot(signal)
    axes[1, 0].plot(labels)

    axes[1, 1].set_title('Распознавание сигнала')
    axes[1, 1].plot(signal)
    axes[1, 1].plot(predict)

    plt.show()


if __name__ == '__main__':
    main()
