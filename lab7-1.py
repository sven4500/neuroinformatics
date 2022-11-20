import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from timeit import default_timer as timer
from matplotlib import pyplot as plt
from tqdm import tqdm


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


def main():
    epochs = 10
    width, height = 32, 32  # соответствует размеру изображений CIFAR-10
    dim_1 = width * height * 3  # 3 цветовых компоненты
    dim_2, dim_3 = dim_1//2, dim_1//3

    encoder = nn.Sequential(
        nn.Linear(in_features=dim_1, out_features=dim_2),
        nn.Linear(in_features=dim_2, out_features=dim_3),
    )

    decoder = nn.Sequential(
        nn.Linear(in_features=dim_3, out_features=dim_2),
        nn.Linear(in_features=dim_2, out_features=dim_1),
    )

    # Задать оптимизатор.
    optimizer_enc = optim.Adam(encoder.parameters(), lr=1e-3)
    optimizer_dec = optim.Adam(decoder.parameters(), lr=1e-3)

    train_data = []
    # train_data += load_train_data(path='cifar-10-batches-py/data_batch_1', requested_label=1)
    train_data += load_train_data(path='cifar-10-batches-py/data_batch_1', requested_label=2)
    # train_data += load_train_data(path='cifar-10-batches-py/data_batch_1', requested_label=3)

    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=10, shuffle=True)

    # Перевести модель в состояние обучения.
    encoder.train()
    decoder.train()

    train_loss = []
    time_start = timer()

    # Обучить модель.
    for i in range(epochs):
        # Создать объект tqdm, чтобы вывести собственный текст.
        pbar = tqdm(enumerate(train_loader))

        for j, (input, output_gt) in pbar:
            #
            output_enc = encoder(input)
            output_dec = decoder(output_enc)

            # Посчитать ошибку.
            crit = nn.MSELoss()
            loss = torch.sqrt(crit(output_gt, output_dec))
            train_loss += [loss.item()]

            # Обновить весовые коэффициенты.
            optimizer_enc.zero_grad()
            optimizer_dec.zero_grad()
            loss.backward()
            optimizer_enc.step()
            optimizer_dec.step()

            # Показать ошибку.
            pbar.set_description('loss: %f' % (train_loss[-1]))

    time_end = timer()

    # Вывести краткую статистику обучения.
    print('Время обучения:', int(time_end - time_start), 'с.',
          'Количество эпох:', epochs,
    )

    # Перевести модель в рабочее состояние.
    encoder.eval()
    decoder.eval()

    # input_test = np.random.uniform(size=dim_3, low=-1.0, high=1.0).astype(np.float32)
    # output_test = decoder(torch.from_numpy(input_test)).detach().numpy()

    core_test = encoder(torch.from_numpy(train_data[0][0])).detach().numpy()
    # core_test[0] = 0
    output_test = decoder(torch.from_numpy(core_test)).detach().numpy()

    output_test = (output_test + 1) / 2
    output_test = np.reshape(output_test, (3, height, width))
    output_test = np.transpose(output_test, [1, 2, 0])

    fig, axes = plt.subplots(2, 2)
    fig.tight_layout()

    axes[0, 0].set_title('Функция потерь')
    axes[0, 0].set_xlabel('Эпоха')
    axes[0, 0].set_ylabel('MSE')
    axes[0, 0].plot(train_loss)

    axes[1, 0].set_aspect(1)
    axes[1, 0].imshow(output_test)

    plt.show()


if __name__ == '__main__':
    main()
