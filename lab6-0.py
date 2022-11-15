import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm


# SOM (self organised map) - карта Кохонена
class SOM():
    def __init__(self, in_features, width, height):
        self.nodes = np.random.randn(width*height, in_features)
        self.indices = np.array([[x, y] for x in range(0, height) for y in range(0, width)])
        # self.indices = np.array([np.arange(0, width)] * height)
        return

    def update(self, input, radius, lr):
        # Найти BMU (best matching unit) для input. Т.е. индекс узла
        dist = np.linalg.norm(self.nodes - input, axis=1)
        i_min = np.argmin(dist)

        # Найти расстояние от BMU до остальных узлов.
        dist = np.linalg.norm(self.indices - self.indices[i_min], axis=1)

        # Обновить только те узлы которые находятся в пределах воздействия.
        for d, node in zip(dist, self.nodes):
            if d < radius:
                influence = np.exp(-d / (2 * radius))
                node += lr * influence * (input - node)


def generate_train_data(num_samples):
    # train_data = np.random.rand(num_samples, 3)
    train_data = []
    train_data += [[1., 0., 0.]]  # красный
    train_data += [[1., 0.5, 0.]]  # оранж
    train_data += [[0., 1., 0.]]  # синий
    train_data += [[0., 0., 1.]]  # зелёный
    return train_data


def main():
    num_epochs = 200
    num_samples = 10

    width, height = 64, 48

    radius = start_radius = max(width, height) // 2
    lr = start_lr = 1

    model = SOM(in_features=3, width=width, height=height)

    train_data = generate_train_data(num_samples)
    # train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=1, shuffle=True)

    for i in range(1, num_epochs):
        np.random.shuffle(train_data)

        # Создать объект tqdm для вывода дополнительного текста.
        pbar = tqdm(enumerate(train_data))

        for j, input in pbar:
            # Обновить карту.
            model.update(input, radius, lr)

        # Обновить радиус и скорость обучения в конце эпохи.
        radius = start_radius * np.exp(-i / (num_epochs / np.log(start_radius)))
        lr = start_lr * np.exp(-i / num_epochs)

        # pbar.set_description('radius: %f, lr: %f' % (radius, lr))

    fig, axes = plt.subplots(2, 2)
    fig.tight_layout()

    # axes[0, 0].set_title('Функция потерь')
    # axes[0, 0].set_xlabel('Эпоха')
    # axes[0, 0].set_ylabel('MSE')

    # axes[1, 0].set_title('')
    # axes[1, 0].set_aspect(1)
    axes[1, 0].imshow(model.nodes.reshape((height, width, 3)))

    plt.show()


if __name__ == '__main__':
    main()
