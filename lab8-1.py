import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from timeit import default_timer as timer
from matplotlib import pyplot as plt
from tqdm import tqdm
from collections import deque  # deque


class TDL(nn.Module):
    def __init__(self, in_features, delays=1):
        super(TDL, self).__init__()
        self.in_features = in_features
        self.delays = delays
        self.line = deque()
        self.clear()

    def clear(self):
        self.line.clear()
        for i in range(self.delays):
            self.line.append(torch.zeros(self.in_features))

    def push(self, inputs):
        self.line.appendleft(inputs)  # add element to the left

    def forward(self, inputs=0):
        return self.line.pop()  # return and remove element from the right


class NARX(nn.Module):
    def __init__(self, in_features, hi_features, out_features, delay1, delay2):
        super(NARX, self).__init__()

        self.in_features = in_features
        self.hi_features = hi_features
        self.out_features = out_features

        self.line1 = TDL(in_features, delay1)
        self.line2 = TDL(out_features, delay2)

        self.w1 = torch.nn.Parameter(torch.randn(in_features, hi_features))
        self.w2 = torch.nn.Parameter(torch.randn(hi_features, out_features))
        self.w3 = torch.nn.Parameter(torch.randn(out_features, hi_features))

        self.b1 = torch.nn.Parameter(torch.zeros(hi_features))
        self.b2 = torch.nn.Parameter(torch.zeros(out_features))

    def clear(self):
        self.line1.clear()
        self.line2.clear()

    def forward(self, inputs):
        out1 = torch.tanh(self.line1() @ self.w1 + self.line2() @ self.w3 + self.b1)  # tanh
        out2 = out1 @ self.w2 + self.b2  # linear

        self.line1.push(torch.tensor(inputs))  # save a copy of the input in TDL1
        self.line2.push(torch.tensor(out2))  # save the output in TDL2

        return out2


def main():
    # Print the PyTorch version number.
    print('PyTorch version:', torch.__version__)

    model = NARX(5, 10, 5, 3, 3)

    # Set the optimizer.
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    epochs = 100

    # Generate training data.
    t = np.arange(0, 10, 0.01)
    N, w = len(t), 5

    uk, yk = np.sin(t**2), [0]

    for i in range(N - 1):
        yk += [yk[-1] / (1 + yk[-1]**2) + uk[i]]

    uk = np.array(uk, dtype=np.float32)
    yk = np.array(yk, dtype=np.float32)

    train_data = [(uk[i:i+w], yk[i:i+w]) for i in range(N-5)]
    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=1, shuffle=False)

    # Set the model to training mode.
    model.train()

    train_loss = []

    # Train the model.
    time_start = timer()

    for i in range(epochs):
        # create tqdm object to display custom text
        pbar = tqdm(enumerate(train_loader))

        # Reset the delay line state.
        model.clear()

        epoch_loss = []

        for _, (inputs, outputs_gt) in pbar:
            outputs = model(inputs)

            # Calculate the loss.
            crit = nn.MSELoss()
            loss = torch.sqrt(crit(outputs_gt, outputs))
            epoch_loss += [loss.item()]

            # Update weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Print loss for the last epoch.
        train_loss += [np.mean(epoch_loss)]
        pbar.write(' %d. loss: %f' % (i + 1, train_loss[-1]))

    time_end = timer()

    # Set the model to evaluation mode.
    model.eval()
    model.clear()

    predict = model(train_data[0][0]).detach().tolist()

    for x, _ in train_data:
        predict += [model(x).detach().numpy().item(-1)]

    predict = np.array(predict, dtype=np.float32)

    # Print brief training statistics.
    print('Training time:', int(time_end - time_start), 's.',
          'Epochs:', epochs,
    )

    fig, axes = plt.subplots(2, 2)
    fig.tight_layout()

    axes[0, 0].set_title('Loss function')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('MSE')
    axes[0, 0].plot(train_loss)

    axes[0, 1].set_title('Result')
    axes[0, 1].plot(uk)
    axes[0, 1].plot(yk)
    axes[0, 1].plot(predict)

    axes[1, 0].set_title('Error')
    axes[1, 0].plot(predict - yk)

    axes[1, 1].axis('off')

    plt.show()


if __name__ == '__main__':
    main()
