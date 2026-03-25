# Neuroinformatics

This repository contains laboratory assignments for the "Neuroinformatics" course. The first four assignments are implemented in Python 3.9.11 using the Keras (TensorFlow) 2.9.1 library. Subsequent assignments are implemented in PyTorch.

Machine configuration: AMD Ryzen 5 5600H 3.30 GHz 6 cores / 12 threads, 16 GB RAM. No GPU is used for model training.

## Lab 1. Rosenblatt's Perceptron

The first assignment explores the classification problem using Rosenblatt's perceptron — a simple mathematical model of a single neuron. The perceptron is an adder with the Heaviside step function. The perceptron model in this assignment is modified because TensorFlow cannot work with non-differentiable functions such as the Heaviside step function. The step function is replaced with the sigmoid activation function, assuming that any value above 0.5 corresponds to 1 and any value below 0.5 corresponds to 0. It is also worth noting that the quality of model training strongly depends on the initial conditions — the values of the weights and bias.

**Files:**
- `lab1-1.py` - Binary classification (2 classes)
- `lab1-2.py` - Multiclass classification (4 classes)

Average training time: 15 s.

## Lab 2. ADALINE Network

The first part explores the ADALINE (Adaptive LInear NEuron) network. In terms of architecture, the ADALINE network repeats the previously studied Rosenblatt's perceptron, except for the activation function — as the name implies, it is linear here. **It is also worth noting the Widrow-Hoff learning rule, which was proposed for training the ADALINE network as the precursor to gradient-based learning methods.** The network is explored in the context of a signal approximation task.

**Files:**
- `lab2-1.py` - Signal approximation / Time series prediction

Average training time: 5 s.

The second part explores the ADALINE network applied to a filtering task. The input signal is a harmonic oscillation distorted relative to the true signal in amplitude and phase shift.

**Files:**
- `lab2-2.py` - Signal filtering / Noise removal

Average training time: 6 s.

## Lab 3. Multilayer Networks

This assignment explores fully connected feedforward neural networks applied to classification and approximation tasks. The work can be divided into two parts: classification and approximation. For the classification task, a neural network model is trained to predict whether a point on a two-dimensional plane belongs to one of three geometric figures. A point is considered to belong to a figure if it lies on its boundary.

There are exactly three classes (figures) on the plane, so each point in space can be easily converted into a pixel of the RGB color space. Pure red, green, and blue represent the corresponding class. Other colors form transition states.

**Files:**
- `lab3-1.py` - Geometric shape classification (three-layer network 2 → 40 → 12 → 3)
- `lab3-2.py` - Approximation of a complex nonlinear function

Average model training time: 30 s.

## Lab 4. Radial Basis Function Networks

This assignment explores networks based on radial basis functions (RBF). Unlike previous architectures, RBF layer neurons compute the distance from the input vector to a trainable center and apply a Gaussian activation function. The advantage of RBF networks lies in their ability to efficiently approximate functions using fewer parameters. The networks are implemented using a custom RBFLayer with trainable center (mu) and width (sigma) parameters.

**Files:**
- `lab4-1.py` — Geometric shape classification using 10 RBF functions. Visualizes the positions of the found centers on the plot.
- `lab4-2.py` — Nonlinear function approximation using 15 RBF functions. Displays the centers and predicted values.

## Lab 5. Recurrent Neural Networks

This assignment explores recurrent neural networks, which have memory of previous states. The work is divided into two architectures: the Elman network and the Hopfield network. Recurrent networks are implemented in PyTorch for the first time.

**Files:**
- `lab5-1.py` — Elman network (PyTorch) for pattern recognition in a signal. A custom ElmanLayer stores the context vector of the previous state. Task: binary classification of temporal signal patterns (labels −1 and +1).
- `lab5-2.py` — Hopfield network (PyTorch) for image restoration. A custom HopfieldLayer implements associative memory. Task: restoring a damaged image from a partially noisy input. Includes an animated visualization of the convergence process.
- `lab5-3.py` — Alternative implementation of the Elman network in Keras (TensorFlow).

## Lab 6. Self-Organizing Maps (Kohonen Maps)

This assignment explores unsupervised learning using Kohonen self-organizing maps (SOM). Unlike previous methods that require target labels, SOM learns without a teacher, revealing the topological structure in data. The algorithm is based on finding the closest neuron (Best Matching Unit, BMU) and updating the weights in its neighborhood with a decreasing learning rate and radius of influence.

**Files:**
- `lab6-1.py` — Self-organizing map for the RGB color space. A neuron grid of size 64×48 self-organizes to represent four input colors (red, orange, green, blue). Includes an animated visualization of the self-organization process, as well as plots of the decreasing radius and learning rate.
- `lab6-2.py` — Self-organizing map for two-dimensional data. A 64×48 neuron grid adapts to a set of points on the plane.

## Lab 7. Autoencoders

This assignment explores the autoencoder architecture for data compression and reconstruction. An autoencoder consists of an encoder that compresses input data into a lower-dimensional latent representation, and a decoder that reconstructs the data from the compressed representation.

**Files:**
- `lab7-1.py` — Autoencoder for images from the CIFAR-10 dataset (PyTorch).
  - Encoder architecture: 3072 → 4608 → 96 (with tanh activation function).
  - Decoder architecture: 96 → 4608 → 3072 (with tanh activation function).
  - Interactive interface with three sliders to manipulate three random components of the latent representation and observe the change in the reconstructed image.
  - Button to randomly sample an image from the dataset.

## Lab 8. NARX Network

This assignment explores the Nonlinear AutoRegressive network with eXogenous inputs (NARX). The network uses Tapped Delay Lines (TDL) to store previous values of the input and output signals, allowing it to model nonlinear dynamic systems.

**Files:**
- `lab8-1.py` — NARX network (PyTorch) for identification of a nonlinear dynamic system. The architecture includes two TDL (Tapped Delay Line) modules for the input and output signals, as well as a two-layer network with a tanh activation function. Task: predicting the output signal of a nonlinear system from the input signal.

---

## Project Structure

```
neuroinformatics/
├── lab1-1.py    # Lab 1, part 1: Perceptron (binary classification)
├── lab1-2.py    # Lab 1, part 2: Perceptron (4 classes)
├── lab2-1.py    # Lab 2, part 1: ADALINE (time series approximation)
├── lab2-2.py    # Lab 2, part 2: ADALINE (noisy signal filtering)
├── lab3-1.py    # Lab 3, part 1: Multilayer network (shape classification)
├── lab3-2.py    # Lab 3, part 2: Multilayer network (function approximation)
├── lab4-1.py    # Lab 4, part 1: RBF network (classification)
├── lab4-2.py    # Lab 4, part 2: RBF network (approximation)
├── lab5-1.py    # Lab 5, part 1: Elman network (PyTorch)
├── lab5-2.py    # Lab 5, part 2: Hopfield network (PyTorch)
├── lab5-3.py    # Lab 5, part 3: Elman network (TensorFlow/Keras)
├── lab6-1.py    # Lab 6, part 1: Kohonen map for colors (NumPy)
├── lab6-2.py    # Lab 6, part 2: Kohonen map for 2D data (NumPy)
├── lab7-1.py    # Lab 7: CIFAR-10 autoencoder (PyTorch)
├── lab8-1.py    # Lab 8: NARX network (PyTorch)
├── README.md    # This file (English)
└── README.ru.md # Documentation (Russian)
```

## Environment Requirements

- Python 3.9.11
- TensorFlow / Keras 2.9.1 (Labs 1–4, Lab 5-3)
- PyTorch (Labs 5-1, 5-2, 6–8)
- NumPy
- Matplotlib
- Pillow (for image processing in Lab 5-2)
- tqdm (for displaying training progress)

## General Notes

The main metrics used to evaluate model quality across all assignments are:
- **Loss function**: MSE (Mean Squared Error).
- **Quality metric**: MAE (Mean Absolute Error).

Each assignment typically generates plots with several subplots:
- Loss curve over epochs.
- Quality metric curve over epochs (where applicable).
- Visualization of input data or model output.
- Scalar field (2D) or other relevant visualization.

Starting from Lab 5, implementations transition from TensorFlow/Keras to PyTorch, providing greater flexibility when defining custom architectures (e.g., Hopfield networks, NARX). Labs 5 and 6 include animated visualizations, and Lab 7 includes interactive controls (sliders and a button).
