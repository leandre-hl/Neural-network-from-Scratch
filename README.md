[🇫🇷 Français](README.fr.md) | [🇬🇧 English](README.md)

# Neural Network from Scratch in C

A small feed-forward neural network written in plain C that learns to recognize handwritten digits from the **MNIST** dataset. Everything is built from scratch: the matrix library, the forward and backward passes, the training loop and the saving/loading of trained networks. The only dependency is the C standard library (plus `libm`).

## Features

- Fully connected network with one hidden layer: **784 → 500 → 10** (configurable)
- ReLU activation on the hidden layer, numerically stable **softmax** on the output layer
- Training by stochastic gradient descent (one image per update) with backpropagation
- Save and load a trained network to and from a directory of plain-text files
- Custom `MatrixLib` for all the linear algebra
- Loading of MNIST images directly from CSV files

## Project structure

```
neural-network/
├── Makefile
├── main.c                 Entry point: what you edit to train or test a network
├── parameters.h           Network size, learning rate, dataset and save paths
├── MatrixLib/
│   ├── MatrixLib.h        Matrix type and operations (declarations)
│   └── MatrixLib.c        Matrix implementation
├── network/
│   ├── neural_network.h/c Network creation, training, prediction, save/load
│   └── activations.h/c    ReLU, softmax, sigmoid
├── util/
│   └── img.h/c            MNIST CSV loader and image printing
├── data/
│   ├── mnist_train.csv    Training set (60,000 images)
│   └── mnist_test.csv     Test set (10,000 images)
├── network_save/          A saved network (created by the program)
└── exe/main.exe           Compiled program (created by `make`)
```

The project uses a **unity build**: `main.c` includes the other `.c` files, so only `main.c` is compiled. You should not compile the other `.c` files separately.

## Requirements

- A C compiler (`gcc` is used by default)
- `make`
- The MNIST dataset in CSV format, placed in `data/` (see below)

## Dataset

The program expects two CSV files:

```
data/mnist_train.csv
data/mnist_test.csv
```

Each file has a header line, followed by one image per line: the digit label (0–9) first, then the 784 pixel values (0–255) of the 28×28 image. Pixel values are scaled to `[0, 1]` when loaded. If you don't have these files, the MNIST dataset in CSV format is widely available online (for example on Kaggle).

## Build and run

```bash
make          # compiles the program to ./exe/main.exe
make run      # builds if needed, then runs it
make clean    # deletes the executable
```

Run all commands from the project root, because the dataset and save paths in `parameters.h` are relative to the working directory.

## How to use it

What the program does is decided by `main.c`. It contains two blocks: one active, one commented out. Switch between them by moving the `/* ... */` comment markers.

### 1. Create and test a new network (default)

The active block creates a network with random weights, measures its accuracy on the test images, and saves it to `network_save/`:

```c
NeuralNetwork* net = network_create(INPUT, HIDDEN, OUTPUT, LEARNING_RATE);
Img** imgs_test = csv_to_imgs(TEST_DATASET_PATH, 0, N_TEST_IMGS);
printf("Accuracy : %1.1f %c\n", 100*network_predict_imgs(net, imgs_test, N_TEST_IMGS), '%');
network_save(net, LOAD_DIR_PATH);
```

An untrained network should score around chance level (about 10 %). Run this first to produce the starting network that the training step loads.

### 2. Train a network

To train, comment out the first block and uncomment the second one:

```c
Img** imgs_train = csv_to_imgs(TRAIN_DATASET_PATH, 0, N_TRAIN_IMGS);
Img** imgs_test = csv_to_imgs(TEST_DATASET_PATH, 0, N_TEST_IMGS);
NeuralNetwork* net = network_load(LOAD_DIR_PATH);
network_train_batch_imgs(net, imgs_train, N_TRAIN_IMGS);
printf("Accuracy : %1.1f %c\n", 100*network_predict_imgs(net, imgs_test, N_TEST_IMGS), '%');
network_save(net, SAVE_DIR_PATH);
```

This loads the network from `network_save/`, trains it once over the first `N_TRAIN_IMGS` training images, prints the accuracy on the test images, and saves the result to `network_save2/`. With the default settings (2,000 training images, 1,000 test images), a single pass reaches roughly 87 % accuracy.

To keep improving a network, set `LOAD_DIR_PATH` to the directory you saved last (or train for several passes by calling `network_train_batch_imgs` repeatedly), and use a different `SAVE_DIR_PATH`, otherwise you overwrite the network you loaded.

## Configuration

All settings live in `parameters.h`:

| Constant | Default | Meaning |
|----------|---------|---------|
| `INPUT` | 784 | Input layer size (28×28 pixels) |
| `HIDDEN` | 500 | Hidden layer size |
| `OUTPUT` | 10 | Output layer size (digits 0–9) |
| `LEARNING_RATE` | 0.02 | Gradient descent step size |
| `N_TRAIN_IMGS` | 2000 | Number of training images used |
| `N_TEST_IMGS` | 1000 | Number of test images used |
| `TRAIN_DATASET_PATH` / `TEST_DATASET_PATH` | `./data/mnist_*.csv` | Dataset locations |
| `LOAD_DIR_PATH` | `network_save` | Directory a network is loaded from (and the default save target in test mode) |
| `SAVE_DIR_PATH` | `network_save2` | Directory a trained network is saved to |

The full datasets contain 60,000 training and 10,000 test images, so you can raise `N_TRAIN_IMGS` and `N_TEST_IMGS` up to those values. Training time grows linearly with the number of images.

## Saved network format

A network is saved as a directory of plain-text files:

| File | Content |
|------|---------|
| `descriptor` | Input size, hidden size, output size and learning rate, one per line |
| `hidden_weights`, `hidden_biases` | Hidden layer parameters |
| `output_weights`, `output_biases` | Output layer parameters |

Each matrix file starts with a line `rows cols`, followed by one line per matrix row. When a network is loaded, the matrix dimensions are checked against the descriptor.

## How it works

For one training image (a 784×1 column vector `A0`) and its one-hot label `Y`:

1. **Forward pass:** `Z1 = W1·A0 + b1`, `A1 = ReLU(Z1)`, `Z2 = W2·A1 + b2`, `A2 = softmax(Z2)`
2. **Backward pass:** `dZ2 = A2 − Y`, `dW2 = dZ2·A1ᵀ`, `dZ1 = (W2ᵀ·dZ2) ⊙ ReLU′(Z1)`, `dW1 = dZ1·A0ᵀ`, and the corresponding bias gradients
3. **Update:** each parameter is reduced by `learning_rate` times its gradient

Weights and biases are initialized with uniform random values in `[-1/√n, 1/√n]`. The random generator is not seeded, so a freshly created network is identical from one run to the next.

### MatrixLib

`MatrixLib` provides the `Matrix` type (`entries[i][j]`, `rows`, `cols`) and the operations the network needs: `add`, `subtract`, `multiply` (element-wise), `dot` (matrix product), `scale`, `transpose`, `apply`, `sum`, `matrix_argmax`, as well as printing, saving and loading. Operations return newly allocated matrices that must be released with `matrix_free`, and they stop the program with an error message if the dimensions do not match.

## Notes

- The compiled `exe/main.exe` is rebuilt by `make` whenever a source or header file changes.
- The `data/` folder is large (about 128 MB). Consider excluding it from version control.
- Training uses one image per update, which is simple but not the fastest approach. Mini-batches would be a natural next improvement.