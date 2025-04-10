# MNIST Classifier

This repository contains a Python-based implementation of a classifier for the MNIST dataset. The code supports multiple machine learning models, including a Random Forest, a Multi-Layer Perceptron (MLP), and a Convolutional Neural Network (CNN). The project is designed to train and evaluate these models on the MNIST dataset, which consists of handwritten digit images.

## Features

- **Random Forest Classifier**: A decision tree-based model for classification.
- **Neural Networks**: Includes both a fully connected MLP and a CNN for image classification.
- **Custom Data Loader**: Reads and preprocesses the MNIST dataset from raw files.
- **Evaluation Metrics**: Computes the F1-macro score for model evaluation.

## File Structure

- **`MnistClassifier.py`**: Main script that initializes, trains, and evaluates the models.
- **`utils.py`**: Contains utility classes, including the MNIST data loader and dataset interface.
- **`models.py`**: Implements the Random Forest, MLP, and CNN models.

## Requirements

- Python 3.8+
- Required libraries:
    - `numpy`
    - `torch`
    - `scikit-learn`

## Usage

1. **Prepare the MNIST Dataset**: Place the MNIST dataset files in the `mnist` directory:
     - `train-images.idx3-ubyte`
     - `train-labels.idx1-ubyte`
     - `t10k-images.idx3-ubyte`
     - `t10k-labels.idx1-ubyte`

2. **Run the Classifier**:
     ```bash
     python MnistClassifier.py
     ```

3. **Evaluate Models**: The script will train and evaluate the Random Forest, MLP, and CNN models, printing their F1-macro scores.

## Example Output

- Random Forest: F1-macro score ~ 0.87
- MLP: F1-macro score ~ 0.96
- CNN: F1-macro score ~ 0.98

## Customization

- Modify the `MnistClassifier` class in `MnistClassifier.py` to add or change models.
- Adjust hyperparameters for the neural networks in `models.py`.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments

- MNIST dataset: [http://yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/)
- PyTorch: [https://pytorch.org/](https://pytorch.org/)
- Scikit-learn: [https://scikit-learn.org/](https://scikit-learn.org/)  