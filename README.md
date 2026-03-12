# Fashion MNIST Classification Project (Author: MD TOHIDUL ISLAM)

## Project Overview
This project implements a neural network classifier for the Fashion MNIST dataset using PyTorch. The model is trained on a small subset of Fashion MNIST (6000 samples) to classify grayscale images of clothing items into 10 categories.

## Dataset
- **Source**: `fmnist_small.csv` (subset of Fashion MNIST)
- **Image Size**: 28x28 pixels (784 features)
- **Training Samples**: 4800
- **Test Samples**: 1200
- **Number of Classes**: 10 (T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot)
- **Data Split**: 80% training, 20% testing
- **Preprocessing**: Pixel values normalized by dividing by 255.0

## Model Architecture

### Architecture Details
- **Input Layer**: 784 neurons (flattened 28×28 images)
- **Hidden Layers**: 
  - Layer 1: 256 neurons with ReLU activation
  - Layer 2: 128 neurons with ReLU activation
  - Layer 3: 64 neurons with ReLU activation
- **Output Layer**: 10 neurons (one per class)

## Training Configuration

### Hyperparameters
- **Epochs**: 100
- **Batch Size**: 32
- **Learning Rate**: 0.1
- **Optimizer**: SGD (Stochastic Gradient Descent)
- **Loss Function**: CrossEntropyLoss
- **Device**: GPU (CUDA) when available, else CPU

### Training Features
- Custom PyTorch Dataset class implementation
- DataLoader with pin_memory for faster GPU transfer
- Regular loss monitoring per epoch
- Model evaluation mode for testing

## Results
- **Test Accuracy**: 84.00%
- **Correct Predictions**: 1008 out of 1200 test samples

## Key Dependencies
- PyTorch
- pandas
- numpy
- matplotlib
- scikit-learn (train_test_split)

## Usage
1. Ensure `fmnist_small.csv` is in the working directory
2. Run the notebook cells sequentially
3. Model will train for 100 epochs and output final accuracy

## Sample Visualizations
The notebook includes visualization of the first 16 images from the dataset with their corresponding labels.
