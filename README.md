
# Project Description

This project aims at exploring and fully conprehensing some basic deeplearning > thoughts and methods. It is written based on numpy, without relying on deep learning framworks that supports auto-gradient, i.e. Pytorch, Tensorflow.

Common functions are contained in directory MyDL which is imported as a package in the three .ipynb files in the root directory. The package realizes: common tensor calculation(add, substract, element-wise multiplication, matrix multiplication, square, exponential, logarith, etc.), construction of computational graph with BP auto-gradient, necessary layers in the MLP task, CE loss function, etc.

***

# Requirements

This project requires numpy and matplotlib. Execute following command to install these packages:

```cmd
pip install numpy matplotlib
```

# Hyperparameter searching

> - In the root dir find **search_hyperparams.ipynb**.
> - Running this file splits the training data into training set and validation set in 5:1 ratio, and trains the model with different hyperparameters.
> - Model parameters and results will be automatically saved in **model_params** and **results**. Best model is selected based on validation accuracy.

***

# Training

> - Running **train.ipynb** trains the best model selected in above on the whole training data.
> - model parameters and results will be saved in **final_model_params** ans **final_results**.

***

# Testing

> - Run **test.ipynb** to test the traind model on test data. This returns the model accuracy on test data.

***
