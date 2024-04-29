<h1 align="center">DATA130051 Project1</h1>

<div align="center">周语诠</div>
<div align="center">2024-4-30</div>

## Contents

- [Contents](#contents)
- [Description \& Features](#description--features)
- [Requirements](#requirements)
- [Classification on Fshion-MNIST](#classification-on-fshion-mnist)
- [Utilizing MyDL](#utilizing-mydl)

***

## Description & Features

This project aims at exploring and fully conprehensing some basic deeplearning thoughts and methods. It is written based on numpy, without relying on deep learning frameworks that supports auto-gradient, i.e. Pytorch, Tensorflow.

Common functions are contained in directory MyDL which is imported as a package in the three .ipynb files in the root directory. The package realizes: common tensor calculation, construction of computational graph with BP auto-gradient, necessary layers in the MLP task, CE loss function, etc.

The usage of the package is very similar to Pytorch. By combining different layers you can build different kinds of networks. And implementing **backward()** on loss assigns gradient to tensors, which are instances of class **MyDL.MyTensor**. Note that FC layer is all we have now, but it should be easy to add other kinds of nets to the package.

***

## Requirements

This project requires numpy and matplotlib. Execute following command to install these packages:

```cmd
pip install numpy matplotlib
```

***

## Classification on Fshion-MNIST

1. **Hyperparameter searching**

    > - In the root dir find **search_hyperparams.ipynb**.
    > - Running this file splits the training data into training set and validation set in 5:1 ratio, and trains the model with different hyperparameters.
    > - Model parameters and results will be automatically saved in **model_params** and **results**. Best model is selected based on validation accuracy.

    ***

2. **Training**

    > - Running **train.ipynb** trains the best model selected in above on the whole training data.
    > - model parameters and results will be saved in **final_model_params** ans **final_results**.

    ***

3. **Testing**

    > - Run **test.ipynb** to test the traind model on test data. This returns the model accuracy on test data.

***

## Utilizing MyDL

If you want to explore the package furthur and build something else, here is a brief instruction. In general it is similar to Pytorch.

- Creating **MyTensor** objects:
  
    ```Python
    import MyDL
    x = MyDL.MyTensor(data[, requires_grad=True])
    ```

    **data** should be a Numpy array.
    Common operations are supported(add, substract, element-wise multiplication, matrix multiplication, square, exponential, logarith, etc.).
- Computation graph is constructed during tensor computation. Apply **.backward()** on a scalar assigns gradient to all leaf tensors that requires gradient. Use **.grad** to visit gradient.

    ```Python
    y = x.sum().item()
    y.backward()
    x.grad
    ```

- Define a network like this:
  
  ```Python
  import MyDL.nn as nn
  class simple_net(NeuralNetwork):
    def __init__(self, input_size, output_size):
        super.__init__(self)
        self.fc = nn.Linear(input_size, output_size, initialize='random')  # FC layer
        self.params += self.fc.params  # You have to add params to the network manually
        self.softmax = nn.Softmax()  # Softmax layer
    def forward(self, x):
        x = self.fc(x)  # call the layer
        x = self.softmax(x)
        return x
  ```

- Loading and saving model parameters:

    ```Python
    model = simple_net(784, 10)
    model.save(f'{filename}.npz', path)
    model.load(path)
    ```

- Switching model status:

    ```Python
    model.train()
    model.eval()
    ```

- Select optimizer:

    ```Python
    import MyDL.optimizer as optim
    optimizer = optim.Adam(model.params, lr=0.001, decay_rate=0.3)
    ```

- Select loss function:

    ```Python
    criterion = nn.CrossEntropyLoss()
    ```

- Create datasets:

    ```Python
    import MyDL.data as data
    train_data = data.dataset(X_train_tensor, y_train_tensor)
    test_data = data.dataset(X_test_tensor, y_test_tensor)
    ```

- Train a classification model:

    ```Python
    train_loss, val_loss, train_acc, val_acc, continued_train \
        = MyDL.train(model, criterion, optimizer, train_data, 
                     test_data, num_epochs=num_epochs, batch_size=256, 
                     lambda_L2=lambda_L2, path='model_params', 
                     continue_if_exists=continue_if_exists)
    ```

    This will save the model's parameters in directory 'model_params'.

- Test a model:

    ```Python
    MyDL.test(model, test_data)
    ```

    This returns the accuracy of the model on test_data.
