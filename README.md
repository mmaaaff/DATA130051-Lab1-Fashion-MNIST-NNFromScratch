<h1 align="center">DATA130011 Project1</h1>

<div align="center">周语诠</div>
<div align="center">2025-5-1</div>

## Contents

- [Contents](#contents)
- [Description \& Features](#description--features)
- [Requirements](#requirements)
- [Classification on Fashion-MNIST](#classification-on-fashion-mnist)
- [Utilizing MyDL](#utilizing-mydl)

***

## Description & Features

This project aims at exploring and fully conprehending some basic deeplearning thoughts and methods. It is written based on numpy, without relying on deep learning frameworks that supports auto-gradient, e.g., Pytorch, Tensorflow.

Common functions are contained in directory MyDL which is imported as a package in the three .ipynb files in the root directory. The package realizes: common tensor calculation, construction of computational graph with BP auto-gradient, necessary layers in image classification(MLP and Conv2D), CE loss function, etc.

The usage of the package is almost same to Pytorch.You can easily migrate network struture from Pytorch to MyDL, only needing to change several layer names. Executing **backward()** on loss assigns gradient to tensors, which are instances of class **MyDL.MyTensor**.

***

## Requirements

This project requires cupy and matplotlib. Execute following command to install cupy:

```cmd
pip install cupy-cuda12x
```

The version should be based on you CUDA version. For CUDAToolkit installation, please refer to the official website <a href="https://developer.nvidia.com/cuda-toolkit">https://developer.nvidia.com/cuda-toolkit</a>

***

## Classification on Fashion-MNIST

1. **Training**

    > - The training code is train.py.
    > - To train a network, use shell scripts in train_mlp.sh and train_resnet.sh in scripts/.

    ```bash
    cd DATA130051-Lab1-DATA130011-PJ1
    bash scripts/train_resnet.sh
    ```

    ***

2. **Testing**

    > - The testing code is test.py
    > - To test a network on test set, use shell scripts in test_mlp.sh and test_resnet.sh in scripts/.
    
    ```bash
    cd DATA130051-Lab1-DATA130011-PJ1
    bash scripts/test_resnet.sh
    ```

    ***

3. Sample Networks

    MyDL contains two sample networks: MLP and ResNet. Find them in MyDL/sample_networks. These two networks are used in training classifiers on MNIST.

## Utilizing MyDL

If you want to explore the package further and build something else, here is a brief instruction. In general it is similar to Pytorch.

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
    model_runner = MyDL.runner(model, model_name, optimizer, 
                               criterion, batch_size=batch_size, 
                               scheduler=scheduler)
    result = model_runner.train(train_data, val_data, 
                                num_epochs, lambda_L2=lambda_L2, 
                                result_path=result_path, model_path=model_path, 
                                continue_if_exists=continue_if_exists, 
                                val_interval=args.val_interval)
    ```

    This will save the model's parameters in directory 'model_params'.

- Test a model:

    ```Python
    model_runner.eval(model, test_data)
    ```

    This returns the accuracy of the model on test_data.
