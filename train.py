import argparse
import os

import cupy as np
import numpy
import gc
from struct import unpack
import gzip
import matplotlib.pyplot as plt

import MyDL
import MyDL.sample_networks
import MyDL.data
import MyDL.optimizer as optim
import MyDL.nn as nn


parser = argparse.ArgumentParser()
parser.add_argument('model_type', choices=['mlp', 'resnet'])
parser.add_argument('optimizer', choices=['sgd', 'momentum', 'adam'])
parser.add_argument('batch_size', type=int, default=64)
parser.add_argument('num_epochs', type=int, default=20)
parser.add_argument('--learning_rate', '-lr', type=float, default=0.01)
parser.add_argument('lambda_L2', type=float, default=0.0)
parser.add_argument('--augment', type=bool, default=False)
parser.add_argument('--augment_prob', type=float, default=0.5)
parser.add_argument('--val_interval', type=int, default=30)
parser.add_argument('--layer_size', nargs='+', type=int, default=[512, 128])
parser.add_argument('--activ_func', choices=['relu', 'sigmoid', 'tanh'])
parser.add_argument('--model_path', type=str, default='MNIST_result/model_params')
parser.add_argument('--result_path', type=str, default='MNIST_result/results')
parser.add_argument('--continue_if_exists', default=False, help='Continue training if model already exists')

args = parser.parse_args()

def read(filepath, show=False):
    with gzip.open(filepath, 'rb') as f:
        if show:
            magic, num, rows, cols = unpack('>4I', f.read(16))
            print('magic\t\t', magic)
            print('num\t\t', num)
            print('rows\t\t', rows)
            print('cols\t\t', cols)
        content=np.frombuffer(f.read(), dtype=np.uint8)
    return content

# Data preperation
train_imgs = read(r'dataset\MNIST\train-images-idx3-ubyte.gz', show=False).reshape(-1, 28, 28)
train_labels = read(r'dataset\MNIST\train-labels-idx1-ubyte.gz')
train_labels = train_labels[-60000:]
test_imgs = read(r'dataset\MNIST\t10k-images-idx3-ubyte.gz', show=False).reshape(-1, 28, 28)
test_labels = read(r'dataset/MNIST/t10k-labels-idx1-ubyte.gz')
test_labels = test_labels[-10000:]
train_imgs = train_imgs.reshape(-1, 1, 28, 28)
test_imgs = test_imgs.reshape(-1, 1, 28, 28)

X_train_mytensor = MyDL.MyTensor(train_imgs, requires_grad=False)
y_train_mytensor = MyDL.MyTensor(train_labels, requires_grad=False)
X_val_mytensor = MyDL.MyTensor(test_imgs, requires_grad=False)
y_val_mytensor = MyDL.MyTensor(test_labels, requires_grad=False)

unfold = True if args.model_type == 'mlp' else False

train_data = MyDL.data.mnist_dataset(X_train_mytensor, y_train_mytensor, augment=args.augment, augment_prob=args.augment_prob, unfold=True)
val_data = MyDL.data.mnist_dataset(X_val_mytensor, y_val_mytensor, unfold=True)


# Start training
continue_if_exists = args.continue_if_exists
num_epochs = args.num_epochs
batch_size = args.batch_size
model_path = args.model_path
optimizer = args.optimizer
lr = args.learning_rate
lambda_L2 = args.lambda_L2
activ_func = args.activ_func
hidden_size1, hidden_size2 = args.layer_size
model_path = args.model_path
result_path = args.result_path

if args.model_type == 'mlp':
    model_name = 'MLP3_({},{})_{}_L2-{}_lr-{}_augment={}'.format(hidden_size1, hidden_size2, activ_func, lambda_L2, lr, train_data.augment)
    model = MyDL.sample_networks.MLP3(hidden_size1=hidden_size1, hidden_size2=hidden_size2, activation=activ_func)
elif args.model_type == 'resnet':
    model_name = 'ResNet_{}_L2-{}_lr-{}_augment={}'.format(activ_func, lambda_L2, lr, train_data.augment)
    model = MyDL.sample_networks.ResNetMNIST()
print(f'model: {model_name}')

criterion = nn.CrossEntropyLoss()
if optimizer == 'sgd':
    optimizer = optim.SGD(model.params, lr=lr, momentum=0.9, weight_decay=lambda_L2)
elif optimizer == 'momentum':
    optimizer = optim.Momentum(model.params, lr=lr, momentum=0.9, weight_decay=lambda_L2)
elif optimizer == 'adam':
    optimizer = optim.Adam(model.params, lr=lr, beta1=0.9, beta2=0.999, eps=1e-8, decay_rate=0.2, weight_decay=lambda_L2)

model_runner = MyDL.runner(model, model_name, optimizer, criterion, batch_size=batch_size)

result = model_runner.train(train_data, val_data, num_epochs, lambda_L2=lambda_L2, result_path=result_path, model_path=model_path,continue_if_exists=continue_if_exists, val_interval=args.val_interval)



def plot_figures(model_name):
    result = numpy.load(os.path.join('MNIST_result/results', f'{model_name}.npz'))
    x1 = numpy.arange(0, len(result['train_loss_iter']))
    train_loss = result['train_loss_iter']
    train_acc = result['train_acc_iter']

    print(result['val_loss_iter'])


    epoch_len = 50000
    if result['val_interval'].item() == 0:  # If no complete val_loss_iter, use val_loss_epoch instead, but need alignment
        # result['batch_size_till_iter'] is in shape (num_train, 2)
        end_point = result['batch_size_till_iter'].T[1]
        batch_size = result['batch_size_till_iter'].T[0]
        x2 = numpy.array([])
        for i in range(len(end_point)):
            iter_per_epoch = epoch_len // batch_size[i] + 1
            start_point = iter_per_epoch if i == 0 else end_point[i - 1] + iter_per_epoch
            x2 = numpy.concatenate((x2, numpy.arange(start_point, end_point[i], iter_per_epoch)))
        val_loss = result['val_loss_epoch']
        val_acc = result['val_acc_epoch']
    else:
        end_point = result['batch_size_till_iter'].T[1]
        batch_size = result['batch_size_till_iter'].T[0]
        interval = result['val_interval'].item()
        x2 = numpy.array([])
        for i in range(len(end_point)): 
            iter_per_epoch = epoch_len // batch_size[i] + 1
            for j in range(len(result['train_loss_epoch'])):  # How many epochs in this train      
                start = j * iter_per_epoch + interval if i == 0 else end_point[i - 1] + j * iter_per_epoch + interval
                end = end_point[i] if j == len(result['train_loss_epoch']) - 1 else (j + 1) * iter_per_epoch
                x2 = numpy.concatenate((x2, numpy.arange(start, end, interval)))
        val_loss = result['val_loss_iter']
        val_acc = result['val_acc_iter']

    plt.style.use('seaborn-v0_8-darkgrid')

    # --- Loss Curve ---
    plt.figure(figsize=(10, 6))
    plt.plot(x1, train_loss, label='Train Loss', linewidth=2)
    plt.plot(x2, val_loss, label='Test Loss', linewidth=2)
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training and Test Loss Curve', fontsize=14)
    plt.legend(loc='upper right', fontsize=10)
    f = os.path.join(result_path, f'{model_name}_loss.png')
    plt.savefig(f, bbox_inches='tight')
    plt.show()
    print(f'Figure saved as {f}')

    # --- Accuracy Curve ---
    plt.figure(figsize=(10, 6))
    plt.plot(x1, train_acc, label='Train Accuracy', linewidth=2)
    plt.plot(x2, val_acc, label='Test Accuracy', linewidth=2)
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Training and Test Accuracy Curve', fontsize=14)
    plt.legend(loc='lower right', fontsize=10)
    f = os.path.join(result_path, f'{model_name}_accuracy.png')
    plt.savefig(f, bbox_inches='tight')
    plt.show()
    print(f'Figure saved as {f}')

plot_figures(model_name)