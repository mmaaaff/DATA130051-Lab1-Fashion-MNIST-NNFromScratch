import argparse
import os
import pickle

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
parser.add_argument('--full-train', type=str, choices=['True', 'False'], default="False")
parser.add_argument('--scheduler', choices=['MultiStepLR', 'None'], default='None')
parser.add_argument('--milestones', nargs='+', type=int, default=[0, 0])
parser.add_argument('--scheduler-gamma', type=float, default=0.5)
parser.add_argument('--learning-rate', '-lr', type=float, default=0.01)
parser.add_argument('--lambda-L2', type=float, default=0.0)
parser.add_argument('--augment', type=str, default=False)
parser.add_argument('--augment-prob', type=float, default=0.5)
parser.add_argument('--val-interval', type=int, default=30)
parser.add_argument('--layer-size', nargs='+', type=int, default=[512, 128])
parser.add_argument('--mlp-dropout', type=float, default=0.0)
parser.add_argument('--activ-func', choices=['relu', 'sigmoid', 'tanh'], default='relu')
parser.add_argument('--model-path', type=str, default='MNIST_result/model_params')
parser.add_argument('--result-path', type=str, default='None')
parser.add_argument('--continue-if-exists', default=False, help='Continue training if model already exists')

args = parser.parse_args()

train_images_path = r'dataset/MNIST/train-images-idx3-ubyte.gz'
train_labels_path = r'dataset/MNIST/train-labels-idx1-ubyte.gz'
test_images_path = r'dataset/MNIST/t10k-images-idx3-ubyte.gz'
test_labels_path = r'dataset/MNIST/t10k-labels-idx1-ubyte.gz'

with gzip.open(train_images_path, 'rb') as f:
        magic, num_train, rows, cols = unpack('>4I', f.read(16))
        train_imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_train, 28*28)
    
with gzip.open(train_labels_path, 'rb') as f:
        magic, num_train = unpack('>2I', f.read(8))
        train_labels = np.frombuffer(f.read(), dtype=np.uint8)

with gzip.open(test_images_path, 'rb') as f:
        magic, num_test, rows, cols = unpack('>4I', f.read(16))
        test_imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_test, 28*28)
    
with gzip.open(test_labels_path, 'rb') as f:
        magic, num_test = unpack('>2I', f.read(8))
        test_labels = np.frombuffer(f.read(), dtype=np.uint8)

# Data preperation
idx = np.random.permutation(np.arange(num_train))

with open('idx.pickle', 'wb') as f:
        pickle.dump(idx, f)
train_imgs = train_imgs[idx]
train_labels = train_labels[idx]


if args.full_train == "False":
    valid_imgs = train_imgs[:10000]
    valid_labels = train_labels[:10000]
    train_imgs = train_imgs[10000:]
    train_labels = train_labels[10000:]
    X_train_mytensor = MyDL.MyTensor(train_imgs.reshape(-1, 1, 28, 28), requires_grad=False)
    y_train_mytensor = MyDL.MyTensor(train_labels, requires_grad=False)
    X_val_mytensor = MyDL.MyTensor(valid_imgs.reshape(-1, 1, 28, 28), requires_grad=False)
    y_val_mytensor = MyDL.MyTensor(valid_labels, requires_grad=False)
else:
    X_train_mytensor = MyDL.MyTensor(train_imgs.reshape(-1, 1, 28, 28), requires_grad=False)
    y_train_mytensor = MyDL.MyTensor(train_labels, requires_grad=False)
    X_val_mytensor = MyDL.MyTensor(test_imgs.reshape(-1, 1, 28, 28), requires_grad=False)
    y_val_mytensor = MyDL.MyTensor(test_labels, requires_grad=False)


unfold = True if args.model_type == 'mlp' else False

augment = True if args.augment == "True" else False
train_data = MyDL.data.mnist_dataset(X_train_mytensor, y_train_mytensor, augment=args.augment, augment_prob=args.augment_prob, unfold=unfold)
val_data = MyDL.data.mnist_dataset(X_val_mytensor, y_val_mytensor, unfold=unfold)


# Start training
continue_if_exists = args.continue_if_exists
num_epochs = args.num_epochs
batch_size = args.batch_size
model_path = args.model_path
optimizer = args.optimizer
scheduler = args.scheduler
lr = args.learning_rate
lambda_L2 = args.lambda_L2
activ_func = args.activ_func
hidden_size1, hidden_size2 = args.layer_size
model_path = args.model_path
result_path = args.result_path


if args.model_type == 'mlp':
    if hidden_size2 > 0:
        model_name = f'MLP3_({hidden_size1},{hidden_size2})_dropout{args.mlp_dropout}_{activ_func}_L2-{lambda_L2}_lr-{lr}_augment={train_data.augment}_optim={args.optimizer}_schduler={args.scheduler}_{args.milestones}_{args.scheduler_gamma}'
        model = MyDL.sample_networks.MLP3(hidden_size1=hidden_size1, hidden_size2=hidden_size2, activation=activ_func, dropout=args.mlp_dropout)
    else:
        model_name = f'MLP3_({hidden_size1})_dropout{args.mlp_dropout}_{activ_func}_L2-{lambda_L2}_lr-{lr}_augment={train_data.augment}_optim={args.optimizer}schduler={args.scheduler}_{args.milestones}_{args.scheduler_gamma}'
        model = MyDL.sample_networks.MLP2(hidden_size=hidden_size1, activation=activ_func)
elif args.model_type == 'resnet':
    model_name = f'ResNet_relu_L2-{lambda_L2}_lr-{lr}_augment={train_data.augment}_schduler={args.scheduler}_{args.milestones}_{args.scheduler_gamma}'
    model = MyDL.sample_networks.ResNetMNIST()

if args.full_train == 'True':  # Train the best model on full training data
    model_name = args.model_type + "_best"

print(f'model: {model_name}')

criterion = nn.CrossEntropyLoss()
if optimizer == 'sgd':
    optimizer = optim.SGD(model.params, lr=lr)
elif optimizer == 'momentum':
    optimizer = optim.Momentum(model.params, lr=lr, momentum=0.9)
elif optimizer == 'adam':
    optimizer = optim.Adam(model.params, lr=lr)

if scheduler == 'MultiStepLR':
    scheduler = optim.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.scheduler_gamma)
else:
    scheduler = None 

model_runner = MyDL.runner(model, model_name, optimizer, criterion, batch_size=batch_size, scheduler=scheduler)

result = model_runner.train(train_data, val_data, num_epochs, lambda_L2=lambda_L2, result_path=result_path, model_path=model_path,continue_if_exists=continue_if_exists, val_interval=args.val_interval)



def plot_figures(model_name):
    result = numpy.load(os.path.join('MNIST_result/results', f'{model_name}.npz'))
    x1 = numpy.arange(0, len(result['train_loss_iter']))
    train_loss = result['train_loss_iter']
    train_acc = result['train_acc_iter']

    epoch_len = 50000
    if result['val_interval'].item() == 0:  # If no complete val_loss_iter, use val_loss_epoch instead, but need alignment
        # result['batch_size_till_iter'] is in shape (num_train, 2)
        end_point = result['batch_size_till_iter'].T[1]
        batch_size = result['batch_size_till_iter'].T[0]
        x2 = numpy.array([])
        for i in range(len(end_point)):
            iter_per_epoch = epoch_len // batch_size[i] + 1
            start_point = iter_per_epoch if i == 0 else end_point[i - 1] + iter_per_epoch
            x2 = numpy.concatenate((x2, numpy.arange(start_point, end_point[i] + 1, iter_per_epoch)))
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
                x2 = numpy.concatenate((x2, numpy.arange(start, end + 1, interval)))
        val_loss = result['val_loss_iter']
        val_acc = result['val_acc_iter']

    plt.style.use('seaborn-v0_8-darkgrid')

    # --- Loss Curve ---
    test_or_val = "Test" if args.full_train == "True" else "Validation"
    plt.figure(figsize=(10, 6))
    plt.plot(x1, train_loss, label='Train Loss', linewidth=2)
    plt.plot(x2, val_loss, label=f'{test_or_val} Loss', linewidth=2)
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title(f'Training and {test_or_val} Loss Curve', fontsize=14)
    plt.legend(loc='upper right', fontsize=10)
    f = os.path.join(result_path, f'{model_name}_loss.pdf')
    plt.savefig(f, bbox_inches='tight')
    plt.show()
    print(f'Figure saved as {f}')

    # --- Accuracy Curve ---
    plt.figure(figsize=(10, 6))
    plt.plot(x1, train_acc, label='Train Accuracy', linewidth=2)
    plt.plot(x2, val_acc, label=f'{test_or_val} Accuracy', linewidth=2)
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title(f'Training and {test_or_val} Accuracy Curve', fontsize=14)
    plt.legend(loc='lower right', fontsize=10)
    f = os.path.join(result_path, f'{model_name}_accuracy.pdf')
    plt.savefig(f, bbox_inches='tight')
    plt.show()
    print(f'Figure saved as {f}')

plot_figures(model_name)