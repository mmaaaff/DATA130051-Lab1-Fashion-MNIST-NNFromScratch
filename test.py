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
parser.add_argument('model_path', type=str)
parser.add_argument('--layer-size', nargs='+', type=int, default=[512, 128])
parser.add_argument('--batch-size', type=int, default=256)
parser.add_argument('--activ-func', choices=['relu', 'sigmoid', 'tanh'], default='relu')

args = parser.parse_args()

test_images_path = r'dataset/MNIST/t10k-images-idx3-ubyte.gz'
test_labels_path = r'dataset/MNIST/t10k-labels-idx1-ubyte.gz'

with gzip.open(test_images_path, 'rb') as f:
        magic, num, rows, cols = unpack('>4I', f.read(16))
        test_imgs=np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28*28)
    
with gzip.open(test_labels_path, 'rb') as f:
        magic, num = unpack('>2I', f.read(8))
        test_labels = np.frombuffer(f.read(), dtype=np.uint8)

# Data preperation
idx = np.random.permutation(np.arange(num))

with open('idx.pickle', 'wb') as f:
        pickle.dump(idx, f)

X_test_mytensor = MyDL.MyTensor(test_imgs.reshape(-1, 1, 28, 28), requires_grad=False)
y_test_mytensor = MyDL.MyTensor(test_labels, requires_grad=False)

unfold = True if args.model_type == 'mlp' else False

test_data = MyDL.data.mnist_dataset(X_test_mytensor, y_test_mytensor, augment=False, unfold=unfold)


# Start Testing
batch_size = args.batch_size
model_path = args.model_path
hidden_size1, hidden_size2 = args.layer_size
activ_func = args.activ_func

if args.model_type == 'mlp':
    if hidden_size2 > 0:
        model = MyDL.sample_networks.MLP3(hidden_size1=hidden_size1, hidden_size2=hidden_size2, activation=activ_func)
    else:
        model = MyDL.sample_networks.MLP2(hidden_size=hidden_size1, activation=activ_func)
elif args.model_type == 'resnet':
    model = MyDL.sample_networks.ResNetMNIST()
criterion = nn.CrossEntropyLoss()

print('Loading model...')
model.load(args.model_path)
print(f'Model loaded successfully from {args.model_path}. Evaluation begins...')

model_runner = MyDL.runner(model, "test", None, criterion, batch_size=batch_size)

loss, acc = model_runner.eval(test_data, batch_size)

print(f'Evaluation complete. Result:')
print(f'Test Loss: {loss} \t Test Accuracy: {acc}')



