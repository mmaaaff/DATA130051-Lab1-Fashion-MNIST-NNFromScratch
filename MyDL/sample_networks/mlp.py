from MyDL import nn

class MLP3(nn.NeuralNetwork):
    def __init__(self, hidden_size1=100, hidden_size2=10, activation='relu'):
        super().__init__()
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.activ_func = activation
        self.fc1 = nn.Linear(784, hidden_size1, initialize='random')
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2, initialize='random')
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(hidden_size2, 10, initialize='random')
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise ValueError('Unknown activation function')
        self.softmax = nn.Softmax()
        self.BN1 = nn.BatchNorm1d(784)
        self.BN2 = nn.BatchNorm1d(hidden_size1)
        self.BN3 = nn.BatchNorm1d(hidden_size2)
    def forward(self, x):
        x = self.BN1(x)
        x = self.fc1(x)
        x = self.BN2(x)
        x = self.activation(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.BN3(x)
        x = self.activation(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        x = self.softmax(x)
        return x