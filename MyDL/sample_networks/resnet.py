from MyDL import nn

class ResiduleBlock(nn.NeuralNetwork):
    """
    Define the residual block
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2D(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2D(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # When channel number not match or stride != 1, use 1x1 conv before jump connect
        self.cross_block = False
        if stride != 1 or in_channels != out_channels:
            self.cross_block = True
            self.conv_shortcut = nn.Conv2D(in_channels, out_channels, kernel_size=1,
                                           padding=0, stride=stride, bias=False)
            self.bn_shortcut = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        out = nn.ReLU.forward(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.cross_block:
            x = self.conv_shortcut(x)
            x = self.bn_shortcut(x)
        out = out + x  # residual connection
        out = nn.ReLU.forward(out)
        return out
    
class ResNetMNIST(nn.NeuralNetwork):
    """
    input: (batch, 1, 28, 28)
    ↓
    Conv 3x3, 16 channels + BN + ReLU
    ↓
    ResiduleBlockx2 (channel = 16, stride=1)
    ↓
    ResiduleBlockx2 (channel = 32, stride=2)
    ↓
    ResiduleBlockx2 (channel = 64, stride=2)
    ↓
    Average Polling (len 64 vector)
    ↓
    FC layer (64 -> 10)
    ↓
    out: (10)
    """
    def __init__(self, block=ResiduleBlock, num_classes=10):
        super(ResNetMNIST, self).__init__()
        self.in_channels = 16

        self.conv = nn.Conv2D(1, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, 2, stride=1)
        self.layer2 = self._make_layer(block, 32, 2, stride=2)
        self.layer3 = self._make_layer(block, 64, 2, stride=2)
        self.avg_pool = nn.FullAveragePool2d()
        # self.dropout1 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64, 10)
        # self.BN = nn.BatchNorm1d(256)
        # self.dropout2 = nn.Dropout(0.5)
        # self.fc2 = nn.Linear(256, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride):
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_channels, out_channels, s))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = nn.ReLU.forward(self.bn(self.conv(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)  # (batch, c)
        # out = self.dropout1(out)  # (batch, c)
        # out = self.dropout2(nn.ReLU.forward(self.BN(self.fc1(out))))
        out = self.fc1(out)
        out = nn.Softmax.forward(out)
        return out