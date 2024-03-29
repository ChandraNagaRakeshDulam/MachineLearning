import torch.nn as nn
import torch.nn.functional as Function
import joblib

from torchvision import models

data_lb = joblib.load('./predictions/data_lb.pkl')

class TranslatorCNN(nn.Module):
    def __init__(self):
        super(TranslatorCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 5)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.conv3 = nn.Conv2d(32, 64, 3)
        self.conv4 = nn.Conv2d(64, 128, 5)

        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, len(data_lb.classes_))

        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.pool(Function.relu(self.conv1(x)))
        x = self.pool(Function.relu(self.conv2(x)))
        x = self.pool(Function.relu(self.conv3(x)))
        x = self.pool(Function.relu(self.conv4(x)))
        bs, _, _, _ = x.shape
        x = Function.adaptive_avg_pool2d(x, 1).reshape(bs, -1)
        x = Function.relu(self.fc1(x))
        x = self.fc2(x)
        return x