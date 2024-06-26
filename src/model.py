
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F

class CustomLargeCNN(nn.Module):
    """
    Custom CNN for large 640x640 images and 8 output classes.
    """
    def __init__(self, in_channels=3, out_classes=8):
        super(CustomLargeCNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(in_features=512 * 10 * 10, out_features=1024)  # Adjust the in_features according to the final feature map size
        self.fc2 = nn.Linear(in_features=1024, out_features=out_classes)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)  # Reduce dimensionality
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool(x)  # Further reduce dimensionality
        x = x.view(-1, 512 * 10 * 10)  # Flatten the output for the fully connected layer
        x = self.dropout(x)  # Apply dropout to reduce overfitting
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class CNN_MNIST(nn.Module):
    """
    CNN for MNIST dataset
    https://arxiv.org/pdf/2008.06180.pdf
    """

    def __init__(self, in_channels=1, out_classes=10):
        super(CNN_MNIST, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=32, kernel_size=(5, 5)
        )
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5, 5))
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(in_features=64 * 4 * 4, out_features=512)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=512, out_features=out_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.flatten(x)
        x = self.fc2(self.relu3(self.fc1(x)))
        return x


class CNN_FashionMNIST(nn.Module):
    """
    CNN for Fashion-MNIST dataset
    https://arxiv.org/pdf/2008.06180.pdf
    """

    def __init__(self, in_channels=1, out_classes=10):
        super(CNN_FashionMNIST, self).__init__()

        # Convolutional Layers
        self.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=32, kernel_size=3, padding=1
        )
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=3, padding=1
        )
        self.bn2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, padding=1
        )
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, padding=1
        )
        self.bn4 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, padding=1
        )
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(
            in_channels=128, out_channels=128, kernel_size=3, padding=1
        )
        self.bn6 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.relu = nn.ReLU()

        self.flatten = nn.Flatten()

        # Fully Connected Layers
        self.fc1 = nn.Linear(in_features=1152, out_features=382)
        self.fc2 = nn.Linear(in_features=382, out_features=192)
        self.fc3 = nn.Linear(in_features=192, out_features=out_classes)

    def forward(self, x):
        x = self.pool1(
            self.relu(self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x))))))
        )
        x = self.pool2(
            self.relu(self.bn4(self.conv4(self.relu(self.bn3(self.conv3(x))))))
        )
        x = self.pool3(
            self.relu(self.bn6(self.conv6(self.relu(self.bn5(self.conv5(x))))))
        )
        x = self.flatten(x)
        x = self.fc3(self.relu(self.fc2(self.relu(self.fc1(x)))))
        return x


def ResNet18_CIFAR10():
    model = torchvision.models.resnet18(num_classes=10)
    model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = torch.nn.Identity()
    return model


def ResNet18_CIFAR100():
    model = torchvision.models.resnet18(num_classes=100)
    model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = torch.nn.Identity()
    return model
