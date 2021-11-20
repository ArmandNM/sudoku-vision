import torch
import numpy as np

from torchvision.models import resnet18


def get_classifier():
    model = resnet18(num_classes=10)
    model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model.load_state_dict(torch.load('./resnet_mnist.wts'))
    model.eval()
    return model


def main():
    model = get_classifier()


if __name__ == '__main__':
    main()
