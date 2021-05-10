import torch
import numpy as np

from torchvision.models import resnet18


def get_classifier():
    model = resnet18(num_classes=10)
    model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model.load_state_dict(torch.load('./resnet_mnist1.wts'))
    model.eval()
    return model


def fix_prediction(digit, preds):
    # 9 can be mislabeled as 4
    if digit == 4 and np.abs(preds[9]-preds[4]) < 0.3:
        preds[4] = 0
        if np.argmax(preds) == 9:
            return 9
    
    # 4 can be mislabeled as 1
    if digit == 1 and np.abs(preds[4]-preds[1]) < 0.98:
        return 4
    
    return digit
    

def main():
    model = get_classifier()


if __name__ == '__main__':
    main()
