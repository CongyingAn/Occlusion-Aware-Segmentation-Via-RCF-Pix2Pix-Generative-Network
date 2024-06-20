import torch
import numpy as np
import cv2
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

class SobelLoss(nn.Module):

    def __init__(self):
        super(SobelLoss, self).__init__()

        sobel_x = torch.Tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        sobel_y = torch.Tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        sobel_3x = torch.Tensor(1, 1, 3, 3)
        sobel_3y = torch.Tensor(1, 1, 3, 3)
        sobel_3x[:, 0, :, :] = sobel_x
        sobel_3y[:, 0, :, :] = sobel_y
        self.conv_hx = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_hy = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_hx.weight.data = sobel_3x.to(self.conv_hx.weight.device).type(self.conv_hx.weight.dtype)
        self.conv_hy.weight.data = sobel_3y.to(self.conv_hy.weight.device).type(self.conv_hy.weight.dtype)

    def forward(self, X, Y):
        # Ensure X and Y are on the same device and of the same data type as the weight tensors
        X = X.to(self.conv_hx.weight.device).type(self.conv_hx.weight.dtype)
        Y = Y.to(self.conv_hy.weight.device).type(self.conv_hy.weight.dtype)

        Y = Y.expand_as(X)

        X_hx = self.conv_hx(X)
        X_hy = self.conv_hy(X)
        G_X = torch.abs(X_hx) + torch.abs(X_hy)

        Y_hx = self.conv_hx(Y)
        self.conv_hx.train(False)
        Y_hy = self.conv_hy(Y)
        self.conv_hy.train(False)
        G_Y = torch.abs(Y_hx) + torch.abs(Y_hy)

        loss = F.mse_loss(G_X, G_Y, reduction='mean')

        return loss

   # if __name__=='__main__':
   # 		criterion = GradientLoss()
   #  	loss = criterion(output,target)
