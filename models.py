## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
         ## Conv layers
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 96, kernel_size = 7,stride=2,padding=0)
        self.conv2 = nn.Conv2d(in_channels = 96, out_channels = 256, kernel_size = 5,stride=1,padding=1)
        self.conv3 = nn.Conv2d(in_channels = 256, out_channels = 384, kernel_size = 3,stride=1,padding=1)
        self.conv4 = nn.Conv2d(in_channels = 384, out_channels = 384, kernel_size = 2,stride=1,padding=1)
        self.conv5=nn.Conv2d(in_channels = 384, out_channels = 256, kernel_size = 2,stride=1,padding=1)

        
        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)

        
        self.fc1 = nn.Linear(in_features = 50176, out_features = 1000) 
        self.fc2 = nn.Linear(in_features = 1000,    out_features = 136)
       

        
        
        self.drop = nn.Dropout(p = 0.5)
        




    def forward(self, x):
        x=self.pool(F.relu(self.conv1(x)))
        x=self.pool(F.relu(self.conv2(x)))
        x=F.relu(self.conv3(x))
        x=F.relu(self.conv4(x))
        x=self.pool(F.relu(self.conv5(x)))
        x=x.view(x.size(0),-1)
        x=self.drop(x)
        x=F.relu(self.fc1(x))
        x=self.drop(x)
        x=self.fc2(x)
        return x