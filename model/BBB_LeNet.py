import paddle
import paddle.nn as nn

from layers import BBB_Linear, BBB_Conv2d
from layers import BBB_LRT_Linear, BBB_LRT_Conv2d
from layers import FlattenLayer, ModuleWrapper


class BBBLeNet(ModuleWrapper):
    '''The architecture of LeNet with Bayesian Layers'''

    def __init__(self, outputs, inputs, priors, layer_type='lrt', activation_type='relu'):
        super(BBBLeNet, self).__init__()

        self.num_classes = outputs
        self.layer_type = layer_type
        self.priors = priors

        if layer_type == 'lrt':
            BBBLinear = BBB_LRT_Linear
            BBBConv2d = BBB_LRT_Conv2d
        elif layer_type == 'bbb':
            BBBLinear = BBB_Linear
            BBBConv2d = BBB_Conv2d
        else:
            raise ValueError("Undefined layer_type")

        if activation_type == 'softplus':
            self.act = nn.Softplus()
        elif activation_type == 'relu':
            self.act = nn.ReLU()
        elif activation_type == 'leakyrelu':
            self.act = nn.LeakyReLU()
        else:
            raise ValueError("Only softplus or relu supported")

        self.conv1 = BBBConv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2,priors=priors)
        self.act1 = self.act
        self.max_pool1 = nn.MaxPool2D(kernel_size=2,  stride=2)         # 16

        self.conv3 = BBBConv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1,priors=priors)#12
        self.act1 = self.act
        self.max_pool2 = nn.MaxPool2D(kernel_size=2, stride=2)   # 6

        self.flatten = FlattenLayer(6 * 6 * 16)     

        self.linear1 = BBBLinear(in_features=16*6*6, out_features=128,priors=priors)
        # self.drop2 = nn.Dropout(0.25)
        self.act3 = self.act

        self.linear2 = BBBLinear(in_features=128, out_features=64,priors=priors)
        # self.drop2 = nn.Dropout(0.25)
        self.act4 = self.act
        
        self.linear3 = BBBLinear(in_features=64, out_features=10,priors=priors)
        
    # def forward(self,x):
    #     y = self.conv1(x)
    #     y = self.max_pool1(y)
    #     y = self.conv2(y)
    #     y = self.max_pool2(y)
    #     y = self.flatten(y)
    #     y = self.linear1(y)
    #     y = self.linear2(y)
    #     y = self.linear3(y)
        
    #     kl = 0.0
    #     for module in self.sublayers():
    #         if hasattr(module, 'kl_loss'):
    #             kl = kl + module.kl_loss()

    #     return y,kl
