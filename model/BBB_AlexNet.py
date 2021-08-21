import paddle
import paddle.nn as nn

from layers import BBB_Linear, BBB_Conv2d
from layers import BBB_LRT_Linear, BBB_LRT_Conv2d
from layers import FlattenLayer, ModuleWrapper


class BBBAlexNet(ModuleWrapper):
    '''The architecture of AlexNet with Bayesian Layers'''

    def __init__(self, outputs, inputs, priors, layer_type='lrt', activation_type='relu'):
        super(BBBAlexNet, self).__init__()

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

        self.conv1 = BBBConv2d(inputs, 96, 11, stride=4, padding=5, bias=True, priors=self.priors)
        self.act1 = self.act
        self.pool1 = nn.MaxPool2D(kernel_size=2, stride=2)  

        self.conv2 = BBBConv2d(96, 256, 5, padding=2, bias=True, priors=self.priors)
        self.act2 = self.act
        self.pool2 = nn.MaxPool2D(kernel_size=2, stride=2)  

        self.conv3 = BBBConv2d(256, 384, 3, padding=1, bias=True, priors=self.priors)
        self.act3 = self.act

        self.conv4 = BBBConv2d(384, 384, 3, padding=1, bias=True, priors=self.priors)
        self.act4 = self.act

        self.conv5 = BBBConv2d(384, 256, 3, padding=1, bias=True, priors=self.priors)
        self.act5 = self.act
        self.pool3 = nn.MaxPool2D(kernel_size=2, stride=2)  

        self.flatten = FlattenLayer(1 * 1 * 256)
        self.classifier = BBBLinear(1 * 1 * 256, outputs, bias=True, priors=self.priors)

if __name__ == '__main__':
    x = paddle.rand([1,3,36,36])

    import config_bayesian as cfg
    # Hyper Parameter settings
    layer_type = cfg.layer_type
    activation_type = cfg.activation_type
    priors = cfg.priors

    train_ens = cfg.train_ens
    valid_ens = cfg.valid_ens
    n_epochs = cfg.n_epochs
    lr_start = cfg.lr_start
    num_workers = cfg.num_workers
    valid_size = cfg.valid_size
    batch_size = cfg.batch_size
    beta_type = cfg.beta_type

    net = BBBAlexNet(10,3,priors)
    print(net)
    y,kl = net(x)
    print(kl)