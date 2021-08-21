from paddle.vision.transforms import Compose, Normalize,ToTensor,Resize
import paddle
import paddle.nn.functional as F
import os
import argparse
import numpy as np
import utils
import metrics
import config_bayesian as cfg
from model.BBB_AlexNet import BBBAlexNet
from model.BBB_LeNet import BBBLeNet

def getModel(net_type, inputs, outputs, priors, layer_type, activation_type):
    if (net_type == 'alexnet'):
        return BBBAlexNet(outputs, inputs, priors, layer_type, activation_type)
    if (net_type == 'lenet'):
        return BBBLeNet(outputs, inputs, priors, layer_type, activation_type)
    else:
        raise ValueError('Network should be either [AlexNet / LeNet ...')


def test(net,test_loader,criterion,num_ens,beta_type=0.1):
    net.eval()
    all_loss = 0.0

    accs = []
    for batch_id, data in enumerate(test_loader()):
        x_data = data[0]
        y_data = data[1]

        outputs = []
        kl = 0.0
        for j in range(num_ens):
            net_out, _kl = net(x_data)
            kl += _kl
            outputs.append(F.log_softmax(net_out, axis=1).unsqueeze(2))

        outputs = paddle.concat(outputs, axis=-1)
        log_outputs = utils.logmeanexp(outputs, dim=2)

        beta = metrics.get_beta(batch_id - 1, len(test_loader), beta_type, 0, 0)    # 0.1固定
        loss = criterion(log_outputs, y_data, kl, beta)
        all_loss+=loss[0].numpy()
        accs.append(metrics.acc(log_outputs, y_data))
        if batch_id % 1 == 0:
            print("batch_id: {}, loss is: {},Test Accuracy:{} Test err is: {}%".format(batch_id, float(all_loss/10.0), np.mean(accs),100*(1.0 - np.mean(accs))))
    print("Final Test err is: {}%".format(100*(1.0 - np.mean(accs))))

def run(dataset, net_type, weight_path):
    # 超参数设置
    layer_type = cfg.layer_type
    activation_type = cfg.activation_type
    priors = cfg.priors
    valid_ens = cfg.valid_ens
    lr_start = cfg.lr_start
    num_workers = cfg.num_workers
    batch_size = cfg.batch_size
    beta_type = cfg.beta_type

    test_dataset = paddle.vision.datasets.MNIST(mode='test',transform=Compose([ToTensor(),Resize((32,32))]))
    test_loader = paddle.io.DataLoader(test_dataset, batch_size=batch_size,num_workers=num_workers)

    net = getModel(net_type, 1, 10, priors, layer_type, activation_type)
    layer_state_dict = paddle.load(weight_path)
    net.set_state_dict(layer_state_dict)

    criterion = metrics.ELBO(len(test_dataset))

    test(net=net,
         test_loader=test_loader,
         criterion=criterion,
         num_ens=valid_ens,
         beta_type=beta_type)
    paddle.save(net.state_dict(), 'weight/best_model.pdparams')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "PyTorch Bayesian Model Training")
    parser.add_argument('--net_type', default='lenet', type=str, help='model')
    parser.add_argument('--dataset', default='MNIST', type=str, help='dataset = [MNIST/CIFAR10/CIFAR100]')
    parser.add_argument('--weight_path', default='weight/lenet_1.259768009185791.pdparams', type=str, help='path of weight')
    args = parser.parse_args()
    run(args.dataset, args.net_type,args.weight_path)