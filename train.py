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

def train(net,epochs,train_loader,optim,scheduler,num_ens,criterion,beta_type=0.1):
    net.train()
    all_loss = 0.0
    accs = []
    kl_list = []

    for epoch in range(epochs):
        for batch_id, data in enumerate(train_loader(),1):
            x_data = data[0]
            y_data = data[1]

            outputs = []
            kl = 0.0
            for j in range(num_ens):
                net_out, _kl = net(x_data)
                kl += _kl
                outputs.append(F.log_softmax(net_out, axis=1).unsqueeze(2))

            kl = kl / num_ens
            kl_list.append(kl.item())
            outputs = paddle.concat(outputs, axis=-1)
            log_outputs = utils.logmeanexp(outputs, dim=2)

            beta = metrics.get_beta(batch_id - 1, len(train_loader), beta_type, epoch, epochs)
            loss = criterion(log_outputs, y_data, kl, beta)
            loss = loss[0]
            all_loss += loss.numpy()
            loss.backward()

            accs.append(metrics.acc(log_outputs, y_data))
            if batch_id % 15 == 0:
                print('Epoch: {} Training Loss: {:.4f} Training Accuracy: {:.4f} train_kl_div: {:.4f}'.format(
                        epoch, float(all_loss/15.0), np.mean(accs),np.mean(kl_list)))
                accs.clear()
                kl_list.clear()
                all_loss = 0.0
                scheduler.step(loss)
            optim.step()
            optim.clear_grad()

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
    return 100*(1.0 - np.mean(accs))

def run(dataset, net_type):
    # 超参数设置
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

    # transform = Compose([Normalize(mean=[127.5],std=[127.5],data_format='CHW')])    # 归一化
    train_dataset = paddle.vision.datasets.MNIST(mode='train',transform=Compose([ToTensor(),Resize((32,32))]))
    test_dataset = paddle.vision.datasets.MNIST(mode='test',transform=Compose([ToTensor(),Resize((32,32))]))

    train_loader = paddle.io.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers=num_workers)
    test_loader = paddle.io.DataLoader(test_dataset, batch_size=batch_size,num_workers=num_workers)

    net = getModel(net_type, 1, 10, priors, layer_type, activation_type)

    criterion = metrics.ELBO(len(train_dataset))
    scheduler = paddle.optimizer.lr.ReduceOnPlateau(learning_rate=lr_start, factor=0.5, patience=5, verbose=True)
    adam = paddle.optimizer.Adam(learning_rate=scheduler, weight_decay=paddle.regularizer.L2Decay(coeff=1e-7),parameters=net.parameters())

    train(net=net,
          epochs=n_epochs,
          train_loader=train_loader,
          optim=adam,scheduler=scheduler,
          num_ens=train_ens,
          criterion=criterion,
          beta_type=beta_type)

    acc = test(net=net,
         test_loader=test_loader,
         criterion=criterion,
         num_ens=valid_ens,
         beta_type=beta_type)
    paddle.save(net.state_dict(), 'weight/{}_{}.pdparams'.format(net_type,acc))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "PyTorch Bayesian Model Training")
    parser.add_argument('--net_type', default='lenet', type=str, help='model')
    parser.add_argument('--dataset', default='MNIST', type=str, help='dataset = [MNIST/CIFAR10/CIFAR100]')
    args = parser.parse_args()
    run(args.dataset, args.net_type)