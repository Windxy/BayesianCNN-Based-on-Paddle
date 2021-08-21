import numpy as np
import paddle.nn.functional as F
from paddle import nn
import paddle


class ELBO(nn.Layer):
    def __init__(self, train_size):
        super(ELBO, self).__init__()
        self.train_size = train_size

    def forward(self, input, target, kl, beta):
        assert target.stop_gradient     #原来nn的True，为执行反向传播，这里的paddle的True，为执行不反向传播
        y1 = F.nll_loss(input, target, reduction='mean') * self.train_size 
        # y1 = F.cross_entropy(input, target)
        y2 = beta * kl
        # print("F.nll_loss",y1.numpy(),"beta * kl",y2.numpy())
        y = y1 + y2
        return y


# def lr_linear(epoch_num, decay_start, total_epochs, start_value):
#     if epoch_num < decay_start:
#         return start_value
#     return start_value*float(total_epochs-epoch_num)/float(total_epochs-decay_start)


def acc(outputs, targets,flag=False):
    if flag:
        out = paddle.argmax(outputs,axis=1)
        targets = paddle.squeeze(targets)
        print(out.numpy())
        print(targets.numpy())
        print()
        return 0.0
    # ans = out == targets
    # a = paddle.sum(ans)
    # b = out.shape[0]
    # ans = a*1.0/b
    ans = paddle.metric.accuracy(outputs,targets)
    return ans.numpy()


def calculate_kl(mu_q, sig_q, mu_p, sig_p):
    kl = paddle.sum(0.5 * (2 * paddle.log(sig_p / sig_q) - 1 + (sig_q / sig_p).pow(2) + ((mu_p - mu_q) / sig_p).pow(2)))
    # kl = paddle.sum(0.5 * (2 * paddle.log(sig_p / sig_q) - 1 + paddle.pow((sig_q / sig_p),2) + paddle.pow(((mu_p - mu_q) / sig_p),2)))
    # kl = paddle.sum((mu_q-mu_p).pow(2)+(sig_q-sig_p).pow(2))
    return kl


def get_beta(batch_idx, m, beta_type, epoch, num_epochs):
    if type(beta_type) is float:
        return beta_type

    if beta_type == "Blundell":
        beta = 2 ** (m - (batch_idx + 1)) / (2 ** m - 1)
    elif beta_type == "Soenderby":
        if epoch is None or num_epochs is None:
            raise ValueError('Soenderby method requires both epoch and num_epochs to be passed.')
        beta = min(epoch / (num_epochs // 4), 1)
    elif beta_type == "Standard":
        beta = 1 / m
    else:
        beta = 0
    return beta
