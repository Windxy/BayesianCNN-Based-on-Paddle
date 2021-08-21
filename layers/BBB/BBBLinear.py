import sys
sys.path.append("..")

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from ..misc import ModuleWrapper

from metrics import calculate_kl as KL_DIV

class BBBLinear(ModuleWrapper):
    def __init__(self, in_features, out_features, bias=True, priors=None):
        super(BBBLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias

        if priors is None:
            priors = {
                'prior_mu': 0,
                'prior_sigma': 0.1,
                'posterior_mu_initial': (0, 0.1),
                'posterior_rho_initial': (-3, 0.1),
            }
        self.prior_mu = priors['prior_mu']
        self.prior_sigma = priors['prior_sigma']
        self.posterior_mu_initial = priors['posterior_mu_initial']
        self.posterior_rho_initial = priors['posterior_rho_initial']

        x = paddle.empty([in_features,out_features])
        self.W_mu = paddle.create_parameter(shape=x.shape,
                        dtype=str(x.numpy().dtype),
                        default_initializer=paddle.nn.initializer.Normal(*self.posterior_mu_initial))

        self.W_rho = paddle.create_parameter(shape=x.shape,
                        dtype=str(x.numpy().dtype),
                        default_initializer=paddle.nn.initializer.Normal(*self.posterior_rho_initial))

        if self.use_bias:
            y = paddle.empty([out_features])
            self.bias_mu = paddle.create_parameter(shape=y.shape,
                        dtype=str(y.numpy().dtype),
                        default_initializer=paddle.nn.initializer.Normal(*self.posterior_mu_initial))
            self.bias_rho = paddle.create_parameter(shape=y.shape,
                        dtype=str(y.numpy().dtype),
                        default_initializer=paddle.nn.initializer.Normal(*self.posterior_rho_initial))


    def forward(self, input, sample=True):
        if self.training or sample:
            W_eps = paddle.normal(mean = 0.0,std = 1.,shape=self.W_mu.shape)
            self.W_sigma = paddle.log1p(paddle.exp(self.W_rho))

            weight = self.W_mu + W_eps * self.W_sigma

            if self.use_bias:
                bias_eps = paddle.normal(mean = 0.0,std = 1.,shape=self.bias_mu.shape)
                self.bias_sigma = paddle.log1p(paddle.exp(self.bias_rho))

                bias = self.bias_mu + bias_eps * self.bias_sigma
            else:
                bias = None
        else:
            weight = self.W_mu
            bias = self.bias_mu if self.use_bias else None

        return F.linear(input, weight, bias)

    def kl_loss(self):
        kl = KL_DIV(self.prior_mu, self.prior_sigma, self.W_mu, self.W_sigma)
        if self.use_bias:
            kl += KL_DIV(self.prior_mu, self.prior_sigma, self.bias_mu, self.bias_sigma)
        return kl
