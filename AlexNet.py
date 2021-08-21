from paddle.vision.transforms import Compose, Normalize
import numpy as np
import matplotlib.pyplot as plt
import paddle
import paddle.nn.functional as F

transform = Compose([Normalize(mean=[127.5],
                               std=[127.5],
                               data_format='CHW')])
# 使用transform对数据集做归一化
print('download training data and load training data')
train_dataset = paddle.vision.datasets.MNIST(mode='train', transform=transform)
test_dataset = paddle.vision.datasets.MNIST(mode='test', transform=transform)
print('load finished')

class AlexNet(paddle.nn.Layer):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv1 = nn.Conv2d(inputs, 96, 11, stride=4, padding=5, bias=True, priors=self.priors)
        self.act1 = nn.ReLU()
        self.pool1 = nn.MaxPool2D(kernel_size=2, stride=2)  

        self.conv2 = nn.Conv2d(96, 256, 5, padding=2, bias=True, priors=self.priors)
        self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool2D(kernel_size=2, stride=2)  

        self.conv3 = BBBConv2d(256, 384, 3, padding=1, bias=True, priors=self.priors)
        self.act3 = nn.ReLU()

        self.conv4 = BBBConv2d(384, 384, 3, padding=1, bias=True, priors=self.priors)
        self.act4 = nn.ReLU()

        self.conv5 = BBBConv2d(384, 256, 3, padding=1, bias=True, priors=self.priors)
        self.act5 = nn.ReLU()
        self.pool3 = nn.MaxPool2D(kernel_size=2, stride=2)  

        self.flatten = FlattenLayer(1 * 1 * 256)
        self.classifier = BBBLinear(1 * 1 * 256, outputs, bias=True, priors=self.priors)
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.max_pool1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.max_pool2(x)
        x = paddle.flatten(x, start_axis=1,stop_axis=-1)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.linear3(x)
        return x

train_loader = paddle.io.DataLoader(train_dataset, batch_size=256, shuffle=True)
def train(model):
    model.train()
    epochs = 2
    optim = paddle.optimizer.Adam(learning_rate=0.001, parameters=model.parameters())
    # 用Adam作为优化函数
    for epoch in range(epochs):
        for batch_id, data in enumerate(train_loader()):
            x_data = data[0]
            y_data = data[1]
            predicts = model(x_data)
            loss = F.cross_entropy(predicts, y_data)
            # 计算损失
            acc = paddle.metric.accuracy(predicts, y_data)
            loss.backward()
            if batch_id % 300 == 0:
                print("epoch: {}, batch_id: {}, loss is: {}, acc is: {}".format(epoch, batch_id, loss.numpy(), acc.numpy()))
            optim.step()
            optim.clear_grad()


test_loader = paddle.io.DataLoader(test_dataset, places=paddle.CPUPlace(), batch_size=64)
# 加载测试数据集
def test(model):
    model.eval()
    batch_size = 8
    for batch_id, data in enumerate(test_loader()):
        x_data = data[0]
        y_data = data[1]
        predicts = model(x_data)
        # 获取预测结果
        loss = F.cross_entropy(predicts, y_data)
        acc = paddle.metric.accuracy(predicts, y_data)
        if batch_id % 1 == 0:
            print("batch_id: {}, loss is: {}, acc is: {}".format(batch_id, loss.numpy(), acc.numpy()))

if __name__ == '__main__':
    model = LeNet()
    train(model)
    test(model)