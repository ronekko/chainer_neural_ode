# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 11:17:51 2019

@author: ryuhei
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from tqdm import tqdm

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda, Variable

from datasets import create_points, plot_points


class Net(chainer.Chain):
    def __init__(self, dim_h=50):
        super(Net, self).__init__()

        with self.init_scope():
            self.fc1 = L.Linear(2, dim_h)
            self.fc2 = L.Linear(dim_h, dim_h)
            self.fc3 = L.Linear(dim_h, dim_h)
            self.fc4 = L.Linear(dim_h, dim_h)
            self.fc_out = L.Linear(dim_h, 2)
#            self.norm1 = L.BatchNormalization(dim_h)
#            self.norm2 = L.BatchNormalization(dim_h)
#            self.norm3 = L.BatchNormalization(dim_h)
#            self.norm4 = L.BatchNormalization(dim_h)

    def forward(self, x):
        h = self.fc1(x)
#        h = self.norm1(h)
        h = F.relu(h)
        h = self.fc2(h)
#        h = self.norm2(h)
        h = F.relu(h)
        h = self.fc3(h)
#        h = self.norm3(h)
        h = F.relu(h)
        y = self.fc4(h)
#        h = self.norm4(h)
        h = F.relu(h)
        y = self.fc_out(h)
        return y


class Net2(chainer.Chain):
    def __init__(self, dim_h=100):
        super(Net2, self).__init__()

        with self.init_scope():
            self.fc1 = L.Linear(2, dim_h)
            self.fc21 = L.Linear(dim_h, dim_h//4)
            self.fc22 = L.Linear(dim_h//4, dim_h)
            self.fc31 = L.Linear(dim_h, dim_h//4)
            self.fc32 = L.Linear(dim_h//4, dim_h)
            self.fc41 = L.Linear(dim_h, dim_h//4)
            self.fc42 = L.Linear(dim_h//4, dim_h)
            self.fc_out = L.Linear(dim_h, 2)
#            self.norm1 = L.BatchNormalization(dim_h)
#            self.norm2 = L.BatchNormalization(dim_h)
#            self.norm3 = L.BatchNormalization(dim_h)

    def forward(self, x):
        h = self.fc1(x)
#        h = self.norm1(h)
        h = h + self.fc22(F.relu(self.fc21(F.relu(h))))
#        h = self.norm2(h)
        h = h + self.fc32(F.relu(self.fc31(F.relu(h))))
#        h = self.norm3(h)
        h = h + self.fc42(F.relu(self.fc41(F.relu(h))))
        y = self.fc_out(h)
        return y


if __name__ == '__main__':
    n_sqrt = 30
    batch_size = 50
    num_epochs = 100

    X, Y = create_points(n_sqrt)
    dataset = chainer.datasets.TupleDataset(X, Y)
    it = chainer.iterators.SerialIterator(dataset, batch_size, shuffle=True)

#    net = Net()
    net = Net2()
    optimizer = chainer.optimizers.Adam().setup(net)

    losses = []
    for epoch in tqdm(range(num_epochs)):
        for batch in it:
            x, y = chainer.dataset.concat_examples(batch)

            y_pred = net(x)
            loss = F.mean_squared_error(y_pred, y)
            losses.append(cuda.to_cpu(loss.array))

            net.cleargrads()
            loss.backward()
            optimizer.update()

            if it.is_new_epoch:
                break
        break
        # Evaluate
        plt.plot(losses)
        plt.grid()
        plt.show()

        with chainer.using_config('train', False):
            Y_pred = cuda.to_cpu(net(X).array)
            plot_points(X, Y_pred)
