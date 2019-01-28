# -*- coding: utf-8 -*-
"""
Created on Sun Jan 27 06:12:39 2019

@author: ryuhei
"""

import copy

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from tqdm import tqdm

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda, Variable

from datasets import create_points, plot_points


class OdeStage(chainer.Chain):
    def __init__(self, ode_block, rtol=1e-3, atol=1e-3):
        super(OdeStage, self).__init__()
        self.rtol = rtol
        self.atol = atol
        with self.init_scope():
            self.ode_block = ode_block

    def forward(self, x):
        ode_stage_function = OdeStageFunction(
            self.ode_block, self.rtol, self.atol)
        x = ode_stage_function(x)
        return x


class OdeStageFunction(chainer.Function):
    def __init__(self, ode_block, rtol=1e-3, atol=1e-3):
        self.ode_block = ode_block
        self.rtol = rtol
        self.atol = atol
        self.nfe_forward = 0
        self.nfe_backward = 0

    def forward(self, inputs):
        xp = chainer.cuda.get_array_module(*inputs)
        x_t0 = inputs[0]
        t = xp.array([0.0, 1.0], dtype=np.float32)  # t_0 = 0 and t_1 = 1
        x_shape = x_t0.shape
        flat_x_t0 = x_t0.ravel()
        flat_solutions = odeint(self._forward_f, flat_x_t0, t,
                                args=(self, x_shape),
                                rtol=self.rtol, atol=self.atol)
        x_t1 = flat_solutions[-1].reshape(x_shape)
        x_t1 = x_t1.astype(np.float32)
        self.retain_outputs((0,))
        print('Forward NFE:', self.nfe_forward)
        return x_t1,

    def backward(self, inputs, grad_inputs):
        xp = chainer.cuda.get_array_module(*grad_inputs)
        gx_t1 = grad_inputs[0]
        x_t1 = self.output_data[0]
        x_shape = x_t1.shape

        gt1 = xp.sum(gx_t1 * x_t1).reshape(1)
        flat_x_t1 = x_t1.ravel()
        flat_gx_t1 = gx_t1.ravel()
        flat_params = self._get_flat_params()
        flat_s_t1 = xp.concatenate((flat_x_t1,   # [0:X]
                                    flat_gx_t1,  # [X:2X]
                                    xp.zeros_like(flat_params),  # [2X:2X+P]
                                    -gt1))  # [2X+P]

        # Backward in time from t=1 to t=0.
        t = xp.array([1.0, 0.0], dtype=np.float32)

        flat_solutions = odeint(self._backward_f, flat_s_t1, t,
                                args=(self, x_shape),
                                rtol=self.rtol, atol=self.atol)

        flat_s_t0 = flat_solutions[-1].astype(np.float32)

        # Unpack the flat array `s_t0` to arrays
        x_size = len(flat_x_t1)
        gx_t0 = flat_s_t0[x_size:2*x_size].reshape(x_shape)  # [X:2X]
        # [2X:2X+P]
        flat_gparams = flat_s_t0[2*x_size:2*x_size+len(flat_params)]
        self._accumulate_grads_of_params(flat_gparams)

        print('Backward NFE:', self.nfe_backward)
        return gx_t0,

    @staticmethod
    def _forward_f(x, t, self, x_shape):
        x = x.astype(np.float32)
        x = x.reshape(x_shape)
        xp = chainer.cuda.get_array_module(x)
        t = xp.array(t, dtype=np.float32)

        with chainer.no_backprop_mode():
            dxdt = self.ode_block(x, t).array

        dxdt = dxdt.ravel()

        self.nfe_forward += 1
        return dxdt

    @staticmethod
    def _backward_f(s, t, self, x_shape):
        xp = chainer.cuda.get_array_module(s)

        # Extract `x(t)` from the augmented state `s(t)`
        x_size = 1
        for l in x_shape:
            x_size *= l
        flat_x = s[:x_size]
        x = flat_x.reshape(x_shape)
        x = Variable(x.astype(np.float32))

        # Wrap `t` as Variable
        t_var = Variable(xp.array(t, dtype=np.float32))

        # Extract `a(t)` from the augmented state
        flat_a = s[x_size:2*x_size]
        a = flat_a.reshape(x_shape)
        a = a.astype(np.float32)

        # Compute grads
        with chainer.force_backprop_mode():
            fx = self.ode_block(x, t_var)
        grads = chainer.grad((fx,),
                             (x,) + self._get_params() + (t_var,),
                             (a,))

        flat_grads = xp.concatenate([grad.array.ravel() for grad in grads])
        s = xp.concatenate([fx.array.ravel(), -flat_grads])

        self.nfe_backward += 1
        return s

    def _get_params(self):
        params = tuple(self.ode_block.params())
        return params

    def _get_flat_params(self):
        xp = self.ode_block.xp
        params = self._get_params()
        return xp.concatenate(tuple(p.array.ravel() for p in params))

    def _get_flat_grads_of_params(self):
        xp = self.ode_block.xp
        params = self._get_params()
        return xp.concatenate(tuple(p.grad.ravel() for p in params))

    def _accumulate_grads_of_params(self, flat_gparams):
        i_start = 0
        for param in self._get_params():
            i_stop = i_start + param.array.size
            flat_gp = flat_gparams[i_start:i_stop]
            gp = flat_gp.reshape(param.shape)
            if param.grad is None:
                param.grad = gp
            else:
                param.grad += gp

            i_start = i_stop


def my_odeint(func, x0, t=None, rtol=1e-3, atol=1e-3):
    xp = chainer.cuda.get_array_module(x0)
    if t is None:
        xp.array([0, 1], dtype=np.float32)

    return OdeStageFunction(func, rtol, atol)(x0, t)


class OdeBlock(chainer.Chain):
    def __init__(self, dim_x=50, dim_mid=None):
        super(OdeBlock, self).__init__()

        if dim_mid is None:
            dim_mid = dim_x
        with self.init_scope():
            self.fc1 = L.Linear(dim_x + 1, dim_mid)
            self.fc2 = L.Linear(dim_mid + 1, dim_x)

    def forward(self, x, t):
        t_shape = (x.shape[0], 1) + x.shape[2:]
        t_ext = F.broadcast_to(t, t_shape)

        x = F.concat((x, t_ext), axis=1)
        x = self.fc1(x)
        x = F.relu(x)
        x = F.concat((x, t_ext), axis=1)
        x = self.fc2(x)
        return x


class Net(chainer.Chain):
    def __init__(self, dim_h=50, dim_h_mid=None, rtol=1e-3, atol=1e-3):
        super(Net, self).__init__()

        if dim_h_mid is None:
            dim_h_mid = dim_h
        with self.init_scope():
            self.fc1 = L.Linear(2, dim_h)
            self.ode_stage = OdeStage(OdeBlock(dim_h, dim_h_mid))
            self.fc_out = L.Linear(dim_h, 2)

    def forward(self, x):
        h = self.fc1(x)
        h = self.ode_stage(h)
        h = F.relu(h)
        y = self.fc_out(h)
        return y


class Net2(chainer.Chain):
    def __init__(self, dim_h=50, dim_h_mid=None, rtol=1e-3, atol=1e-3):
        super(Net2, self).__init__()

        if dim_h_mid is None:
            dim_h_mid = dim_h
        with self.init_scope():
            self.ode_stage = OdeStage(OdeBlock(dim_h, dim_h_mid))

    def forward(self, x):
        x = chainer.Variable(x)
        return self.ode_stage(x)


if __name__ == '__main__':
    n_sqrt = 30
    batch_size = 100
    num_epochs = 1000

    X, Y = create_points(n_sqrt)
    dataset = chainer.datasets.TupleDataset(X, Y)
    it = chainer.iterators.SerialIterator(dataset, batch_size, shuffle=True)

#    net = Net()
    net = Net2(2, 50)

#    optimizer = chainer.optimizers.Adam().setup(net)
    optimizer = chainer.optimizers.RMSprop(1e-2).setup(net)

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
            print()

            if it.is_new_epoch:
                break

        # Evaluate
        plt.plot(losses)
        plt.grid()
        plt.show()

        with chainer.using_config('train', False):
            Y_pred = cuda.to_cpu(net(X).array)
            plot_points(X, Y_pred)

