import numpy as np
import pandas as pd
import io

from collections import deque
from scipy import signal
import copy
import psutil

import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import matplotlib.pyplot as plt
import matplotlib
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data
from torch.utils.data import Dataset, DataLoader
from torch import nn
import math
n = 0.98
RL = 500e-3
Lr = 60e-6
Ts = 1/50e3
Tslen = 250
dt = Ts/Tslen
def sync(vp, Vin):
    idx = 0
    for i in range(len(vp)-1):
        if (vp[i]<-Vin/2) and (vp[i+1]>=-Vin/2):
            idx = i+1
            break
    return idx


# For hybrid phase shift and duty cycle modulation (5-DOF)
def create_vpvs(D0, D1, D2, Vin, Vref, Ts, Tsample, Tsim,
                D1_cycle=0.5, D2_cycle=0.5):
    D0 = D0 + D1 - D2
    t = np.linspace(0, Ts, round(Ts / Tsample), endpoint=False)
    s_pri = signal.square(2 * np.pi / Ts * t, D1_cycle)
    s_pri2 = deque(-s_pri)
    s_pri2.rotate(int(np.ceil(np.round(D1 * Ts / 2 / Tsample, 5))))

    s_sec = deque(signal.square(2 * np.pi / Ts * t, D2_cycle))
    s_sec.rotate(int(np.ceil(np.round(D0 * Ts / 2 / Tsample, 5))))
    s_sec2 = deque(-np.array(s_sec))
    s_sec2.rotate(int(np.ceil(np.round(D2 * Ts / 2 / Tsample, 5))))

    vp = (np.array(s_pri) + np.array(s_pri2)).clip(-1, 1) * Vin
    vs = (np.array(s_sec) + np.array(s_sec2)).clip(-1, 1) * Vref

    vp = np.tile(vp, (round(Tsim / Ts),))
    vs = np.tile(vs, (round(Tsim / Ts),))

    idx = sync(vp, Vin)
    vp = vp[idx:Tslen * (round(Tsim / Ts) - 1) + idx]
    vs = vs[idx:Tslen * (round(Tsim / Ts) - 1) + idx]
    return vp, vs


# For 5DOF
def transform(input_, pred, convert_to_mean=False):
    """Perform synchronization, which extends to general modulation modeling"""
    pred_o = torch.zeros(input_.shape[0], Tslen, pred.shape[-1]).to(input_.device)
    input_o = torch.zeros(input_.shape[0], Tslen, input_.shape[-1]).to(input_.device)

    for i in range(input_.shape[0]):
        vp = input_[i, :, 0]
        idx = sync(vp, 200)
        pred_o[i] = pred[i, idx:idx + Tslen]
        input_o[i] = input_[i, idx:idx + Tslen]
        if convert_to_mean:
            pred_o[i, :, 0] -= pred_o[i, :, 0].mean()
    return pred_o, input_o


class WeightClamp(object):
    """
        Clamp the weights to specified limits
        arguments:
            arg::attrs -> a list of attributes in 'str' format for the respective modules
            arg::limits -> a list of limits for the respective modules,
                            where limits[idx] follows [lower bound, upper bound]
    """

    def __init__(self, attrs, limits):
        self.attrs = attrs
        self.limits = limits

    def __call__(self, module):
        for i, (attr, limit) in enumerate(zip(self.attrs, self.limits)):
            w = getattr_(module, attr).data
            w = w.clamp(limit[0], limit[1])
            getattr_(module, attr).data = w


def getattr_(module, attr):
    # recurrence to the final layer of attributes
    attrs = attr.split('.')
    if len(attrs) == 1:
        return getattr(module, attrs[0])
    else:
        return getattr_(getattr(module, attrs[0]), ".".join(attrs[1:]))


class Implicit_PINN(nn.Module):
    def __init__(self, cell, **kwargs):
        super(Implicit_PINN, self).__init__(**kwargs)
        self.cell = cell

    def forward(self, inputs, x):
        outputs = []
        _x = x[:, 0]
        for t in range(inputs.shape[1]):
            state_next = self.cell.forward(inputs[:, t, :], _x)
            _x = state_next
            outputs.append(_x)
        return torch.stack(outputs, dim=1)


# REQUIRED for online inference
class ImplicitEulerCell(nn.Module):
    def __init__(self, dt, Lr, RL, n, **kwargs):
        super(ImplicitEulerCell, self).__init__(**kwargs)
        self.dt = dt
        self.Lr = nn.Parameter(torch.Tensor([Lr]))
        self.RL = nn.Parameter(torch.Tensor([RL]))
        self.n = nn.Parameter(torch.Tensor([n]))

    def forward(self, inputs, states):
        iL_next = (self.Lr / (self.Lr + self.RL * self.dt)) * states[:, 0] + \
                  (self.dt / (self.Lr + self.RL * self.dt)) * (inputs[:, 0] - self.n * inputs[:, 1])
        return iL_next[:, None]


"""Evaluate all for PINN"""


def evaluate(inputs, targets, model_PINN, convert_to_mean=True):
    model_PINN = model_PINN.to("cpu")
    model_PINN.eval()
    with torch.no_grad():
        if targets is None:
            state0 = torch.zeros((inputs.shape[0], 1, 1))
        else:
            state0 = targets[:, 0:1]
        pred = model_PINN.forward(inputs, state0)

        #         pred = pred-(pred[:, -2*Tslen:].max(dim=1)[0]+pred[:, -2*Tslen:].min(dim=1)[0])[..., None]/2
        pred, inputs = transform(inputs[:, -2 * Tslen:], pred[:, -2 * Tslen:],
                                 convert_to_mean=convert_to_mean)
        if targets is None:
            return pred, inputs
        test_loss = (targets[:, 1:] - pred).abs().mean().item()
        return pred, inputs, test_loss

def Training_PINN(inputs,states):
    scripted_ImplicitEulerCell = torch.jit.script(ImplicitEulerCell(dt, Lr, RL, n))
    model_implicit_PINN = torch.jit.script(Implicit_PINN(scripted_ImplicitEulerCell))
    train_pct = 0.1 * 1
    test_pct = 0.45

    np.random.seed(888)
    idx = np.random.permutation(inputs.shape[0])
    train_inputs = inputs[idx[:round(train_pct * inputs.shape[0])]]
    test_inputs = inputs[idx[round(train_pct * inputs.shape[0]):round((train_pct + test_pct) * inputs.shape[0])]]
    val_inputs = inputs[idx[round((train_pct + test_pct) * inputs.shape[0]):]]

    train_states = states[idx[:round(train_pct * inputs.shape[0])]]
    test_states = states[idx[round(train_pct * inputs.shape[0]):round((train_pct + test_pct) * inputs.shape[0])]]
    val_states = states[idx[round((train_pct + test_pct) * inputs.shape[0]):]]

    class CustomDataset(Dataset):
        def __init__(self, states, inputs, targets):
            super(CustomDataset, self).__init__()
            self.states = states  # states are training x
            self.inputs = inputs
            self.targets = targets

        def __getitem__(self, index):
            return self.states[index], self.inputs[index], self.targets[index]

        def __len__(self):
            return len(self.states)

    train_inputs = torch.Tensor(train_inputs)
    test_inputs = torch.Tensor(test_inputs)
    val_inputs = torch.Tensor(val_inputs)
    train_states = torch.Tensor(train_states)
    test_states = torch.Tensor(test_states)
    val_states = torch.Tensor(val_states)

    data_loader = DataLoader(
        dataset=CustomDataset(train_states[:, :-1], train_inputs[:, 0:-1],
                              train_states[:, 1:]),
        batch_size=40,
        shuffle=True,
        drop_last=False)

    data_loader_test = DataLoader(
        dataset=CustomDataset(test_states[:, :-1], test_inputs[:, 0:-1],
                              test_states[:, 1:]),
        batch_size=110,
        shuffle=True,
        drop_last=False)
    param_list = ['cell.Lr']
    params = list(filter(lambda kv: kv[0] in param_list, model_implicit_PINN.named_parameters()))
    base_params = list(filter(lambda kv: kv[0] not in param_list, model_implicit_PINN.named_parameters()))
    optimizer_implicit_PINN = torch.optim.Adam([{"params": [param[1] for param in params], "lr": 5e-6},
                                                {"params": [base_param[1] for base_param in base_params]}], lr=1e-1)

    clamp1 = WeightClamp(['cell.Lr', 'cell.RL', 'cell.n'],
                         [(10e-6, 200e-6), (1000e-6, 5e0), (0.85, 1.15)])  # clamp the coefficient F
    loss_pinn = nn.MSELoss()
    device = "cpu"

    MIN_implicit_PINN = np.inf
    best_implicit_PINN = None

    model_implicit_PINN.train()
    model_implicit_PINN = model_implicit_PINN.to(device)
    for epoch in range(50):
        # Forward pass
        total_loss = 0.
        for data in data_loader:
            """ Logic is:
                input_ (full length) -> smooth_all -> PINN pred -> segment final Tslen*2 points -> sync
            """
            state, input_, target = data
            state, input_, target = state.to(device), input_.to(device), target.to(device)
            #         state0 = state[:, :1] # should be zero to avoid learning the initial state
            state0 = torch.zeros(state.shape).to(device)  # should be zero to avoid learning the initial state
            pred = model_implicit_PINN.forward(input_, state0)
            Vin = 200
            pred, _ = transform(input_[:, -2 * Tslen:], pred[:, -2 * Tslen:])

            loss_train = loss_pinn(pred, target)
            optimizer_implicit_PINN.zero_grad()
            loss_train.backward()
            optimizer_implicit_PINN.step()
            clamp1(model_implicit_PINN)  # comment out this line if using pure data-driven model for dk
            total_loss += loss_train.item()
       # print(list(map(lambda x: round(x.item(), 7), model_implicit_PINN.parameters())))
        #print(f"Epoch {epoch}, Training loss {total_loss / len(data_loader)}")
        if epoch % 1 == 0:
            *_, test_loss = evaluate(test_inputs[:, 1:], test_states, model_implicit_PINN)
            if test_loss < MIN_implicit_PINN:
                MIN_implicit_PINN, best_implicit_PINN = test_loss, copy.deepcopy(model_implicit_PINN)
                #print(f"New loss is {MIN_implicit_PINN}.")
                #print('-' * 81)
    model_implicit_PINN = best_implicit_PINN
    #print(model_implicit_PINN.cell.Lr.item(), model_implicit_PINN.cell.RL.item())

    test_pred, test_inputs, test_loss = evaluate(test_inputs[:, 0:], test_states, model_implicit_PINN)
    val_pred, val_inputs, val_loss = evaluate(val_inputs[:, 0:], val_states, model_implicit_PINN)
    #print("Mean absolute errors are: ", test_loss, val_loss)

    plt.plot(val_pred[2, :, 0].detach().numpy(), label='Predicted waveform')
    plt.plot(val_states[2, 1:, 0].detach().numpy(), label='Experimental waveform')
    plt.legend()
    plt.show()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    # 返回字节缓冲区
    return buf,test_loss,val_loss