import numpy as np
import pandas as pd

from collections import deque
from scipy import signal
import copy
from scipy.fft import fft

# # REQUIRED for online inference
def duty_cycle_mod(D0, Ts, Tsample, Tsim):
    t = np.linspace(0, Ts, round(Ts/Tsample), endpoint=False)
    s_pri = deque(signal.square(2*np.pi/Ts*t, D0))
    vp = np.array(s_pri).clip(0, 1)
    vp = np.tile(vp, (round(Tsim/Ts),))
    return vp


import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data
from torch.utils.data import Dataset, DataLoader
from torch import nn
import math
import copy


def init_params(params, init='uniform'):
    for param in params:
        if param.requires_grad and len(param.shape) > 0:
            stddev = 1 / math.sqrt(param.shape[0])
            if init == 'uniform':
                torch.nn.init.uniform_(param, a=-0.05, b=0.05)
            elif init == 'normal':
                torch.nn.init.normal_(param, std=stddev)


def normalize_gradient(net):
    grad_norm_sq = 0
    for p in net.parameters():
        if p.grad is None: continue
        grad_norm_sq += p.grad.data.norm() ** 2
    grad_norm = math.sqrt(grad_norm_sq)
    if grad_norm < 1e-4:
        net.zero_grad()
        print('grad norm close to zero')
    else:
        for p in net.parameters():
            if p.grad is None: continue
            p.grad.data.div_(grad_norm)
    return grad_norm


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


import pandas as pd
import numpy as np
import torch
from torch import nn
import math
import psutil
import time

import matplotlib.pyplot as plt
import matplotlib

from io import BytesIO


# REQUIRED for online inference
class Implicit_PINN(nn.Module):
    def __init__(self, cell, **kwargs):
        super(Implicit_PINN, self).__init__(**kwargs)
        self.cell = cell

    def forward(self, inputs, x):
        outputs = []
        _x = x[:, 0]
        for t in range(inputs.shape[1]):
            _x = self.cell.forward(inputs[:, t, :], _x)
            outputs.append(_x)
        return torch.stack(outputs, dim=1)  # , torch.stack(outputs_va, dim=1)


# # REQUIRED for online inference
# # NEED OPTIMIZATION of CODES
class ImplicitEulerCell(nn.Module):
    def __init__(self, Vin, dt, L, Co, Ro, **kwargs):
        super(ImplicitEulerCell, self).__init__(**kwargs)
        self.Vin = nn.Parameter(torch.Tensor([Vin]))
        self.dt = dt
        self.L = nn.Parameter(torch.Tensor([L]))
        self.Co = nn.Parameter(torch.Tensor([Co]))
        self.Ro = nn.Parameter(torch.Tensor([Ro]))

    # Explicit method for training
    def forward(self, inputs, states):
        # explict method for training
        # inputs : represent s_pri, respectively
        vo = states[:, 1]
        va = self.Vin * inputs[:, 0]
        idx = (inputs[:, 0] == 0) & (states[:, 0] <= 0)
        va[idx] = vo[idx]
        iL_next = states[:, 0] + self.dt / self.L * (va - vo)  #
        iL_next = torch.relu(iL_next)
        vC_next = states[:, 1] + self.dt / self.Co * (states[:, 0] - vo / self.Ro)
        return torch.stack((iL_next, vC_next), dim=1)


L = 490e-6
Co = 13e-6
Tslen = 200

def optimization(Vin,Vo,PL,fs,i_ripple_lim,v_ripple_lim):
    D1 = Vo / Vin
    Ro = Vo ** 2 / PL
    Ts, dt = 1 / fs, 1 / fs / Tslen

    scripted_ImplicitEulerCell = torch.jit.script(ImplicitEulerCell(Vin, dt, L, Co, Ro))
    scripted_Implicit_PINN = torch.jit.script(Implicit_PINN(scripted_ImplicitEulerCell))

    model_implicit_PINN = scripted_Implicit_PINN

    def obj_func(x, model_PINN,
                 Vin, Vref, return_all=False,
                 i_ripple_lim=i_ripple_lim, v_ripple_lim=v_ripple_lim):
        sim_len = 150
        Lf, Cf = x.T
        model_PINN.cell.L = torch.FloatTensor(Lf)
        model_PINN.cell.Co = torch.FloatTensor(Cf)

        model_PINN.eval()
        with torch.no_grad():
            state = torch.zeros((len(x), 1, 2))
            state[..., 1] = Vref * 0.8
            D1 = 0 * state[..., 0] + Vref / Vin
            vp0 = []

            for _D1 in D1:
                _vp = duty_cycle_mod(_D1.item() - 0.005,
                                     Ts, dt, Ts * sim_len)
                vp0.append(torch.FloatTensor(_vp)[None, :, None])
            vp0 = torch.cat(vp0, dim=0)
            _input = vp0

            with torch.no_grad():
                pred = model_PINN.forward(_input, state).numpy()
                iL = pred[:, -3 * round(Ts / dt):, 0]
                Vo = pred[:, -3 * round(Ts / dt):, 1]

        iL_mean = np.mean(iL, axis=1)
        Vo_mean = np.mean(Vo, axis=1)

        i_ripple = (np.max(iL, axis=1) - np.min(iL, axis=1)) / iL_mean
        v_ripple = (np.max(Vo, axis=1) - np.min(Vo, axis=1)) / Vo_mean

        i_ripple_value = np.max(iL, axis=1) - np.min(iL, axis=1)
        v_ripple_value = np.max(Vo, axis=1) - np.min(Vo, axis=1)

        obj_value = np.ones((len(pred),)) * 1e5
        idx = i_ripple <= i_ripple_lim
        idx2 = v_ripple <= v_ripple_lim
        idx_all = idx & idx2
        obj_value[idx_all] = (Lf * 1e5 + Cf * 1e6)[idx_all]
        if return_all:
            return _input, pred, i_ripple, v_ripple, i_ripple_value, v_ripple_value, iL_mean, Vo_mean
        return obj_value

    import pyswarms as ps

    def optimize_cs(nums, model_PINN, Vin, Vref,upper_bound,lower_bound):
        upper_bounds = np.array(upper_bound)
        lower_bounds = np.array(lower_bound)
        PSO_optimizer = ps.single.GlobalBestPSO(n_particles=50, dimensions=len(upper_bounds),
                                                bounds=(lower_bounds, upper_bounds),
                                                options={'c1': 2.05, 'c2': 2.05, 'w': 0.9},
                                                bh_strategy=bh_strategy,
                                                velocity_clamp=None,  # (lower_bounds*0.2, upper_bounds*0.2)
                                                vh_strategy=vh_strategy,
                                                oh_strategy={"w": "lin_variation"})
        cost, pos = PSO_optimizer.optimize(obj_func, nums,
                                           model_PINN=model_PINN,
                                           Vin=Vin, Vref=Vref,
                                           i_ripple_lim=i_ripple_lim,
                                           v_ripple_lim=v_ripple_lim,verbose=False)
        return cost, pos

    upper_bound = [2e-3, 100e-6]
    lower_bound = [50e-6, 1e-6]
    bh_strategy = "periodic"
    vh_strategy = "unmodified"
    np.random.seed(888)
    cost, pos = optimize_cs(40, model_implicit_PINN, Vin, Vo,upper_bound,lower_bound)
    # 提取值
    L_best = pos[0]
    Co_best = pos[1]

    # 保留4位有效数字
    L_best = float(f"{L_best:.4g}")
    Co_best = float(f"{Co_best:.4g}")
    _input, pred, i_ripple, v_ripple, i_ripple_value, v_ripple_value, iL_mean, Vo_mean = obj_func(np.array([pos]),
                                                                                                  model_implicit_PINN,
                                                                                                  Vin, Vo,
                                                                                                  return_all=True,
                                                                                                  i_ripple_lim=i_ripple_lim,
                                                                                                  v_ripple_lim=v_ripple_lim)
    i_ripple_value=i_ripple_value[0]
    i_ripple_value=float(f"{i_ripple_value:.3f}")

    v_ripple_value=v_ripple_value[0]
    v_ripple_value = float(f"{v_ripple_value:.3f}")

    i_ripple_percentage = "{:.2f}%".format(i_ripple[0] * 100)
    v_ripple_percentage = "{:.2f}%".format(v_ripple[0] * 100)


    # compute fft values
    fft_iL = np.abs(fft(pred[0, -round(Ts / dt):, 0]))
    fft_Vo = np.abs(fft(pred[0, -round(Ts / dt):, 1]))
    fft_iL[0] /= round(Ts / dt)
    fft_Vo[0] /= round(Ts / dt)
    fft_iL[1:] /= round(Ts / dt) / 2
    fft_Vo[1:] /= round(Ts / dt) / 2

    iLdc=fft_iL[0]
    iL1=fft_iL[1]
    iL2 = fft_iL[2]
    iL3=fft_iL[3]
    iLdc=float("{:.3g}".format(iLdc))
    iL1 = float("{:.3g}".format(iL1))
    iL2 = float("{:.3g}".format(iL2))
    iL3 = float("{:.3g}".format(iL3))

    Vodc=fft_Vo[0]
    Vo1 = fft_Vo[1]
    Vo2 = fft_Vo[2]
    Vo3 = fft_Vo[3]
    Vodc = float("{:.3g}".format(Vodc))
    Vo1 = float("{:.3g}".format(Vo1))
    Vo2 = float("{:.3g}".format(Vo2))
    Vo3 = float("{:.3g}".format(Vo3))

    def prep4loss(input_, iL, Vin):
        # switching on
        Vds_swon = Vin
        Id_swon = iL.min()

        # switching off
        Vds_swoff = Vin
        Id_swoff = iL.max()

        # conduction
        Irms_on = ((input_ * iL) ** 2).mean().sqrt()

        return (Vds_swon, Id_swon), (Vds_swoff, Id_swoff), Irms_on

    (Vds_swon, Id_swon), (Vds_swoff, Id_swoff), Irms_on = prep4loss(_input[0, -3 * round(Ts / dt):, 0],
                                                                    pred[0, -3 * round(Ts / dt):, 0], Vin)

    def Loss(Vds_swon, Id_swon, Vds_swoff, Id_swoff, Irms_on, fs):
        tdon = 15 * 1e-9
        tr = 22 * 1e-9
        tdoff = 24 * 1e-9
        tf = 14 * 1e-9
        Rds = 80 * 1e-3
        # 计算开通损耗
        switchingon_loss = 0.5 * Vds_swon * Id_swon * (tdon + tr) * fs
        P_on = round(switchingon_loss, 3)
        # 计算关断损耗
        switchingoff_loss = 0.5 * Vds_swoff * Id_swoff * (tdoff + tf) * fs
        P_off = round(switchingoff_loss, 3)
        # 计算导通损耗
        P_cond_tensor = Irms_on ** 2 * Rds
        P_cond_value = P_cond_tensor.item()
        P_cond = round(P_cond_value, 3)

        return P_on, P_off, P_cond

    P_on, P_off, P_cond = Loss(Vds_swon, Id_swon, Vds_swoff, Id_swoff, Irms_on, fs)

    return L_best,Co_best,i_ripple_value,v_ripple_value,i_ripple_percentage,v_ripple_percentage ,iLdc,iL1,iL2,iL3,Vodc,Vo1,Vo2,Vo3,P_on,P_off,P_cond
#L_best,Co_best,i_ripple_value,v_ripple_value,i_ripple_percentage,v_ripple_percentage ,iLdc,iL1,iL2,iL3,Vodc,Vo1,Vo2,Vo3,P_on,P_off,P_cond=optimization(200,80,800,5e4,0.2,0.005)
#print(L_best,Co_best,i_ripple_value,v_ripple_value,i_ripple_percentage,v_ripple_percentage ,iLdc,iL1,iL2,iL3,Vodc,Vo1,Vo2,Vo3,P_on,P_off,P_cond)
def draw(L, Co, Vin, Vref, PL, fs):
    D1 = Vref / Vin
    Ro = Vref ** 2 / PL
    Tslen = 200
    Ts, dt = 1 / fs, 1 / fs / Tslen
    scripted_ImplicitEulerCell = torch.jit.script(ImplicitEulerCell(Vin, dt, L, Co, Ro))
    scripted_Implicit_PINN = torch.jit.script(Implicit_PINN(scripted_ImplicitEulerCell))

    model_implicit_PINN = scripted_Implicit_PINN

    bs = 100
    sim_len = 100  # the number of switching cycles
    model_implicit_PINN.cell.L = L * (torch.rand((bs,)) * 2)
    model_implicit_PINN.cell.Co = Co * (torch.rand((bs,)) * 2)
    model_implicit_PINN.cell.L[-1] = torch.tensor([L])
    model_implicit_PINN.cell.Co[-1] = torch.tensor([Co])
    state = torch.zeros((bs, 1, 2))
    state[..., 1] = 60.
    D1 = 0 * state[..., 0] + Vref / Vin
    vp0 = []

    for _D1 in D1:
        _vp = duty_cycle_mod(_D1.item() - 0.005, Ts, dt, Ts * sim_len)
        _vp = torch.FloatTensor(_vp)
        vp0.append(_vp[None, :, None])
    vp0 = torch.cat(vp0, dim=0)
    # print(vp0.shape, state.shape)
    _input = vp0

    with torch.no_grad():
        pred = model_implicit_PINN.forward(_input, state)
        Vo = pred[..., 1]

    def draw_plot(data, x_labels, ylabel):
        fig, ax = plt.subplots()
        ax.plot(x_labels, data)
        ax.set_ylabel(ylabel)
        ax.set_xlabel('Time (us)')

        buf = BytesIO()
        fig.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)

        return buf

    # 电感电流图像
    data1 = pred[-1, -10 * round(Ts / dt):, 0].detach().numpy()
    x_values1 = np.arange(len(data1))
    x_labels1 = x_values1 * dt * 1e6
    buf1 = draw_plot(data1, x_labels1, 'Inductor current iL (A)')

    # 电容电压图像
    data2 = pred[-1, -10 * round(Ts / dt):, 1].detach().numpy()
    x_values2 = np.arange(len(data2))
    x_labels2 = x_values2 * dt * 1e6

    buf2 = draw_plot(data2, x_labels2, 'Capacitance voltage Vc (V)')
    # 返回字节缓冲区
    return buf1, buf2
#L_best,Co_best,i_ripple_value,v_ripple_value,i_ripple_percentage,v_ripple_percentage ,iLdc,iL1,iL2,iL3,Vodc,Vo1,Vo2,Vo3,P_on,P_off,P_cond=optimization(200,80,800,5e4,0.2,0.005)
#print(L_best,Co_best,i_ripple_value,v_ripple_value,i_ripple_percentage,v_ripple_percentage ,iLdc,iL1,iL2,iL3,Vodc,Vo1,Vo2,Vo3,P_on,P_off,P_cond)
#plot1,plot2=draw(L_best, Co_best, 200, 80, 800, 5e4)
def answer1(L,C,v_ripple_value,v_ripple_percentage,i_ripple_value,i_ripple_percentage):
   L=round(L*1e6,2)
   C=round(C*1e6,2)
   response="The optimal inductance L is designed to be {} uH, and the optimal capacitance C is designed to be {} uF. The output voltage ripple and inductor current ripple are {} V ({}) and {} A ({}), respectively. ".format(L,C,v_ripple_value,v_ripple_percentage,i_ripple_value,i_ripple_percentage)
   return response

def answer2(iLdc,iL1,iL2,iL3,Vodc,Vo1,Vo2,Vo3):
    response="The harmonics of inductor current are {} A (DC), {} A (1st),  {} A (2nd), {} A (3rd)… The harmonics of output voltage are {} V (DC), {} V (1st), {} V (2nd), {} V (3rd)…".format(iLdc,iL1,iL2,iL3,Vodc,Vo1,Vo2,Vo3)
    return response

def answer3(P_on,P_off,P_cond):
    response="According to the C2M0080120D MOSFET datasheet, the operating conditions and circuit parameter values, the switching-on loss for the C2M0080120D MOSFET is approximately {} W, and the switching-off loss is approximately {} W. The conduction loss for the C2M0080120D MOSFET is approximately {} W.".format(P_on,P_off,P_cond)
    return response
#answer1=ansewer1(L_best,Co_best,v_ripple_value,v_ripple_percentage,i_ripple_value,i_ripple_percentage)
#answer2=answer2(iLdc,iL1,iL2,iL3,Vodc,Vo1,Vo2,Vo3)
#answer3=answer3(P_on,P_off,P_cond)

#print(answer1,answer2,answer3)
def draw(L, Co, Vin, Vref, PL, fs):
    D1 = Vref / Vin
    Ro = Vref ** 2 / PL
    Tslen = 200
    Ts, dt = 1 / fs, 1 / fs / Tslen
    scripted_ImplicitEulerCell = torch.jit.script(ImplicitEulerCell(Vin, dt, L, Co, Ro))
    scripted_Implicit_PINN = torch.jit.script(Implicit_PINN(scripted_ImplicitEulerCell))

    model_implicit_PINN = scripted_Implicit_PINN

    bs = 100
    sim_len = 100  # the number of switching cycles
    model_implicit_PINN.cell.L = L * (torch.rand((bs,)) * 2)
    model_implicit_PINN.cell.Co = Co * (torch.rand((bs,)) * 2)
    model_implicit_PINN.cell.L[-1] = torch.tensor([L])
    model_implicit_PINN.cell.Co[-1] = torch.tensor([Co])
    state = torch.zeros((bs, 1, 2))
    state[..., 1] = 60.
    D1 = 0 * state[..., 0] + Vref / Vin
    vp0 = []

    for _D1 in D1:
        _vp = duty_cycle_mod(_D1.item() - 0.005, Ts, dt, Ts * sim_len)
        _vp = torch.FloatTensor(_vp)
        vp0.append(_vp[None, :, None])
    vp0 = torch.cat(vp0, dim=0)
    # print(vp0.shape, state.shape)
    _input = vp0

    with torch.no_grad():
        pred = model_implicit_PINN.forward(_input, state)
        Vo = pred[..., 1]

    def draw_plot(data, x_labels, ylabel):
        fig, ax = plt.subplots()
        ax.plot(x_labels, data)
        ax.set_ylabel(ylabel)
        ax.set_xlabel('Time (us)')

        buf = BytesIO()
        fig.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)

        return buf

    # 电感电流图像
    data1 = pred[-1, -10 * round(Ts / dt):, 0].detach().numpy()
    x_values1 = np.arange(len(data1))
    x_labels1 = x_values1 * dt * 1e6
    buf1 = draw_plot(data1, x_labels1, 'Inductor current iL (A)')

    # 电容电压图像
    data2 = pred[-1, -10 * round(Ts / dt):, 1].detach().numpy()
    x_values2 = np.arange(len(data2))
    x_labels2 = x_values2 * dt * 1e6

    buf2 = draw_plot(data2, x_labels2, 'Capacitance voltage Vc (V)')
    # 返回字节缓冲区
    return buf1, buf2
