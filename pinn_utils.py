"""
Created on Thu Mar 21 07:50:04 2024

@author: XinzeLee
@github: https://github.com/XinzeLee

@reference:
    1: Temporal Modeling for Power Converters With Physics-in-Architecture Recurrent Neural Network
        Authors: Xinze Li, Fanfan Lin (corresponding and co-first author), Huai Wang, Xin Zhang, Hao Ma, Changyun Wen and Frede Blaabjerg
        Paper DOI: 10.1109/TIE.2024.3352119
    2: Data-Light Physics-Informed Modeling for the Modulation Optimization of a Dual-Active-Bridge Converter
        Authors: Xinze Li, Fanfan Lin (corresponding and co-first author), Xin Zhang, Hao Ma and Frede Blaabjerg
        Paper DOI: 10.1109/TPEL.2024.3378184

"""

import numpy as np
import pandas as pd
import math
import copy
import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from collections import deque
from scipy import signal

import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn

from pinn_vars import *




def sync(vp, Vin):
    """
        Conduct synchronization
    """
    idx = 0
    for i in range(len(vp)-1):
        if (vp[i]<-Vin/2) and (vp[i+1]>=-Vin/2):
            idx = i+1
            break
    return idx


def create_vpvs(D0, D1, D2, Vin, Vref, 
                Ts, dt, Tsim, 
                D1_cycle=0.5, D2_cycle=0.5):
    """
        Support these modulation strategies:
            1. Single phase shift (SPS)
            2. Double phase shift (DPS)
            3. Extended phase shift (EPS)
            4. Triple phase shift (TPS)
            5. Hybrid phase shift and duty cycle (5DOF)
    """
    
    D0 = D0+D1-D2
    t = np.linspace(0, Ts, round(Ts/dt), endpoint=False)
    s_pri = signal.square(2*np.pi/Ts*t, D1_cycle)
    s_pri2 = deque(-s_pri)
    s_pri2.rotate(int(np.ceil(np.round(D1*Ts/2/dt, 5))))
    
    s_sec = deque(signal.square(2*np.pi/Ts*t, D2_cycle))
    s_sec.rotate(int(np.ceil(np.round(D0*Ts/2/dt, 5))))
    s_sec2 = deque(-np.array(s_sec))
    s_sec2.rotate(int(np.ceil(np.round(D2*Ts/2/dt, 5))))
    
    vp = (np.array(s_pri)+np.array(s_pri2)).clip(-1, 1)*Vin
    vs = (np.array(s_sec)+np.array(s_sec2)).clip(-1, 1)*Vref
    
    vp = np.tile(vp, (round(Tsim/Ts),))
    vs = np.tile(vs, (round(Tsim/Ts),))
    
    idx = sync(vp, Vin)
    vp = vp[idx:Tslen*(round(Tsim/Ts)-1)+idx]
    vs = vs[idx:Tslen*(round(Tsim/Ts)-1)+idx]
    return vp, vs


def transform(input_, pred, Vin, convert_to_mean=False):
    pred_o = torch.zeros(input_.shape[0], Tslen, pred.shape[-1]).to(input_.device)
    input_o = torch.zeros(input_.shape[0], Tslen, input_.shape[-1]).to(input_.device)
    
    for i in range(input_.shape[0]):
        vp = input_[i, :, 0]
        idx = sync(vp, Vin)
        pred_o[i] = pred[i, idx:idx+Tslen]
        input_o[i] = input_[i, idx:idx+Tslen]
        if convert_to_mean:
            pred_o[i, :, 0] -= pred_o[i, :, 0].mean()
    return pred_o, input_o


def get_inputs(D0, D1, D2, phi1, phi2, Vin, Vref):
    """
        Expert system: store the knowledge to generate key switching waveforms vp and vs
    """
    
    inputs = []
    
    for _D0, _D1, _D2, _phi1, _phi2 in zip(D0, D1, D2, phi1, phi2):
        vp_, vs_ = create_vpvs(_D0, _D1, _D2, Vin, Vref, Ts, dt, Ts*10,
                              D1_cycle=0.5+_phi1/2, D2_cycle=0.5+_phi2/2)
        _input = np.concatenate([vp_[None, :, None], 
                                  vs_[None, :, None]], axis=-1)
        inputs.append(_input)
    return np.concatenate(inputs, axis=0)




#######################################################
# Codes below will be updated in future versions #
#######################################################
def get_D012(vp, vs, Vin, Vref):
    # get D0, D1, D2 from experimental waveforms
    D1 = (vp>Vin/3*2).sum()/(Tslen/2)
    D2 = (vs>Vref/3*2).sum()/(Tslen/2)
    idx1 = np.where(np.logical_and(vp[:-1]>-Vin/2,
                                   vp[1:]>-Vin/2))[0][0]
    idx2 = np.where(np.logical_and(vs[:-1]>-Vref/2,
                                   vs[1:]>-Vref/2))[0][0]
    D0 = (idx2-idx1)/(Tslen/2)
    return D0, D1, D2


def load_waves_source(path, df_perform, Vin, skip_i=None):
    inputs = []
    inputs_origin = []
    states = [] # states after synchronize, used for training
    for i, idx in enumerate(df_perform['idx']):
        if skip_i is not None:
            if i in skip_i: continue
        vp = np.loadtxt(os.path.join(path, f"C1--XX--{idx:05}.csv"), delimiter=",", skiprows=5)[::25, 1][None, :, None]
        vs = np.loadtxt(os.path.join(path, f"C2--XX--{idx:05}.csv"), delimiter=",", skiprows=5)[::25, 1][None, :, None]
        iL = np.loadtxt(os.path.join(path, f"C3--XX--{idx:05}.csv"), delimiter=",", skiprows=5)[::25, 1][None, :, None]
        iL = iL-(iL.max()+iL.min())/2
        
        input_ = np.concatenate([vp, vs], axis=-1)
        idx1 = sync(vp[0, :, 0], Vin)
        for idx1_ in range(idx1, vp.shape[1], Tslen):
            if idx1_+Tslen+1 > vp.shape[1]:
                break
            tmp = (iL[:, idx1_+1:idx1_+1+Tslen//2]-iL[:, idx1_+1+Tslen//2:idx1_+1+Tslen])/2
            iL_ = copy.deepcopy(iL)
            iL_[:, idx1_+1:idx1_+1+Tslen//2] = tmp
            iL_[:, idx1_+1+Tslen//2:idx1_+1+Tslen] = -tmp
            try:
                D0, D1, D2 = get_D012(vp[0, idx1_:idx1_+Tslen+1, 0], 
                                    vs[0, idx1_:idx1_+Tslen+1, 0], 
                                    Vin, df_perform.iloc[i, 1])
            except Exception as e:
                print(e)
                continue
        
            vp_, vs_ = create_vpvs(D0, D1, D2, Vin, df_perform.iloc[i, 1], Ts, dt, Tsim)
            vp_, vs_ = vp_[None, :, None], vs_[None, :, None]
            states.append(iL_[:, idx1_:idx1_+Tslen+1])
            input2_ = np.concatenate([vp_, vs_], axis=-1)
            inputs.append(input2_)
            inputs_origin.append(input_)
        
    return np.concatenate(inputs, axis=0), np.concatenate(states, axis=0)




