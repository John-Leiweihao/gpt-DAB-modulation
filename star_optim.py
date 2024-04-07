from pinn_utils import *
from pinn_net import *
def locate(v, V):
    """Should be applied after sync"""
    idx = [None]*4
    v0 = v[0]
    for i in range(1, len(v)+1):
        i = i % len(v)
        dV = v[i]-v0
        if (dV > V/2):
            if v0 < -V/2: idx[0] = i
            else: idx[1] = i
            if (dV > V*1.5):
                idx[1] = i
        elif (dV < -V/2):
            if v0 > V/2: idx[2] = i
            else: idx[3] = i
            if (dV < -V*1.5):
                idx[3] = i
        v0 = v[i]
    return idx
def eval_cs(pred, criterion="ipp"):
    assert criterion in ["ipp", "irms"]
    if criterion == "ipp":
        current_stress = (pred.max(axis=1).ravel() - pred.min(axis=1).ravel())
    elif criterion == "irms":
        current_stress = (pred[..., 0] ** 2).mean(axis=1)
    return current_stress


def eval_ZVZCS(inputs, pred, Vin, Vref):
    ZVS = np.zeros((len(pred),))
    ZCS = np.zeros((len(pred),))
    threshold = 1e-2  # EXTENSION
    for i in range(len(pred)):
        index_p = locate(inputs[i, :, 0], Vin)
        index_s = locate(inputs[i, :, 1], Vref)
        i_p = pred[i, index_p, 0]
        i_s = pred[i, index_s, 0]
        ZVS[i] = (i_p[:2] <= threshold).sum() + (i_p[2:] >= -threshold).sum() + \
                 (i_s[:2] >= -threshold).sum() + (i_s[2:] <= threshold).sum()
        ZCS[i] = (np.abs(i_p) <= threshold).sum() + (np.abs(i_s) <= threshold).sum()
    return ZVS, ZCS

import torch
def obj_func(x, model_PINN, P_required,
             Vin, Vref, modulation="5DOF",
             with_ZVS=False, return_all=False):
    if modulation == "5DOF":
        D0, D1, D2, phi1, phi2 = x.T.tolist()
    elif modulation == "TPS":
        D0, D1, D2 = x.T.tolist()
        phi1, phi2 = [0.0] * len(D0), [0.0] * len(D0)
    elif modulation == "DPS":
        D0, D1 = x.T.tolist()
        D2, phi1, phi2 = D1, [0.0] * len(D0), [0.0] * len(D0)
    elif modulation == "EPS1":
        D0, D1 = x.T.tolist()
        D2, phi1, phi2 = [1.0] * len(D0), [0.0] * len(D0), [0.0] * len(D0)
    elif modulation == "EPS2":
        D0, D2 = x.T.tolist()
        D1, phi1, phi2 = [1.0] * len(D0), [0.0] * len(D0), [0.0] * len(D0)
    elif modulation == "SPS":
        D0 = x.flatten().tolist()  # 假设x是二维数组，将其展平并转换为列表
        D1, D2, phi1, phi2 = [1.0 for _ in D0], [1.0 for _ in D0], [0.0 for _ in D0], [0.0 for _ in D0]

    model_PINN.eval()
    with torch.no_grad():
        inputs = torch.FloatTensor(get_inputs(D0, D1, D2, phi1, phi2,Vin,Vref))
        pred, inputs = evaluate(inputs, None, model_PINN,Vin, convert_to_mean=True)
        P_predicted = (inputs[..., 0] * pred[..., 0]).mean(dim=1).numpy()
        pred, inputs = pred.numpy(), inputs.numpy()
        ipp = eval_cs(pred, criterion="ipp")
        # irms = eval_cs(pred, criterion="irms")

    if with_ZVS:
        ZVS, ZCS = eval_ZVZCS(inputs, pred, Vin, Vref)
    else:
        ZVS, ZCS = 0, 0  # do not consider ZVS and ZCS performances

    penalty = np.zeros((len(pred),))
    P_threshold = 5.
    idx = np.abs(P_predicted - P_required) > P_threshold
    # penalty[idx] = 100.0
    penalty[idx] = (np.abs(P_predicted[idx] - P_required) - P_threshold) * 10
    ipp_origin = copy.deepcopy(ipp)
    ipp[~idx] = ipp[~idx] * P_required / P_predicted[~idx]  # *(np.abs(P_predicted[~idx]-P_required)/P_required+1)
    if return_all:
        return ipp_origin, P_predicted, pred, inputs, ZVS, ZCS, penalty
    return 5*ipp-(ZVS+ZCS)*20+penalty

import pyswarms as ps
import numpy as np

def optimize_cs(nums, model_PINN,
                P_required, Vin, Vref, modulation,upper_bound,lower_bound,bh_strategy,vh_strategy,
                with_ZVS=False):
    upper_bounds = np.array(upper_bound)
    lower_bounds = np.array(lower_bound)
    PSO_optimizer = ps.single.GlobalBestPSO(n_particles=50, dimensions=len(upper_bounds), bounds=(lower_bounds,
                                                                                  upper_bounds),
                                            options={'c1': 2.05, 'c2': 2.05, 'w':0.9},
                                            bh_strategy=bh_strategy,
                                            velocity_clamp=(lower_bounds*0.1, upper_bounds*0.1),
                                            vh_strategy=vh_strategy,
                                            oh_strategy={"w": "lin_variation"})
    cost, pos = PSO_optimizer.optimize(obj_func, nums,
                                       model_PINN=model_PINN,
                                       P_required=P_required,
                                       modulation=modulation,
                                       Vin=Vin, Vref=Vref,
                                       with_ZVS=with_ZVS)
    return cost, pos

