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

from pinn_net import *
import pyswarms as ps




def obj_func(x, model_PINN, 
             P_required, Vin, Vref, 
             modulation):
    """
        objective function for modulation optimization
        objectives: minimal peak-to-peak current stress with required power transfer
    """
    if modulation == "Five-Degree":
        D0, D1, D2, phi1, phi2 = x.T.tolist()
    elif modulation == "TPS":
        D0, D1, D2 = x.T.tolist()
        phi1, phi2 = [0.0]*len(D0), [0.0]*len(D0)
    elif modulation == "DPS":
        D0, D1 = x.T.tolist()
        D2, phi1, phi2 = D1, [0.0]*len(D0), [0.0]*len(D0)
    elif modulation == "EPS1":
        D0, D1 = x.T.tolist()
        D2, phi1, phi2 = [1.0]*len(D0), [0.0]*len(D0), [0.0]*len(D0)
    elif modulation == "EPS2":
        D0, D2 = x.T.tolist()
        D1, phi1, phi2 = [1.0]*len(D0), [0.0]*len(D0), [0.0]*len(D0)
    
    model_PINN.eval()
    with torch.no_grad():
        inputs = torch.FloatTensor(get_inputs(D0, D1, D2, phi1, phi2, Vin, Vref))
        pred, inputs = evaluate(inputs, None, model_PINN, Vin, convert_to_mean=True)
        P_predicted = (inputs[..., 0]*pred[..., 0]).mean(dim=1).numpy()
        ipp = (pred.max(dim=1)[0].ravel()-pred.min(dim=1)[0].ravel()).numpy()
    penalty = np.zeros((len(pred),))
    penalty[np.abs(P_predicted-P_required)>min(P_required/10, 35)] = 100.0
    return ipp+penalty


def optimize_cs(P_required, Vin, Vref,
                nums, model_PINN, upper_bounds,
                lower_bounds,modulation, n_particles=100):
    upper_bounds = np.array(upper_bounds)
    lower_bounds = np.array(lower_bounds)
    dimension=len(upper_bounds)
    PSO_optimizer = ps.single.GlobalBestPSO(n_particles=n_particles, dimensions=dimension, bounds=(lower_bounds,
                                                                                  upper_bounds),
                                            options={'c1': 2.05, 'c2': 2.05, 'w':0.9},
                                            bh_strategy="nearest",
                                            velocity_clamp=(lower_bounds*0.1, upper_bounds*0.1),
                                            vh_strategy="invert",
                                            oh_strategy={"w": "lin_variation"})
    cost, pos = PSO_optimizer.optimize(obj_func, nums, 
                                       model_PINN=model_PINN,
                                       Vin=Vin, Vref=Vref,
                                       P_required=P_required,
                                       modulation=modulation)
    return cost, pos