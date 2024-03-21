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

from pinn_utils import *


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
    if len(attrs) == 1: return getattr(module, attrs[0])
    else: return getattr_(getattr(module, attrs[0]), ".".join(attrs[1:]))
        
    
class Implicit_PINN(nn.Module):
    """
        Define the generic physics-in-architecture recurrent neural network (PA-RNN) structure
        
        References:
            1: Temporal Modeling for Power Converters With Physics-in-Architecture Recurrent Neural Network
                Paper DOI: 10.1109/TIE.2024.3352119
            2: Data-Light Physics-Informed Modeling for the Modulation Optimization of a Dual-Active-Bridge Converter
                Paper DOI: 10.1109/TPEL.2024.3378184
    """
    
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
    
    
class ImplicitEulerCell(nn.Module):
    """
        Define implicit Euler Recurrent Cell for the PA-RNN
    """
    
    def __init__(self, dt, Lr, RL, n, **kwargs):
        super(ImplicitEulerCell, self).__init__(**kwargs)
        self.dt = dt
        self.Lr = nn.Parameter(torch.Tensor([Lr]))
        self.RL = nn.Parameter(torch.Tensor([RL]))
        self.n = nn.Parameter(torch.Tensor([n]))
        
    def forward(self, inputs, states):
        iL_next = (self.Lr/(self.Lr+self.RL*self.dt))*states[:, 0]+\
                    (self.dt/(self.Lr+self.RL*self.dt))*(inputs[:, 0]-self.n*inputs[:, 1])
        return iL_next[:, None]
    
    
def evaluate(inputs, targets, model_PINN, Vin, convert_to_mean=True):
    """
        Evaluate all for PINN
    """
    
    model_PINN = model_PINN.to("cpu")
    model_PINN.eval()
    with torch.no_grad():
        if targets is None:
            state0 = torch.zeros((inputs.shape[0], 1, 1))
        else:
            state0 = targets[:, 0:1]
        pred = model_PINN.forward(inputs, state0)
        
        pred, inputs = transform(inputs[:, -2*Tslen:], pred[:, -2*Tslen:], Vin,
                                convert_to_mean=convert_to_mean)
        if targets is None:
            return pred, inputs
        test_loss = (targets[:, 1:]-pred).abs().mean().item()
        return pred, inputs, test_loss

