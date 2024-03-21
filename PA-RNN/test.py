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
from pinn_train import *
from star_optim import *
import matplotlib.pyplot as plt
import matplotlib



def plot_modulation(idx, inputs, pred):
    fig, ax1 = plt.subplots(figsize=(9, 6))
    ax2 = ax1.twinx()  
    ax1.plot(inputs[idx,:,0],label='Input 1',color='r')
    ax1.plot(inputs[idx,:,1],label='Input 2',color='g')
    ax2.plot(pred[idx,:,0],label='Input 3',color='b')
    plt.show()


if __name__ == "__main__":
    
    # define network
    n = 2.0
    RL = 120e-3
    Lr = 65e-6
    scripted_ImplicitEulerCell = torch.jit.script(ImplicitEulerCell(dt, Lr, RL, n))
    model_implicit_PINN = torch.jit.script(Implicit_PINN(scripted_ImplicitEulerCell))
    
    
    Vin, Vref = 200, 80
    # 5DOF
    D0 = [0.008*i for i in range(1, 11)]
    D1 = [1-0.016*i for i in range(0, 10)]
    D2 = [0.96-0.016*i for i in range(0, 10)]
    phi1 = [-0.16+0.032*i for i in range(0, 10)]
    phi2 = [0.16-0.032*i for i in range(0, 10)]
    inputs = torch.FloatTensor(get_inputs(D0, D1, D2, phi1, phi2, Vin, Vref))
    pred, inputs = evaluate(inputs, None, model_implicit_PINN, Vin, convert_to_mean=True)
    plot0=plot_modulation(7, inputs, pred)
    
    
    # TPS
    D0 = [0.008*i for i in range(1, 11)]
    D1 = [1-0.016*i for i in range(0, 10)]
    D2 = [0.96-0.016*i for i in range(0, 10)]
    phi1 = [0.0 for i in range(0, 10)] # set to 0, 50% duty cycle for devices
    phi2 = [0.0 for i in range(0, 10)] # set to 0, 50% duty cycle for devices%
    inputs = torch.FloatTensor(get_inputs(D0, D1, D2, phi1, phi2, Vin, Vref))
    pred, inputs = evaluate(inputs, None, model_implicit_PINN, Vin, convert_to_mean=True)
    plot1=plot_modulation(7, inputs, pred)
    
    
    # EPS
    D0 = [0.008*i for i in range(1, 11)]
    D1 = [1-0.016*i for i in range(0, 10)]
    D2 = [1.0 for i in range(0, 10)] # set to 1
    phi1 = [0.0 for i in range(0, 10)] # set to 0, 50% duty
    phi2 = [0.0 for i in range(0, 10)] # set to 0, 50% duty
    inputs = torch.FloatTensor(get_inputs(D0, D1, D2, phi1, phi2, Vin, Vref))
    pred, inputs = evaluate(inputs, None, model_implicit_PINN, Vin, convert_to_mean=True)
    plot2=plot_modulation(7, inputs, pred)
    
    D0 = [0.008*i for i in range(1, 11)]
    D1 = [1 for i in range(0, 10)]
    D2 = [0.96-0.016*i for i in range(0, 10)]
    phi1 = [0.0 for i in range(0, 10)]
    phi2 = [0.0 for i in range(0, 10)]
    inputs = torch.FloatTensor(get_inputs(D0, D1, D2, phi1, phi2, Vin, Vref))
    pred, inputs = evaluate(inputs, None, model_implicit_PINN, Vin, convert_to_mean=True)
    plot3=plot_modulation(7, inputs, pred)
    
    
    # DPS
    D0 = [0.008*i for i in range(1, 11)]
    D1 = [1-0.016*i for i in range(0, 10)]
    D2 = D1
    phi1 = [0.0 for i in range(0, 10)]
    phi2 = [0.0 for i in range(0, 10)]
    inputs = torch.FloatTensor(get_inputs(D0, D1, D2, phi1, phi2, Vin, Vref))
    pred, inputs = evaluate(inputs, None, model_implicit_PINN, Vin, convert_to_mean=True)
    plot4=plot_modulation(7, inputs, pred)
    
    
    # SPS
    D0 = [0.008*i for i in range(1, 11)]
    D1 = [1.0 for i in range(0, 10)]
    D2 = D1
    phi1 = [0.0 for i in range(0, 10)]
    phi2 = [0.0 for i in range(0, 10)]
    inputs = torch.FloatTensor(get_inputs(D0, D1, D2, phi1, phi2, Vin, Vref))
    pred, inputs = evaluate(inputs, None, model_implicit_PINN, Vin, convert_to_mean=True)
    plot5=plot_modulation(7, inputs, pred)
    
    
    
    
    # Current-stress-oriented Optimization
    upper_bounds = [0.36, 1.0, 1.0, 0.36, 0.36]
    lower_bounds = [-0.16, 0.72, 0.72, -0.36, -0.36]
    P_required = 200
    current_Stress,pos=optimize_cs(P_required, Vin, Vref, 50, model_implicit_PINN, upper_bounds, lower_bounds,"5DOF")
    print(current_Stress)
    
    
    
    # Training