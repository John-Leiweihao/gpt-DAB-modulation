from pinn_net import *
from pinn_train import *
from star_optim import *
import matplotlib.pyplot as plt
import matplotlib
from io import BytesIO
def plot_modulation(idx, inputs, pred):
    fig, ax1 = plt.subplots(figsize=(9, 6))
    ax2 = ax1.twinx()
    ax1.plot(inputs[idx,:,0],label='Vp',color='r')
    ax1.plot(inputs[idx,:,1],label='Vs',color='g')
    ax2.plot(pred[idx,:,0],label='IL',color='b')
    ax1.legend(loc='upper right')  # 将输入的图例放在左上角
    ax2.legend(loc='upper left')  # 将预测的图例放在右上角
    buf = BytesIO()
    fig.savefig(buf, format='png')
    # 重要：关闭图像，防止内存泄露
    plt.close(fig)
    # 移动缓冲区的读取指针到开始位置
    buf.seek(0)
    
    # 返回字节缓冲区
    return buf
def PINN(Vin,Vref,P_required,modulation):
    n = 2.0
    RL = 120e-3
    Lr = 65e-6
    M=3
    scripted_ImplicitEulerCell = torch.jit.script(ImplicitEulerCell(dt, Lr, RL, n))
    model_implicit_PINN = torch.jit.script(Implicit_PINN(scripted_ImplicitEulerCell))
    if modulation=="SPS":
        D0 = [0.008 * i for i in range(1, 11)]
        D1 = [1.0 for i in range(0, 10)]
        D2 = D1
        phi1 = [0.0 for i in range(0, 10)]
        phi2 = [0.0 for i in range(0, 10)]
    if modulation=="EPS":
        upper_bounds = [0.36, 1.0]
        lower_bounds = [-0.16, 0.72]
        current_Stress1, pos1 = optimize_cs(P_required, Vin, Vref, 50, model_implicit_PINN, upper_bounds, lower_bounds, "EPS1")
        current_Stress2, pos2= optimize_cs(P_required, Vin, Vref, 50, model_implicit_PINN, upper_bounds, lower_bounds, "EPS2")
        if current_Stress1<current_Stress2:
            current_Stress=current_Stress1
            pos=pos1
            D0 = [0.008 * i for i in range(1, 11)]
            D1 = [1 - 0.016 * i for i in range(0, 10)]
            D2 = [1.0 for i in range(0, 10)]  # set to 1
            phi1 = [0.0 for i in range(0, 10)]  # set to 0, 50% duty
            phi2 = [0.0 for i in range(0, 10)]  # set to 0, 50% duty
            M=1
        else:
            current_Stress=current_Stress2
            pos=pos2
            D0 = [0.008 * i for i in range(1, 11)]
            D1 = [1 for i in range(0, 10)]
            D2 = [0.96 - 0.016 * i for i in range(0, 10)]
            phi1 = [0.0 for i in range(0, 10)]
            phi2 = [0.0 for i in range(0, 10)]
            M=2
    if modulation=="DPS":
        D0 = [0.008 * i for i in range(1, 11)]
        D1 = [1 - 0.016 * i for i in range(0, 10)]
        D2 = D1
        phi1 = [0.0 for i in range(0, 10)]
        phi2 = [0.0 for i in range(0, 10)]
        upper_bounds = [0.36, 1.0]
        lower_bounds = [-0.16, 0.72]
        current_Stress, pos = optimize_cs(P_required, Vin, Vref, 50, model_implicit_PINN, upper_bounds, lower_bounds,"DPS")
    if modulation=="TPS":
        D0 = [0.008 * i for i in range(1, 11)]
        D1 = [1 - 0.016 * i for i in range(0, 10)]
        D2 = [0.96 - 0.016 * i for i in range(0, 10)]
        phi1 = [0.0 for i in range(0, 10)]  # set to 0, 50% duty cycle for devices
        phi2 = [0.0 for i in range(0, 10)]  # set to 0, 50% duty cycle for devices%
        upper_bounds = [0.36, 1.0, 1.0 ]
        lower_bounds = [-0.16, 0.72, 0.72]
        current_Stress, pos = optimize_cs(P_required, Vin, Vref, 50, model_implicit_PINN, upper_bounds, lower_bounds,"TPS")
    if modulation=="Five-Degree":
        D0 = [0.008 * i for i in range(1, 11)]
        D1 = [1 - 0.016 * i for i in range(0, 10)]
        D2 = [0.96 - 0.016 * i for i in range(0, 10)]
        phi1 = [-0.16 + 0.032 * i for i in range(0, 10)]
        phi2 = [0.16 - 0.032 * i for i in range(0, 10)]
        upper_bounds = [0.36, 1.0, 1.0, 0.36, 0.36]
        lower_bounds = [-0.16, 0.72, 0.72, -0.36, -0.36]
        current_Stress, pos = optimize_cs(P_required, Vin, Vref, 50, model_implicit_PINN, upper_bounds, lower_bounds,"Five-Degree")
    inputs = torch.FloatTensor(get_inputs(D0, D1, D2, phi1, phi2, Vin, Vref))
    pred, inputs = evaluate(inputs, None, model_implicit_PINN, Vin, convert_to_mean=True)
    plot=plot_modulation(7, inputs, pred)
    return current_Stress,pos,plot,M

def answer(pos,modulation,Current_stress,M=3):
    Current_stress=round(Current_stress,2)
    if modulation=="EPS":
        n=4
        D0, D1 = round(pos[0], 3), round(pos[1], 3)
        if M==1:
            response="Under the {}{} modulation strategy,the optimal D0 is designed to be {},D1 is designed to be {}, and the inductive current waveform diagram is shown with the following figure.The current stress is {}A,the number of switches that achieve zero-voltage turn-on is {}. ".format(modulation,M,D0,D1,Current_stress,n)
        if M==2:
            response = "Under the {}{} modulation strategy,the optimal D0 is designed to be {},D2 is designed to be {}, and the inductive current waveform diagram is shown with the following figure.The current stress is {}A,the number of switches that achieve zero-voltage turn-on is {}.".format(modulation, M, D0, D1,Current_stress,n)
    if modulation=="DPS":
        n=5
        D0, D1 = round(pos[0], 3), round(pos[1], 3)
        response="Under the {}modulation strategy,the optimal D0 is designed to be {},D1 and D2 are designed to be {}, and the inductive current waveform diagram is shown with the following figure.The current stress is {}A,the number of switches that achieve zero-voltage turn-on is {}".format(modulation,D0,D1,Current_stress,n)
    if modulation=="TPS":
        n=6
        D0, D1,D2 = round(pos[0], 3), round(pos[1], 3),round(pos[2], 3)
        response="Under the {} modulation strategy,the optimal D0 is designed to be {},D1 is designed to be {},D2 is designed to be {}, and the inductive current waveform diagram is shown with the following figure.The current stress is {}A,the number of switches that achieve zero-voltage turn-on is {}".format(modulation,D0,D1,D2,Current_stress,n)
    if modulation=="Five-Degree":
        n=8
        D0,D1,D2,phi1,phi2=round(pos[0], 3), round(pos[1], 3),round(pos[2], 3), round(pos[3], 3),round(pos[4], 3)
        response="Under the {} modulation strategy,the optimal D0 is designed to be {},D1 is designed to be {},D2 is designed to be {},phi1 is designed to be {},phi2 is designed to be {}, and the inductive current waveform diagram is shown with the following figure.The current stress is {}A,the number of switches that achieve zero-voltage turn-on is {}".format(modulation,D0,D1,D2,phi1,phi2,Current_stress,n)
    return response
