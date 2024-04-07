from pinn_net import *
from pinn_train import *
from star_optim import *
import matplotlib.pyplot as plt
import matplotlib
from io import BytesIO


def plot_modulation(idx, inputs, pred):
    fig, ax1 = plt.subplots(figsize=(9, 6))
    ax2 = ax1.twinx()
    ax1.plot(inputs[idx, :, 0], label='Vp', color='r')
    ax1.plot(inputs[idx, :, 1], label='Vs', color='g')
    ax2.plot(pred[idx, :, 0], label='IL', color='b')
    ax1.legend(loc='upper right')  # 将输入的图例放在左上角
    ax2.legend(loc='upper left')  # 将预测的图例放在右上角
    #plt.show()
    buf = BytesIO()
    fig.savefig(buf, format='png')
    # 重要：关闭图像，防止内存泄露
    plt.close(fig)
    # 移动缓冲区的读取指针到开始位置
    buf.seek(0)

    # 返回字节缓冲区
    return buf


def PINN(Vin, Vref, P_required, modulation):
    n = 1
    RL = 120e-3
    Lr = 63e-6
    Ts = 1 / 50e3
    Tslen = 250
    dt = Ts / Tslen
    np.random.seed(889)
    scripted_ImplicitEulerCell = torch.jit.script(ImplicitEulerCell(dt, Lr, RL, n))
    model_implicit_PINN = torch.jit.script(Implicit_PINN(scripted_ImplicitEulerCell))
    if modulation == "SPS":
        upper_bound = [0.36]
        lower_bound = [-0.16]
        obj, optimal_x = optimize_cs(50, model_implicit_PINN, 100, Vin, Vref, "SPS",upper_bound,lower_bound)
        ipp, P_predicted, pred, inputs, ZVS, ZCS, penalty = obj_func(optimal_x[None], model_implicit_PINN, 100, Vin,
                                                                 Vref, with_ZVS=True, modulation="SPS", return_all=True)
        Current_Stress = ipp[0]
        nZVS=ZVS[0]
        obj3, optimal_x3 = optimize_cs(50, model_implicit_PINN, 1000, Vin, Vref, "SPS", upper_bound, lower_bound)
        ipp3, P_predicted3, pred3, inputs3, ZVS3, ZCS3, penalty3 = obj_func(optimal_x3[None], model_implicit_PINN, 1000,
                                                                            Vin, Vref, with_ZVS=True, modulation="SPS",
                                                                            return_all=True)
        Current_Stress1=ipp3[0]
        pos = list(map(lambda x: round(x, 3), optimal_x))
        M=3

    if modulation == "EPS":
        upper_bound = [0.36, 1.0]
        lower_bound = [-0.16, 0.72]
        obj1, optimal_x1 = optimize_cs(50, model_implicit_PINN, P_required, Vin, Vref,"EPS1",upper_bound,lower_bound)
        obj2, optimal_x2 = optimize_cs(50, model_implicit_PINN, P_required, Vin, Vref, "EPS2",upper_bound,lower_bound)
        ipp1, P_predicted1, pred1, inputs1, ZVS1, ZCS1, penalty1 = obj_func(optimal_x1[None], model_implicit_PINN,P_required, Vin, Vref, with_ZVS=True, modulation="EPS1",return_all=True)
        ipp2, P_predicted2, pred2, inputs2, ZVS2, ZCS2, penalty2 = obj_func(optimal_x2[None], model_implicit_PINN,P_required, Vin, Vref, with_ZVS=True,modulation="EPS2", return_all=True)
        if ipp1 < ipp2:
            Current_Stress = ipp1[0]
            pos = list(map(lambda x: round(x,3), optimal_x1))
            nZVS=ZVS1[0]
            obj3, optimal_x3 = optimize_cs(50, model_implicit_PINN, 1000, Vin, Vref, "EPS1", upper_bound,
                                           lower_bound)
            ipp3, P_predicted3, pred3, inputs3, ZVS3, ZCS3, penalty3 = obj_func(optimal_x3[None], model_implicit_PINN,
                                                                                1000, Vin, Vref, with_ZVS=True,
                                                                                modulation="EPS1", return_all=True)
            Current_Stress1=ipp3[0]
            #D0 = [0.008 * i for i in range(1, 11)]
            #D1 = [1 - 0.016 * i for i in range(0, 10)]
           # D2 = [1.0 for i in range(0, 10)]  # set to 1
            #phi1 = [0.0 for i in range(0, 10)]  # set to 0, 50% duty
           # phi2 = [0.0 for i in range(0, 10)]  # set to 0, 50% duty
            inputs=inputs1
            pred=pred1
            M = 1
        else:
            Current_Stress = ipp2[0]
            pos = list(map(lambda x: round(x,3), optimal_x2))
            nZVS=ZVS2[0]
            obj3, optimal_x3 = optimize_cs(50, model_implicit_PINN, 1000, Vin, Vref, "EPS1", upper_bound,
                                           lower_bound)
            ipp3, P_predicted3, pred3, inputs3, ZVS3, ZCS3, penalty3 = obj_func(optimal_x3[None], model_implicit_PINN,
                                                                                1000, Vin, Vref, with_ZVS=True,
                                                                                modulation="EPS2", return_all=True)
            Current_Stress1=ipp3[0]
            #D0 = [0.008 * i for i in range(1, 11)]
            #D1 = [1 for i in range(0, 10)]
            #D2 = [0.96 - 0.016 * i for i in range(0, 10)]
            #phi1 = [0.0 for i in range(0, 10)]
            #phi2 = [0.0 for i in range(0, 10)]
            inputs=inputs2
            pred=pred2
            M = 2
    if modulation == "DPS":
        #D0 = [0.008 * i for i in range(1, 11)]
        #D1 = [1 - 0.016 * i for i in range(0, 10)]
        #D2 = D1
        #phi1 = [0.0 for i in range(0, 10)]
        #phi2 = [0.0 for i in range(0, 10)]
        upper_bound = [0.36, 1.0]
        lower_bound = [-0.16, 0.72]
        obj, optimal_x = optimize_cs(50, model_implicit_PINN, P_required, Vin, Vref,"DPS",upper_bound,lower_bound)
        ipp, P_predicted, pred, inputs, ZVS, ZCS, penalty = obj_func(optimal_x[None], model_implicit_PINN,1000, Vin, Vref, with_ZVS=True,modulation="DPS", return_all=True)
        Current_Stress=ipp[0]
        pos = list(map(lambda x: round(x, 3), optimal_x))
        nZVS=ZVS[0]
        Current_Stress=ipp[0]
        obj3, optimal_x3 = optimize_cs(50, model_implicit_PINN, 1000, Vin, Vref, "DPS", upper_bound, lower_bound)
        ipp3, P_predicted3, pred3, inputs3, ZVS3, ZCS3, penalty3 = obj_func(optimal_x3[None], model_implicit_PINN, 1000,
                                                                            Vin, Vref, with_ZVS=True, modulation="DPS",
                                                                            return_all=True)
        Current_Stress1=ipp3[0]
        M=3
    if modulation == "TPS":
        #D0 = [0.008 * i for i in range(1, 11)]
        #D1 = [1 - 0.016 * i for i in range(0, 10)]
       # D2 = [0.96 - 0.016 * i for i in range(0, 10)]
        #phi1 = [0.0 for i in range(0, 10)]  # set to 0, 50% duty cycle for devices
        #phi2 = [0.0 for i in range(0, 10)]  # set to 0, 50% duty cycle for devices%
        upper_bound = [0.36, 1.0, 1.0]
        lower_bound = [-0.16, 0.72, 0.72]
        obj, optimal_x = optimize_cs(50, model_implicit_PINN, P_required, Vin, Vref, "TPS",upper_bound,lower_bound)
        ipp, P_predicted, pred, inputs, ZVS, ZCS, penalty = obj_func(optimal_x[None], model_implicit_PINN, P_required, Vin,
                                                                 Vref, with_ZVS=True, modulation="TPS", return_all=True)
        Current_Stress = ipp[0]
        nZVS=ZVS[0]
        obj3, optimal_x3 = optimize_cs(50, model_implicit_PINN, 1000, Vin, Vref, "TPS", upper_bound, lower_bound)
        ipp3, P_predicted3, pred3, inputs3, ZVS3, ZCS3, penalty3 = obj_func(optimal_x3[None], model_implicit_PINN, 1000,
                                                                            Vin, Vref, with_ZVS=True, modulation="TPS",
                                                                            return_all=True)
        Current_Stress1=ipp3[0]
        pos = list(map(lambda x: round(x, 3), optimal_x))
        M=3
    if modulation == "5DOF":
       # D0 = [0.008 * i for i in range(1, 11)]
        #D1 = [1 - 0.016 * i for i in range(0, 10)]
        #D2 = [0.96 - 0.016 * i for i in range(0, 10)]
       # phi1 = [-0.16 + 0.032 * i for i in range(0, 10)]
       # phi2 = [0.16 - 0.032 * i for i in range(0, 10)]
        upper_bound = [0.36, 1.0, 1.0, 0.36, 0.36]
        lower_bound = [-0.16, 0.72, 0.72, -0.36, -0.36]
        obj, optimal_x = optimize_cs(50, model_implicit_PINN, P_required, Vin, Vref, "5DOF",upper_bound,lower_bound)
        ipp, P_predicted, pred, inputs, ZVS, ZCS, penalty = obj_func(optimal_x[None], model_implicit_PINN, P_required, Vin,
                                                                 Vref, with_ZVS=True, modulation="5DOF", return_all=True)
        Current_Stress = ipp[0]
        nZVS=ZVS[0]
        obj3, optimal_x3 = optimize_cs(50, model_implicit_PINN, 1000, Vin, Vref, "5DOF", upper_bound, lower_bound)
        ipp3, P_predicted3, pred3, inputs3, ZVS3, ZCS3, penalty3 = obj_func(optimal_x3[None], model_implicit_PINN, 1000,
                                                                        Vin, Vref, with_ZVS=True, modulation="5DOF",
                                                                        return_all=True)
        Current_Stress1 = ipp3[0]
        pos = list(map(lambda x: round(x, 3), optimal_x))
        M=3
    #inputs = torch.FloatTensor(get_inputs(D0, D1, D2, phi1, phi2, Vin, Vref))
    #pred, inputs = evaluate(inputs, None, model_implicit_PINN, Vin, convert_to_mean=True)
    plot = plot_modulation(0, inputs, pred)
    #plot_modulation(0, inputs, pred)
    return Current_Stress,Current_Stress1,nZVS,P_required, pos, plot, M
    #return Current_Stress,nZVS, pos, M

def answer(pos, modulation, ipp,ipp1,nZVS,P_required, M=3):
    response = "No valid modulation strategy found."
    if modulation == "EPS":
        D0, D1 = pos[0], pos[1]
        if M == 1:
            response = "Under the {}{} modulation strategy,the optimal D0 is designed to be {},D1 is designed to be {},the number of switches that achieve zero-voltage turn-on is {:.0f}. And the current stress performance is shown with the following figure.At rated power level, the peak-to-peak current is {:.2f}A. When load power PL = {}W, the peak-to-peak current is {:.2f}A.".format(modulation,M,D0,D1,nZVS,ipp1,P_required,ipp)
        if M == 2:
            response = "Under the {}{} modulation strategy,the optimal D0 is designed to be {},D1 is designed to be {},the number of switches that achieve zero-voltage turn-on is {:.0f}. And the current stress performance is shown with the following figure.At rated power level, the peak-to-peak current is {:.2f}A. When load power PL = {}W, the peak-to-peak current is {:.2f}A.".format(modulation,M,D0,D1,nZVS,ipp1,P_required,ipp)
    if modulation == "DPS":
        D0, D1 = round(pos[0], 3), round(pos[1], 3)
        response = "Under the {} modulation strategy,the optimal D0 is designed to be {},D1 and D2 are designed to be {}, the number of switches that achieve zero-voltage turn-on is {:.0f}. And the current stress performance is shown with the following figure.At rated power level, the peak-to-peak current is {:.2f}A. When load power PL = {}W, the peak-to-peak current is {:.2f}A.".format(modulation,D0,D1,nZVS,ipp1,P_required,ipp)
    if modulation == "TPS":
        D0, D1, D2 = round(pos[0], 3), round(pos[1], 3), round(pos[2], 3)
        response = "Under the {} modulation strategy,the optimal D0 is designed to be {},D1 is designed to be {},D2 is designed to be {}, the number of switches that achieve zero-voltage turn-on is {:.0f}. And the current stress performance is shown with the following figure.At rated power level, the peak-to-peak current is {:.2f}A. When load power PL = {}W, the peak-to-peak current is {:.2f}A.".format(modulation,D0,D1,D2,nZVS,ipp1,P_required,ipp)
    if modulation == "5DOF":
        D0, D1, D2, phi1, phi2 = round(pos[0], 3), round(pos[1], 3), round(pos[2], 3), round(pos[3], 3), round(pos[4],
                                                                                                               3)
        response = "Under the {} modulation strategy,the optimal D0 is designed to be {},D1 is designed to be {},D2 is designed to be {},phi1 is designed to be {},phi2 is designed to be {}, the number of switches that achieve zero-voltage turn-on is {:.0f}. And the current stress performance is shown with the following figure.At rated power level, the peak-to-peak current is {:.2f}A. When load power PL = {}W, the peak-to-peak current is {:.2f}A.".format(modulation,D0,D1,D2,phi1,phi2,nZVS,ipp1,P_required,ipp)

    return response

def answer1(ipp,ipp1):
    response1 ="The conventional SPS realizes {:.2f}A peak-to-peak current at rated power, while showing {:.2f}A peak-to-peak current at 10% light load (100 W).The inductor current waveforms in steady state are shown below. ".format(ipp1,ipp)
    return response1
