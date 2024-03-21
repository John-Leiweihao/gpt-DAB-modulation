def pinn(Uin,Uo,Prated,fsw,modulation):
    D0 = None
    current_stress = None
    efficiency = None
    if modulation=="SPS":
        D0=0.5
        current_stress=1
        efficiency=0.9
    elif modulation=="EPS":
        D0=1
        current_stress = 2
        efficiency=0.8
    elif modulation=="DPS":
        D0=1
        current_stress = 2
        efficiency=0.8
    elif modulation=="TPS":
        D0=1
        current_stress = 2
        efficiency=0.8
    elif modulation=="Five-Degree":
        D0=1
        current_stress = 2
        efficiency=0.8
    return D0,current_stress,efficiency

