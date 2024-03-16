def pinn(Uin,Uo,Prated,fsw,modulation):
    if modulation=="SPS":
        if Uin>0 and Uo>0 and Prated>0 and fsw>0:
            D0=0.5
            current_stress=1
            efficiency=0.9
        else:
            D0=1
            current_stress = 2
            efficiency=0.8
    return D0,current_stress,efficiency

