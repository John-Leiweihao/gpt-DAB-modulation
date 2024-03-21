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


# Define global variables for PINN
Tslen = 250
fs = 5e4
Ts = 1/fs
dt = Ts/Tslen
Tsim = Ts*10


