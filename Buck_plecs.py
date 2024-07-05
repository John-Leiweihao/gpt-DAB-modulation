import threading
import numpy as np
import xmlrpc.client
import os
import pandas as pd
import csv
import time
import logging
import os
import shutil
import func_timeout
from func_timeout import func_set_timeout
from tqdm import tqdm
import datetime
import time
import test_buck

import os

def startplecs(Vin,Vref,P,L_best,Co_best,fs):
    class PlecsThread(threading.Thread):

        opts = {'ModelVars': {'Tstop':0.4, 'Tdead':0e-9,'Vin': 200, 'Vref': 80,'P': 800, 'Ro': 800 ** 2 / 80}}

        def __init__(self, model_name,Vin,Vref,P,Lr,Cf,fs, num_req=1e4):
            super(PlecsThread, self).__init__()
            self.model_name = model_name
            self.num_req = num_req
            self.model_path = os.getcwd() + f"\\{self.model_name}.plecs"
            self.server = xmlrpc.client.Server('http://localhost:1080/RPC2"')
            self.Vin = Vin
            self.Vref = Vref
            self.P = P
            self.Lr = Lr
            self.Cf = Cf
            self.fs = fs

        def load(self):
            self.server.plecs.load(self.model_path)

        def close(self):
            self.server.plecs.close(self.model_name)

        @func_set_timeout(60 * 3)
        def run_sim(self, opts):
            self.load()
            self.server.plecs.simulate(self.model_name, opts)
            # self.close()

        def run(self):
            for i in tqdm(range(self.num_req)):
                self.loop(i,self.Vin,self.Vref,self.P,self.Lr,self.Cf,self.fs)
        def loop(self, i,Vin,Vref,P,Lr,Cf,fs):
            try:
                opts = {'ModelVars': {'Tstop': self.__class__.opts['ModelVars']['Tstop'],
                                    'Tdead': self.__class__.opts['ModelVars']['Tdead']}}
                opts['ModelVars']["Vin"]=Vin
                opts['ModelVars']["Vref"] = Vref
                opts['ModelVars']["P"] = P
                opts['ModelVars']["Lr"] = Lr
                opts['ModelVars']["Cf"] = Cf
                opts['ModelVars']["Ro"] = round(Vref ** 2 / P, 2)
                opts['ModelVars']["fs"] = fs
                opts['ModelVars']["Ts"] = 1/fs
                opts['ModelVars']["fc"] = fs
                self.run_sim(opts)
            except Exception as e:
                print(e)

    #num_req = 1
    start_time = datetime.datetime.now()
    #Vin=200
    #Vref=80
    #P=800
    #fs=5e4
    #L_best,Co_best,i_ripple_value,v_ripple_value,i_ripple_percentage,v_ripple_percentage ,iLdc,iL1,iL2,iL3,Vodc,Vo1,Vo2,Vo3,P_on,P_off,P_cond=test_buck.optimization(200,80,800,5e4,0.2,0.005)

    def startplecs(file_path):
        if platform.system() == 'Windows':
            os.startfile(file_path)
        else:
            opener = "open" if platform.system() == "Darwin" else "xdg-open"
            subprocess.call([opener, file_path])

    # 使用方法
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 构建相对路径
    plecs_file_path = os.path.join(current_dir, "Buck.plecs")
    startplecs(plecs_file_path)

    time.sleep(5)

    thread = PlecsThread(f"Buck",Vin,Vref,P,L_best,Co_best,fs,num_req=1)
    thread.start()
