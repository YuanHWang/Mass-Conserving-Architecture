import argparse
import numpy as np
import pandas as pd
import os
import math
from math import tanh
import spotpy
from typing import Callable
from pyDOE import * #hipercube
import scipy.stats
import random as rand
from spotpy.parameter import Uniform
from spotpy.objectivefunctions import kge

# Parser
parser = argparse.ArgumentParser()

parser.add_argument('--Trial',
                        type=int,
                        default=1,
                        help="default is no dropout")

parser.add_argument('--CaseName',
                        type=str,
                        default='GR4J-Test',
                        help="")

# Argparser
cfg = vars(parser.parse_args())
CaseName = cfg["CaseName"]
Trial_no = cfg["Trial"]

# File Dir info
directory = CaseName
parent_dir = "/Users/ywv/Desktop/Paper2-Revision/GR4J-Yuan"
path = os.path.join(parent_dir, directory)
isExist = os.path.exists(path)

if not isExist:
    os.mkdir(path)
    print("The new directory is created!")

#if Trial_no == 0:
#   np.random.seed(0)

def s_curves1(t, x4):
    """
        Unit hydrograph ordinates for UH1 derived from S-curves.
    """

    if t <= 0:
        xx = 0
    elif t < x4:
        xx = (t/x4)**2.5
    else: # t >= x4
        xx = 1

    return xx

def s_curves2(t, x4):
    """
        Unit hydrograph ordinates for UH2 derived from S-curves.
    """

    if t <= 0:
        return 0
    elif t < x4:
        return 0.5*(t/x4)**2.5
    elif t < 2*x4:
        return 1 - 0.5*(2 - t/x4)**2.5
    else: # t >= x4
        return 1

class spotpy_setup():
    def __init__(self,dim=4):
        self.dim = dim
        self.params = [spotpy.parameter.Uniform('p1',low=1, high=5000, optguess=1500.0),
                       spotpy.parameter.Uniform('p2',low=-10, high=10, optguess=-2.0),
                       spotpy.parameter.Uniform('p3',low=1, high=1500, optguess=200.0),
                       spotpy.parameter.Uniform('p4',low=0.501, high=4.5, optguess=1.6),
                       ]
        self.model = model()
        self.sim = None
        self.obs = None
        
    def parameters(self):
        return spotpy.parameter.generate(self.params)
    
    def simulation(self, vector):
                       
        sim, obs = self.model.GR4J_local(self.model.df, p1 = vector[0], p2 = vector[1], p3 = vector[2], p4 = vector[3])
        self.sim = sim
        self.obs = obs
        #self.sim, self.obs = self.model.GR4J_local(df, 0.5, 0.5, 350, 0, 90, 1.7)         
        return self.sim
    
    def evaluation(self):
        observations = self.obs
        return observations        

    def objectivefunction(self, simulation, evaluation):

        like = spotpy.objectivefunctions.kge(self.evaluation(),simulation) #rmse
        
        return 1-like

class model():
    def __init__(self):

        self.load_data()
        
    def load_data(self):

        # Input/Output Dataset
        F_data = pd.read_csv('LeafRiverDaily_43YR.txt', header=None, delimiter=r"\s+")
        F_data = F_data.rename(columns={0: 'P', 1: 'PET', 2: 'Q'})
        #PP = F_data['P']
        #PET = F_data['PET']
        #qobs = F_data['Q']

        # Flag
        flag = pd.read_csv('LeafRiverDaily_43YR_Flag.txt', header=None, delimiter=r"\s+")
        Flag = flag.rename(columns={0: 'Flag'})
        FG = Flag['Flag'].to_numpy()
        self.df = F_data
        self.FG = FG
        return
        
    def GR4J_local(self, df_i, p1, p2, p3, p4):

        states = {}
        precip = df_i['P']
        potential_evap = df_i['PET']
        qobs = df_i['Q']

        X1 = p1#.values
        X2 = p2#.values
        X3 = p3#.values
        X4 = p4#.values

        nUH1 = int(math.ceil(X4))
        nUH2 = int(math.ceil(2.0*X4))

        uh1_ordinates = [0] * nUH1
        uh2_ordinates = [0] * nUH2

        UH1 = states.get('UH1', [0] * nUH1)
        UH2 = states.get('UH2', [0] * nUH2)

        for t in range(1, nUH1 + 1):
            uh1_ordinates[t - 1] = s_curves1(t, X4) - s_curves1(t-1, X4)

        for t in range(1, nUH2 + 1):
            uh2_ordinates[t - 1] = s_curves2(t, X4) - s_curves2(t-1, X4)

        production_store = states.get('production_store', 0) # S
        routing_store = states.get('routing_store', 0) # R

        qsim = []
        for P, E in zip(precip, potential_evap):

            if P > E:
               net_evap = 0
               scaled_net_precip = (P - E)/X1
               if scaled_net_precip > 13:
                  scaled_net_precip = 13.
               tanh_scaled_net_precip = tanh(scaled_net_precip)
               reservoir_production = (X1 * (1 - (production_store/X1)**2) * tanh_scaled_net_precip) / (1 + production_store/X1 * tanh_scaled_net_precip)

               routing_pattern = P-E-reservoir_production
            else:
               scaled_net_evap = (E - P)/X1
               if scaled_net_evap > 13:
                  scaled_net_evap = 13.
               tanh_scaled_net_evap = tanh(scaled_net_evap)

               ps_div_x1 = (2 - production_store/X1) * tanh_scaled_net_evap
               net_evap = production_store * (ps_div_x1) / \
                       (1 + (1 - production_store/X1) * tanh_scaled_net_evap)

               reservoir_production = 0
               routing_pattern = 0

            production_store = production_store - net_evap + reservoir_production

            percolation = production_store / (1 + (production_store/2.25/X1)**4)**0.25

            routing_pattern = routing_pattern + (production_store-percolation)
            production_store = percolation


            for i in range(0, len(UH1) - 1):
                UH1[i] = UH1[i+1] + uh1_ordinates[i]*routing_pattern
            UH1[-1] = uh1_ordinates[-1] * routing_pattern

            for j in range(0, len(UH2) - 1):
                UH2[j] = UH2[j+1] + uh2_ordinates[j]*routing_pattern
            UH2[-1] = uh2_ordinates[-1] * routing_pattern

            groundwater_exchange = X2 * (routing_store / X3)**3.5
            routing_store = max(0, routing_store + UH1[0] * 0.9 + groundwater_exchange)

            R2 = routing_store / (1 + (routing_store / X3)**4)**0.25
            QR = routing_store - R2
            routing_store = R2
            QD = max(0, UH2[0]*0.1+groundwater_exchange)
            Q = QR + QD

            qsim.append(Q)

        QQsim = np.asarray(qsim)

        QQsim_out = QQsim[self.FG==-1]
        qobs_out = qobs[self.FG==-1]
        
        return QQsim_out, qobs_out

def GR4J_sce_ua():

    global spotpy_setup
    spotpy_setup = spotpy_setup()  
    n_sample = 5000
    sampler = spotpy.algorithms.sceua(spotpy_setup, dbname="test", dbformat='csv')
    sampler.sample(n_sample, ngs=10) 
    results=sampler.getdata()
    best = spotpy.analyser.get_best_parameterset(results, maximize=False)
        
    return results, best

# Output 
rep=5000
outfile = CaseName + '/GR4J-Test_' + str(Trial_no) 
sampler = spotpy.algorithms.sceua(spotpy_setup(), dbname=outfile, dbformat='csv')
sampler.sample(rep, ngs=7, kstop=3, peps=0.1, pcento=0.1)
results=sampler.getdata()
best = spotpy.analyser.get_best_parameterset(results) # the best result

print(best)
parameters = np.zeros((1,4))
parameters = pd.DataFrame(parameters, columns=['x1', 'x2', 'x3', 'x4'])
parameters.at[0,'x1'] = best[0][0] #best[3]
parameters.at[0,'x2'] = best[0][1] #best[4]
parameters.at[0,'x3'] = best[0][2] #best[5]
parameters.at[0,'x4'] = best[0][3] #best[6]

outname = CaseName + '/GR4J-Test_BestPar_' + str(Trial_no) + '.csv'
parameters.to_csv(outname)