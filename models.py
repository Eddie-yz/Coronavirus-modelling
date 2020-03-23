import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

class SSEIR(nn.Module):
    '''
    Model that only has constraints on beta, sigma and gamma,\n
    but assumes that they all remain unchanged through time. '''
    def __init__(self, beta_range, sigma_range, gamma_range):
        super(SSEIR, self).__init__()
        self.betaL = beta_range[0]
        self.betaH = beta_range[1]
        self.sigmaL = sigma_range[0]
        self.sigmaH = sigma_range[1]
        self.gammaL = gamma_range[0]
        self.gammaH = gamma_range[1]
        self.params = torch.tensor([0.17, 0.2, 0.07], dtype=torch.float, requires_grad=True)
    
    def forward(self, X):
        S_0 = X[0]
        E_0 = X[1]
        I_0 = X[2]
        R_0 = X[3]
        S_1 = X[4]
        E_1 = X[5]
        I_1 = X[6]
        R_1 = X[7]
        N = S_0 + E_0 + I_0 + R_0
        preE_1 = (1 - self.params[1])*E_0 + self.params[0]*I_0*S_0/N
    
        preS = (1 - self.params[0]*I_1/N) * S_1
        preE = (1 - self.params[1])*E_1 + self.params[0]*I_1*S_1/N
        preI = (1 - self.params[2])*I_1 + self.params[1]*preE_1
        preR = R_1 + self.params[2]*I_1
        return (preS, preE, preI, preR)



class DSEIR(nn.Module):
    '''
    Model that considers the change of beta and gamma through time.\n
    Beta and gamma are characterized by piecewise linear functions.
    '''
    def __init__(self, sigma):
        super(DSEIR, self).__init__()
        self.sigma = sigma
        self.params = torch.tensor([0.07, 0.05, 0.02, 0.005, 0.0001, 0.001, 0.02, 0.05], dtype=torch.float, requires_grad=True)
    
    def computeBeta(self, t):
        if t <= 4:
            return torch.max(torch.tensor([0], dtype=torch.float),self.params[0])
        elif t <= 15:
            return torch.max(torch.tensor([0], dtype=torch.float),self.params[0]+(self.params[1]-self.params[0])*(t-4)/(15-4))
        elif t <= 30:
            return torch.max(torch.tensor([0], dtype=torch.float),self.params[1]+(self.params[2]-self.params[1])*(t-15)/(30-15))
        else:
            return torch.max(torch.tensor([0], dtype=torch.float),self.params[2]+(self.params[3]-self.params[2])*(t-30)/(50-30))
    
    def computeGamma(self, t):
        if t <= 4:
            return torch.max(torch.tensor([0], dtype=torch.float),self.params[4])
        elif t <= 15:
            return torch.max(torch.tensor([0], dtype=torch.float),self.params[4]+(self.params[5]-self.params[4])*(t-4)/(15-4))
        elif t <= 30:
            return torch.max(torch.tensor([0], dtype=torch.float),self.params[5]+(self.params[6]-self.params[5])*(t-15)/(30-15))
        else:
            return torch.max(torch.tensor([0], dtype=torch.float),self.params[6]+(self.params[7]-self.params[6])*(t-30)/(50-30))
        
    def forward(self, X):
        S_0 = X[0]
        E_0 = X[1]
        I_0 = X[2]
        R_0 = X[3]
        S_1 = X[4]
        E_1 = X[5]
        I_1 = X[6]
        R_1 = X[7]
        t = X[8]
        N = S_0 + E_0 + I_0 + R_0
        preE_1 = (1 - self.sigma)*E_0 + self.computeBeta(t)*I_0*S_0/N
    
        preS = (1 - self.computeBeta(t+1)*I_1/N) * S_1
        preE = (1 - self.sigma)*E_1 + self.computeBeta(t+1)*I_1*S_1/N
        preI = (1 - self.computeGamma(t+1))*I_1 + self.sigma*preE_1
        preR = R_1 + self.computeGamma(t+1)*I_1
        return (preS, preE, preI, preR)


class PDSEIR(nn.Module):
    '''
    Model that built upon the base of DSEIR, but considers that \n
    the exposed group is as well infectious.
    '''
    def __init__(self, sigma):
        super(PDSEIR, self).__init__()
        self.sigma = sigma
        self.params = torch.tensor([0.07, 0.05, 0.02, 0.005, 0.0001, 0.001, 0.02, 0.05], dtype=torch.float, requires_grad=True)
    
    def computeBeta(self, t):
        if t <= 4:
            return torch.max(torch.tensor([0], dtype=torch.float),self.params[0])
        elif t <= 15:
            return torch.max(torch.tensor([0], dtype=torch.float),self.params[0]+(self.params[1]-self.params[0])*(t-4)/(15-4))
        elif t <= 30:
            return torch.max(torch.tensor([0], dtype=torch.float),self.params[1]+(self.params[2]-self.params[1])*(t-15)/(30-15))
        else:
            return torch.max(torch.tensor([0], dtype=torch.float),self.params[2]+(self.params[3]-self.params[3])*(t-30)/(50-30))
    
    def computeGamma(self, t):
        if t <= 4:
            return torch.max(torch.tensor([0], dtype=torch.float),self.params[4])
        elif t <= 15:
            return torch.max(torch.tensor([0], dtype=torch.float),self.params[4]+(self.params[5]-self.params[4])*(t-4)/(15-4))
        elif t <= 30:
            return torch.max(torch.tensor([0], dtype=torch.float),self.params[5]+(self.params[6]-self.params[5])*(t-15)/(30-15))
        else:
            return torch.max(torch.tensor([0], dtype=torch.float),self.params[6]+(self.params[7]-self.params[6])*(t-30)/(50-30))
        
    def forward(self, X):
        S_0 = X[0]
        E_0 = X[1]
        I_0 = X[2]
        R_0 = X[3]
        S_1 = X[4]
        E_1 = X[5]
        I_1 = X[6]
        R_1 = X[7]
        t = X[8]
        N = S_0 + E_0 + I_0 + R_0
        preE_1 = (1 - self.sigma)*E_0 + self.computeBeta(t)*(I_0+E_0)*S_0/N
    
        preS = (1 - self.computeBeta(t+1)*I_1/N) * S_1
        preE = (1 - self.sigma)*E_1 + self.computeBeta(t+1)*(I_1+E_1)*S_1/N
        preI = (1 - self.computeGamma(t+1))*I_1 + self.sigma*preE_1
        preR = R_1 + self.computeGamma(t+1)*I_1
        return (preS, preE, preI, preR)