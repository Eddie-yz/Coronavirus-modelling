#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from data.fetchdata import getProvinceData, getCountryData


# In[38]:


#d, con, rec = getProvinceData("湖北省")
d, con, rec = getCountryData("KR")


# In[39]:


plt.plot(rec, label='cured')
plt.plot(con, label='confirmed')
plt.plot(con-rec, label='being sick')
plt.legend(loc='upper left')


# In[12]:


d


# In[86]:


def dataPrepare(total_population, confirmed, recovered, exposed_ratio):
    days = len(confirmed)
    X = []
    y = []
    for i in range(days-1):
        I = confirmed[i]
        R = recovered[i]
        E = I * exposed_ratio
        S = total_population - I - R - E
        X.append([S, E, I, R])
        y.append(confirmed[i+1])
    return np.array(X), np.array(y).reshape((-1,1))


# In[95]:


province_population = 57000000
X, y = dataPrepare(province_population, con, rec, 4)


# In[97]:


X_train = X
y_train = y


# - $\sigma$: $\frac{1}{incubation time}$ ~ [$\frac{1}{14}$, 1]
# - $\gamma$: $\frac{1}{recovery time}$ ~ [$\frac{1}{17+4}$, $\frac{1}{17-4}$]
# - $\beta$: $R_0 = \frac{\beta}{\gamma}$, $R_0$ of COVID-19 is around 1.4~3.8, so $\beta$ ~ [0.067, 0.292]

# In[102]:


class SSEIR(nn.Module):
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
        S = X[0]
        E = X[1]
        I = X[2]
        R = X[3]
        N = S + E + I + R
        preS = (1 - self.params[0]*I/N) * S
        preE = (1 - self.params[1])*E + self.params[0]*I*S/N
        preI = (1 - self.params[2])*I + self.params[1]*E
        preR = R + self.params[2]*I
        return (preS, preE, preI, preR)


# In[137]:


sseir = SSEIR(beta_range=[0.066, 0.292], sigma_range=[1.0/14, 1.0], gamma_range=[1.0/21, 1.0/13])
sseir = sseir.float()


# In[138]:


criterion = nn.MSELoss()
optimizer = torch.optim.Adam([sseir.params], lr=0.0005)


# In[139]:


num_epoch = 200
lam0 = 0.5
lam1 = 1
lam2 = 1000
lam3 = 1000

for epoch in range(num_epoch):
    ids = np.random.permutation(X_train.shape[0])
    epoch_loss = 0.0
    for sample_id in ids:
        currday = torch.from_numpy(X_train[sample_id])
        nextInfected = torch.from_numpy(y_train[sample_id])
        
        optimizer.zero_grad()
        out = sseir(currday.float())
        preInfected = out[2]
        loss = criterion(preInfected, nextInfected.float())
        reg = lam1*(F.relu(sseir.params[0]-sseir.betaH) + F.relu(sseir.betaL-sseir.params[0]))              +lam2*(F.relu(sseir.params[1]-sseir.sigmaH) + F.relu(sseir.sigmaL-sseir.params[1]))              +lam3*(F.relu(sseir.params[2]-sseir.gammaH) + F.relu(sseir.gammaL-sseir.params[2]))
        loss = lam0*loss + (1-lam0)*reg
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()
    print (epoch_loss/len(ids))
    

