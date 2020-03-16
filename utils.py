import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from data.fetchdata import getProvinceData, getCountryData


def dataPrepare(total_population, confirmed, recovered, exposed_ratio):
    days = len(confirmed)
    X = []
    y = []
    for i in range(days-2):
        I_0 = confirmed[i] - recovered[i]
        R_0 = recovered[i]
        E_0 = I_0 * exposed_ratio
        S_0 = total_population - I_0 - R_0 - E_0
        I_1 = confirmed[i+1] - recovered[i+1]
        R_1 = recovered[i+1]
        E_1 = I_1 * exposed_ratio
        S_1 = total_population - I_1 - R_1 - E_1
        
        X.append([S_0, E_0, I_0, R_0, S_1, E_1, I_1, R_1])
        y.append(confirmed[i+2] - recovered[i+2])
    return np.array(X), np.array(y).reshape((-1,1))

def dataPrepareWithTime(total_population, confirmed, recovered, exposed_ratio):
    days = len(confirmed)
    X = []
    y = []
    for i in range(days-2):
        I_0 = confirmed[i] - recovered[i]
        R_0 = recovered[i]
        E_0 = I_0 * exposed_ratio
        S_0 = total_population - I_0 - R_0 - E_0
        I_1 = confirmed[i+1] - recovered[i+1]
        R_1 = recovered[i+1]
        E_1 = I_1 * exposed_ratio
        S_1 = total_population - I_1 - R_1 - E_1
        
        X.append([S_0, E_0, I_0, R_0, S_1, E_1, I_1, R_1, i])
        y.append(confirmed[i+2] - recovered[i+2])
    return np.array(X), np.array(y).reshape((-1,1))

def check_and_plot(X, y, date, params):
    days = X.shape[0]
    predict_infected = []
    for day in range(days):
        data = X[day]
        S_1 = data[4]
        E_1 = data[5]
        I_1 = data[6]
        R_1 = data[7]
        N = S_1 + E_1 + I_1 + R_1
        preS = (1 - params[0]*I_1/N) * S_1
        preE = (1 - params[1])*E_1 + params[0]*I_1*S_1/N
        preI = (1 - params[2])*I_1 + params[1]*E_1
        preR = R_1 + params[2]*I_1
        predict_infected.append(preI)
    plt.figure(figsize=(12,6))
    plt.plot(predict_infected, label="predicted")
    plt.plot(y, label="ground truth")
    plt.legend(loc='upper left')
    date = [d[5:] for d in date]
    plt.xticks(np.arange(days), date)
    plt.show()
    mape = np.mean(np.abs(np.array(predict_infected) - y.flatten()))
    return predict_infected, mape

def predict_and_plot(X, y, date, start_predict, params):
    days = X.shape[0]
    predict_infected = []
    for day in range(start_predict):
        data = X[day]
        S_1 = data[4]
        E_1 = data[5]
        I_1 = data[6]
        R_1 = data[7]
        N = S_1 + E_1 + I_1 + R_1
        preS = (1 - params[0]*I_1/N) * S_1
        preE = (1 - params[1])*E_1 + params[0]*I_1*S_1/N
        preI = (1 - params[2])*I_1 + params[1]*E_1
        preR = R_1 + params[2]*I_1
        predict_infected.append(preI)
        
    S_1 = X[start_predict][4]
    E_1 = X[start_predict][5]
    I_1 = X[start_predict][6]
    R_1 = X[start_predict][7]
    for day in range(start_predict, days):
        N = S_1 + E_1 + I_1 + R_1
        preS = (1 - params[0]*I_1/N) * S_1
        preE = (1 - params[1])*E_1 + params[0]*I_1*S_1/N
        preI = (1 - params[2])*I_1 + params[1]*E_1
        preR = R_1 + params[2]*I_1
        predict_infected.append(preI)
        S_1 = preS
        E_1 = preE
        I_1 = preI
        R_1 = preR
    plt.figure(figsize=(10,6))
    plt.plot(predict_infected, label="predicted")
    plt.plot(y, label="ground truth")
    plt.legend(loc='upper left')
    plt.xlabel("Days")
    plt.ylabel("Still Infected")
    date = [d[5:] for d in date]
    plt.xticks(np.arange(0, days, 5))
    plt.show()
    predict_error = (np.array(predict_infected) - y.flatten())[start_predict:]
    mape = np.mean(np.abs(predict_error))
    return predict_infected, mape

def computeBeta(params, t):
    if t <= 4:
        return max(0,params[0])
    elif t <= 15:
        return max(0,params[0]+(params[1]-params[0])*(t-4)/(15-4))
    elif t <= 30:
        return max(0,params[1]+(params[2]-params[1])*(t-15)/(30-15))
    else:
        return max(0,params[2]+(params[3]-params[2])*(t-30)/(50-30))

def computeGamma(params, t):
    if t <= 4:
        return max(0,params[4])
    elif t <= 15:
        return max(0,params[4]+(params[5]-params[4])*(t-4)/(15-4))
    elif t <= 30:
        return max(0,params[5]+(params[6]-params[5])*(t-15)/(30-15))
    else:
        return max(0,params[6]+(params[7]-params[6])*(t-30)/(50-30))
    
def check_and_plot_dyn(X, y, date, params, sigma=0.02531358, gamma=0.07680123):

    days = X.shape[0]
    predict_infected = []
    for day in range(days):
        data = X[day]
        S_1 = data[4]
        E_1 = data[5]
        I_1 = data[6]
        R_1 = data[7]
        t = day
        N = S_1 + E_1 + I_1 + R_1
        preS = (1 - computeBeta(params, t+1)*I_1/N) * S_1
        preE = (1 - sigma)*E_1 + computeBeta(params, t+1)*I_1*S_1/N
        preI = (1 - computeGamma(params,t+1))*I_1 + sigma*E_1
        preR = R_1 + computeGamma(params,t+1)*I_1
        predict_infected.append(preI)
    plt.figure(figsize=(12,6))
    plt.plot(predict_infected, label="predicted")
    plt.plot(y, label="ground truth")
    plt.legend(loc='upper left')
    date = [d[5:] for d in date]
    plt.xticks(np.arange(days))
    # plt.xticks(np.arange(days), date)
    plt.show()
    mae = np.mean(np.abs(np.array(predict_infected) - y.flatten()))
    return predict_infected, mae

def predict_and_plot_dyn(X, y, date, start_predict, params, sigma=0.02531358, gamma=0.07680123): 
    days = X.shape[0]
    predict_infected = []
    for day in range(start_predict):
        data = X[day]
        S_1 = data[4]
        E_1 = data[5]
        I_1 = data[6]
        R_1 = data[7]
        N = S_1 + E_1 + I_1 + R_1
        t = day
        preS = (1 - computeBeta(params, t+1)*I_1/N) * S_1
        preE = (1 - sigma)*E_1 + computeBeta(params, t+1)*I_1*S_1/N
        preI = (1 - computeGamma(params,t+1))*I_1 + sigma*E_1
        preR = R_1 + computeGamma(params,t+1)*I_1
        predict_infected.append(preI)
        
    S_1 = X[start_predict][4]
    E_1 = X[start_predict][5]
    I_1 = X[start_predict][6]
    R_1 = X[start_predict][7]
    for day in range(start_predict, days):
        N = S_1 + E_1 + I_1 + R_1
        t = day
        preS = (1 - computeBeta(params, t+1)*I_1/N) * S_1
        preE = (1 - sigma)*E_1 + computeBeta(params, t+1)*I_1*S_1/N
        preI = (1 - computeGamma(params,t+1))*I_1 + sigma*E_1
        preR = R_1 + computeGamma(params,t+1)*I_1
        predict_infected.append(preI)
        S_1 = preS
        E_1 = preE
        I_1 = preI
        R_1 = preR
    plt.figure(figsize=(10,6))
    plt.plot(predict_infected, label="predicted")
    plt.plot(y, label="ground truth")
    plt.legend(loc='upper left')
    plt.xlabel("Days")
    plt.ylabel("Still Infected")
    date = [d[5:] for d in date]
    plt.xticks(np.arange(0, days, 5))
    plt.show()
    predict_error = (np.array(predict_infected) - y.flatten())[start_predict:]
    mae = np.mean(np.abs(predict_error))
    return predict_infected, mae

def predict_and_plot_dyn_for_other_region(X, y, days, start_predict, params, sigma=0.02531358, gamma=0.07680123): 
    predict_infected = []
    for day in range(start_predict):
        data = X[day]
        S_1 = data[4]
        E_1 = data[5]
        I_1 = data[6]
        R_1 = data[7]
        N = S_1 + E_1 + I_1 + R_1
        t = day
        preS = (1 - computeBeta(params, t+1)*I_1/N) * S_1
        preE = (1 - sigma)*E_1 + computeBeta(params, t+1)*I_1*S_1/N
        preI = (1 - computeGamma(params,t+1))*I_1 + sigma*E_1
        preR = R_1 + computeGamma(params,t+1)*I_1
        predict_infected.append(preI)
        
    S_1 = X[start_predict][4]
    E_1 = X[start_predict][5]
    I_1 = X[start_predict][6]
    R_1 = X[start_predict][7]
    for day in range(start_predict, days):
        N = S_1 + E_1 + I_1 + R_1
        t = day
        preS = (1 - computeBeta(params, t+1)*I_1/N) * S_1
        preE = (1 - sigma)*E_1 + computeBeta(params, t+1)*I_1*S_1/N
        preI = (1 - computeGamma(params,t+1))*I_1 + sigma*E_1
        preR = R_1 + computeGamma(params,t+1)*I_1
        predict_infected.append(preI)
        S_1 = preS
        E_1 = preE
        I_1 = preI
        R_1 = preR
    plt.figure(figsize=(10,6))
    plt.plot(predict_infected, label="predicted number using parameters from Zhejiang")
    plt.plot(y, label="ground truth number in the Italy")
    plt.legend(loc='upper left')
    plt.xlabel("Days")
    plt.ylabel("Still Infected")
    plt.xticks(np.arange(0, days, 5))
    plt.show()
    return predict_infected