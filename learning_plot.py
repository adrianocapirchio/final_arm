# -*- coding: utf-8 -*-
"""
Created on Sat Nov 24 17:08:48 2018

@author: Alex
"""

import os
import numpy as np
import matplotlib.pyplot as plt




multinet = True

actETA =  6.0 * 10 ** ( -1)# 8.0 * 10 ** ( -1)
critETA = 6.0 * 10 ** ( -4) #8.0 * 10 ** ( -4)
cbETA   = 6.0 * 10 ** ( -1) #8.0 * 10 ** ( -1)
LMA = 1000 #600
proprio_units = 101**2 #101


maxStep = 250
maxTrial = 7
maxEpoch = 2500
seed =0
maxSeed = 10#20
avg_stats = 20


mydir = os.getcwd
os.chdir("C:\Users/Alex/Desktop/final_arm/data/")

if multinet == True:
    os.chdir(os.curdir + "/multinet/")
else:
    os.chdir(os.curdir + "/uninet/")
    
full_sistem_learning = np.zeros(maxEpoch/avg_stats)

    
os.chdir(os.curdir + "/cerebellum/actETA=%s_critETA=%s_cbETA=%s_proprioUnits=%s_LMA=%s/" % (actETA,critETA,cbETA,proprio_units,LMA))


for seed in xrange(maxSeed):
    a = np.load("finalTrainingAccuracy_seed=%s.npy" %(seed))[:-3]
    full_sistem_learning = np.vstack([full_sistem_learning,a]) 
    
full_sistem_learning = full_sistem_learning[1:]   
full_sistem_avg_learning = np.mean(full_sistem_learning, axis =0)
full_sistem_std_learning = np.std(full_sistem_learning, axis =0)







mydir = os.getcwd
os.chdir("C:\Users/Alex/Desktop/final_arm/data/")

if multinet == True:
    os.chdir(os.curdir + "/multinet/")
else:
    os.chdir(os.curdir + "/uninet/")

act_crit_learning = np.zeros(maxEpoch/avg_stats)

os.chdir(os.curdir + "/onlyGanglia/actETA=%s_critETA=%s_proprioUnits=%s_LMA=%s/" % (actETA,critETA,proprio_units,LMA))



for seed in xrange(maxSeed):
    a = np.load("finalTrainingAccuracy_seed=%s.npy" %(seed))[:-3]
    act_crit_sistem_learning = np.vstack([full_sistem_learning,a]) 
    
act_crit_sistem_learning = act_crit_sistem_learning[1:]   
act_crit_sistem_avg_learning = np.mean(act_crit_sistem_learning, axis =0)
act_crit_sistem_std_learning = np.std(act_crit_sistem_learning, axis =0)







 


fig1   = plt.figure("Workspace", figsize=(9,8))
ax1 = fig1.add_subplot(111)
ax1.errorbar(np.linspace(0,maxEpoch/avg_stats,maxEpoch/avg_stats), full_sistem_avg_learning , yerr = full_sistem_std_learning, marker= '^' , mec ='orange', mfc='white' , ecolor ='orange', )
ax1.errorbar(np.linspace(0,maxEpoch/avg_stats,maxEpoch/avg_stats), act_crit_sistem_avg_learning , yerr = act_crit_sistem_std_learning, marker= '^' , mec ='blue', mfc='white' , ecolor ='blue')
