# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 12:01:37 2018

@author: Alex
"""

import os
import numpy as np
import copy as copy
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import utilities as utils
import scipy.stats as stats

multinet = True
cerebellum = False

armdt = 0.01

maxStep = 250
maxTrial = 7
maxEpoch = 60
seed =0
maxSeed = 10#20

actETA =  6.0 * 10 ** ( -1)# 8.0 * 10 ** ( -1)
critETA = 6.0 * 10 ** ( -4) #8.0 * 10 ** ( -4)
cbETA   = 6.0 * 10 ** ( -1) #8.0 * 10 ** ( -1)
LMA = 1000 #600

#noise = 2.25

proprio_units = 101**2 #101

n = 20
startAtaxia = maxEpoch - 40
#ataxia mag = 15 0.001


target_normal_lin_idx = 4.13
target_ataxia_lin_idx = 24.09

target_normal_asy_idx = 1.25
target_ataxia_asy_idx = 1.92

target_normal_smo_idx = 5.66
target_ataxia_smo_idx = 8.28




mydir = os.getcwd
os.chdir("C:\Users/Alex/Desktop/final_arm/data/")

if multinet == True:
    os.chdir(os.curdir + "/multinet/")
else:
    os.chdir(os.curdir + "/uninet/")
    
if cerebellum == True:
    os.chdir(os.curdir + "/cerebellum/actETA=%s_critETA=%s_cbETA=%s_proprioUnits=%s_LMA=%s/" % (actETA,critETA,cbETA,proprio_units,LMA))
else:
    os.chdir(os.curdir + "/onlyGanglia/actETA=%s_critETA=%s_proprioUnits=%s_LMA=%s/" % (actETA,critETA,proprio_units,LMA))
    






#allGoalPosition = np.zeros([2, maxTrial, maxEpoch, maxSeed])
#allGoalAngles = np.zeros([2, maxStep, maxTrial, maxEpoch, maxSeed])


#allArmAngles = np.zeros([2, maxStep, maxTrial, maxEpoch, maxSeed])
allTrajectories = np.zeros([2, maxStep, maxTrial, maxEpoch, maxSeed])
#allGangliaAngles = np.zeros([2, maxStep, maxTrial, maxEpoch, maxSeed])

#if cerebellum == True:
#    allCerebAngles = np.zeros([2, maxStep, maxTrial, maxEpoch, maxSeed])



#allVelocity = np.zeros([maxStep, maxTrial, maxEpoch, maxSeed])
#allAccelleration = np.zeros([maxStep, maxTrial, maxEpoch, maxSeed])
allJerk = np.zeros([maxStep, maxTrial, maxEpoch, maxSeed])

allLinIdx = np.zeros([maxTrial,maxEpoch,maxSeed])
allAsyIdx = np.zeros([maxTrial,maxEpoch,maxSeed])
allSmoIdx = np.zeros([maxTrial,maxEpoch,maxSeed])



for seed in xrange (0,maxSeed):
    
    allTrajectories[:,:,:,:,seed]              = np.load("trajectoriesTestData_seed=%s.npy" % (seed))
  #  allArmAngles[:,:,:,:,seed]                 = np.load("gameTrialArmAngles_seed=%s.npy" % (seed))
  #  allGangliaAngles[:,:,:,:,seed]             = np.load("gameTrialGangliaAngles_seed=%s.npy" % (seed))

   # if cerebellum == True:
   #     allCerebAngles[:,:,:,:,seed]           = np.load("gameTrialCerebAngles_seed=%s.npy" %(seed)) 

    #allGoalPosition[:,:,:,seed]              = np.load("goalPositionHistory_seed=%s.npy" %(seed))
    #allGoalAngles[:,:,:,:,seed]                = np.load("goalAnglesHistory_seed=%s.npy" %(seed))
 #   allVelocity[:,:,:,seed]                  = np.load("gameVelocity_seed=%s.npy" %(seed) )
 #   allAccelleration[:,:,:,seed]             = np.load("gameAccelleration_seed=%s.npy" %(seed))
 #   allJerk[:,:,:,seed]                   = np.load("gameJerk_seed=%s.npy" %(seed))



linearityIndex = np.zeros(1)
asimmetryIndex = np.zeros(1)
smoothnessIndex = np.zeros(1)

"""
reach1LinIdx = np.zeros([n,maxSeed])
reach1AsyIdx = np.zeros([n,maxSeed])
reach1SmoIdx = np.zeros([n,maxSeed])


normalAVGLinIdx = np.zeros(n)
ataxiaAVGLinIdx = np.zeros(n)

normalAVGAsyIdx = np.zeros(n)
ataxiaAVGAsyIdx = np.zeros(n)

normalAVGSmoIdx = np.zeros(n)
ataxiaAVGSmoIdx = np.zeros(n)
"""

for seed in xrange(maxSeed):
    
    for epoch in xrange(maxEpoch):
        
        print seed ,epoch
        
        for trial in xrange(1 ,maxTrial):
            
            trialTraj = allTrajectories[:,:,trial,epoch,seed].copy()
            
            
            if trialTraj[0,:].any() != 0:
            
                trimmedTraj = utils.trimTraj(trialTraj)
                
                minDistance = utils.distance(trimmedTraj[:,0], trimmedTraj[:,len(trimmedTraj[0,:]) -1])
                trajLen = utils.trajLen(trimmedTraj)
                
                trialTangVel = np.zeros(len(trimmedTraj[0,:]))
                
                
                for step in xrange(len(trimmedTraj[0,:])):
                    
                    if step > 0:
                        
                        trialTangVel[step] = utils.distance(trimmedTraj[:,step],trimmedTraj[:,step -1]) / armdt





                trialdXdY = np.zeros([ 2, len(trimmedTraj[0,:]) ])
                
                trialdXdY[0,:] = np.ediff1d(trimmedTraj[0,:], to_begin = np.array([0]))  / armdt
                trialdXdY[1,:] = np.ediff1d(trimmedTraj[1,:], to_begin = np.array([0]))  / armdt
                
               
                trialddXddY = np.zeros([ 2, len(trimmedTraj[0,:]) ])
                
                trialddXddY[0,:] = np.ediff1d(trialdXdY[0,:], to_begin = np.array([0]))  / armdt
                trialddXddY[1,:] = np.ediff1d(trialdXdY[1,:], to_begin = np.array([0]))  / armdt
                           
                
                           
                           
                trialdddXdddY = np.zeros([ 2, len(trimmedTraj[0,:]) ])
                
                trialdddXdddY[0,:] = np.ediff1d(trialddXddY[0,:], to_begin = np.array([0]))  / armdt
                trialdddXdddY[1,:] = np.ediff1d(trialddXddY[1,:], to_begin = np.array([0]))  / armdt
                        
                    
                trialJerk = np.mean(trialdddXdddY)
                
                
                
    
    
                linearityIndex = (utils.linearityIndex(trajLen, minDistance) - 1.) * 100
                smoothnessIndex = utils.smoothnessIndex(trialdddXdddY, trajLen, armdt)
                asimmetryIndex = utils.asimmetryIndex(trialTangVel)
                
                
                allLinIdx[trial,epoch,seed] = copy.deepcopy(linearityIndex)
                allAsyIdx[trial,epoch,seed] = copy.deepcopy(asimmetryIndex)
                allSmoIdx[trial,epoch,seed] = copy.deepcopy(smoothnessIndex)





"GET STAT DATA FOR ANALISYS" 
 
"REACH 1"         
statNormalReach1LinIdx = allLinIdx[1, startAtaxia -n : startAtaxia , :]  
statNormalReach1AsyIdx = allAsyIdx[1, startAtaxia -n : startAtaxia , :]
statNormalReach1SmoIdx = allSmoIdx[1, startAtaxia -n : startAtaxia , :] 

statAtaxiaReach1LinIdx = allLinIdx[1,startAtaxia : startAtaxia +n,:]  
statAtaxiaReach1AsyIdx = allAsyIdx[1,startAtaxia : startAtaxia +n,:]
statAtaxiaReach1SmoIdx = allSmoIdx[1,startAtaxia : startAtaxia +n,:]  

stattDCSReach1LinIdx = allLinIdx[1,startAtaxia +n : startAtaxia + 2*n,:]  
stattDCSReach1AsyIdx = allAsyIdx[1,startAtaxia +n : startAtaxia + 2*n,:]
stattDCSReach1SmoIdx = allSmoIdx[1,startAtaxia +n : startAtaxia + 2*n,:]  


"REACH 3"
statNormalReach3LinIdx = allLinIdx[3,startAtaxia -n : startAtaxia,:]  
statNormalReach3AsyIdx = allAsyIdx[3,startAtaxia -n : startAtaxia,:]
statNormalReach3SmoIdx = allSmoIdx[3,startAtaxia -n : startAtaxia,:]  

statAtaxiaReach3LinIdx = allLinIdx[3,startAtaxia : startAtaxia +n,:]  
statAtaxiaReach3AsyIdx = allAsyIdx[3,startAtaxia : startAtaxia +n,:]
statAtaxiaReach3SmoIdx = allSmoIdx[3,startAtaxia : startAtaxia +n,:]  

stattDCSReach3LinIdx = allLinIdx[3,startAtaxia +n : startAtaxia + 2*n,:]  
stattDCSReach3AsyIdx = allAsyIdx[3,startAtaxia +n : startAtaxia + 2*n,:]
stattDCSReach3SmoIdx = allSmoIdx[3,startAtaxia +n : startAtaxia + 2*n,:]  


"REACH 5"
statNormalReach5LinIdx = allLinIdx[5,startAtaxia -n : startAtaxia,:]  
statNormalReach5AsyIdx = allAsyIdx[5,startAtaxia -n : startAtaxia,:]
statNormalReach5SmoIdx = allSmoIdx[5,startAtaxia -n : startAtaxia,:]  

statAtaxiaReach5LinIdx = allLinIdx[5,startAtaxia : startAtaxia +n,:]  
statAtaxiaReach5AsyIdx = allAsyIdx[5,startAtaxia : startAtaxia +n,:]
statAtaxiaReach5SmoIdx = allSmoIdx[5,startAtaxia : startAtaxia +n,:]

stattDCSReach5LinIdx = allLinIdx[5,startAtaxia +n : startAtaxia + 2*n,:]  
stattDCSReach5AsyIdx = allAsyIdx[5,startAtaxia +n : startAtaxia + 2*n,:]
stattDCSReach5SmoIdx = allSmoIdx[5,startAtaxia +n : startAtaxia + 2*n,:]  


"FORWARD REACHINGS"
statNormalForwardReachingsLinIdx = np.concatenate([statNormalReach1LinIdx,statNormalReach3LinIdx,statNormalReach5LinIdx], axis =0).copy()
statNormalForwardReachingsAsyIdx = np.concatenate([statNormalReach1AsyIdx,statNormalReach3AsyIdx,statNormalReach5AsyIdx], axis =0).copy()
statNormalForwardReachingsSmoIdx = np.concatenate([statNormalReach1SmoIdx,statNormalReach3SmoIdx,statNormalReach5SmoIdx], axis =0).copy()

statAtaxiaForwardReachingsLinIdx = np.concatenate([statAtaxiaReach1LinIdx,statAtaxiaReach3LinIdx,statAtaxiaReach5LinIdx], axis =0).copy()
statAtaxiaForwardReachingsAsyIdx = np.concatenate([statAtaxiaReach1AsyIdx,statAtaxiaReach3AsyIdx,statAtaxiaReach5AsyIdx], axis =0).copy()
statAtaxiaForwardReachingsSmoIdx = np.concatenate([statAtaxiaReach1SmoIdx,statAtaxiaReach3SmoIdx,statAtaxiaReach5SmoIdx], axis =0).copy()

stattDCSForwardReachingsLinIdx = np.concatenate([stattDCSReach1LinIdx,stattDCSReach3LinIdx,stattDCSReach5LinIdx], axis =0).copy()
stattDCSForwardReachingsAsyIdx = np.concatenate([stattDCSReach1AsyIdx,stattDCSReach3AsyIdx,stattDCSReach5AsyIdx], axis =0).copy()
stattDCSForwardReachingsSmoIdx = np.concatenate([stattDCSReach1SmoIdx,stattDCSReach3SmoIdx,stattDCSReach5SmoIdx], axis =0).copy()






"REACH 2"         
statNormalReach2LinIdx = allLinIdx[2,startAtaxia -n : startAtaxia,:]  
statNormalReach2AsyIdx = allAsyIdx[2,startAtaxia -n : startAtaxia,:]
statNormalReach2SmoIdx = allSmoIdx[2,startAtaxia -n : startAtaxia,:] 

statAtaxiaReach2LinIdx = allLinIdx[2,startAtaxia : startAtaxia +n,:]  
statAtaxiaReach2AsyIdx = allAsyIdx[2,startAtaxia : startAtaxia +n,:]
statAtaxiaReach2SmoIdx = allSmoIdx[2,startAtaxia : startAtaxia +n,:]

stattDCSReach2LinIdx = allLinIdx[2,startAtaxia +n : startAtaxia + 2*n,:]  
stattDCSReach2AsyIdx = allAsyIdx[2,startAtaxia +n : startAtaxia + 2*n,:]
stattDCSReach2SmoIdx = allSmoIdx[2,startAtaxia +n : startAtaxia + 2*n,:] 


"REACH 4"
statNormalReach4LinIdx = allLinIdx[4,startAtaxia -n : startAtaxia,:]  
statNormalReach4AsyIdx = allAsyIdx[4,startAtaxia -n : startAtaxia,:]
statNormalReach4SmoIdx = allSmoIdx[4,startAtaxia -n : startAtaxia,:]  

statAtaxiaReach4LinIdx = allLinIdx[4,startAtaxia : startAtaxia +n,:]  
statAtaxiaReach4AsyIdx = allAsyIdx[4,startAtaxia : startAtaxia +n,:]
statAtaxiaReach4SmoIdx = allSmoIdx[4,startAtaxia : startAtaxia +n,:]

stattDCSReach4LinIdx = allLinIdx[4,startAtaxia +n : startAtaxia + 2*n,:]  
stattDCSReach4AsyIdx = allAsyIdx[4,startAtaxia +n : startAtaxia + 2*n,:]
stattDCSReach4SmoIdx = allSmoIdx[4,startAtaxia +n : startAtaxia + 2*n,:] 


"REACH 6"
statNormalReach6LinIdx = allLinIdx[6,startAtaxia -n : startAtaxia,:]  
statNormalReach6AsyIdx = allAsyIdx[6,startAtaxia -n : startAtaxia,:]
statNormalReach6SmoIdx = allSmoIdx[6,startAtaxia -n : startAtaxia,:]  

statAtaxiaReach6LinIdx = allLinIdx[6,startAtaxia : startAtaxia +n,:]  
statAtaxiaReach6AsyIdx = allAsyIdx[6,startAtaxia : startAtaxia +n,:]
statAtaxiaReach6SmoIdx = allSmoIdx[6,startAtaxia : startAtaxia +n,:]

stattDCSReach6LinIdx = allLinIdx[6,startAtaxia +n : startAtaxia + 2*n,:]  
stattDCSReach6AsyIdx = allAsyIdx[6,startAtaxia +n : startAtaxia + 2*n,:]
stattDCSReach6SmoIdx = allSmoIdx[6,startAtaxia +n : startAtaxia + 2*n,:] 


"BACKWARD REACHINGS"
statNormalBackwardReachingsLinIdx = np.concatenate([statNormalReach2LinIdx,statNormalReach4LinIdx,statNormalReach6LinIdx], axis =0).copy()
statNormalBackwardReachingsAsyIdx = np.concatenate([statNormalReach2AsyIdx,statNormalReach4AsyIdx,statNormalReach6AsyIdx], axis =0).copy()
statNormalBackwardReachingsSmoIdx = np.concatenate([statNormalReach2SmoIdx,statNormalReach4SmoIdx,statNormalReach6SmoIdx], axis =0).copy()

statAtaxiaBackwardReachingsLinIdx = np.concatenate([statAtaxiaReach2LinIdx,statAtaxiaReach4LinIdx,statAtaxiaReach6LinIdx], axis =0).copy()
statAtaxiaBackwardReachingsAsyIdx = np.concatenate([statAtaxiaReach2AsyIdx,statAtaxiaReach4AsyIdx,statAtaxiaReach6AsyIdx], axis =0).copy()
statAtaxiaBackwardReachingsSmoIdx = np.concatenate([statAtaxiaReach2SmoIdx,statAtaxiaReach4SmoIdx,statAtaxiaReach6SmoIdx], axis =0).copy()

stattDCSBackwardReachingsLinIdx = np.concatenate([stattDCSReach2LinIdx,stattDCSReach4LinIdx,stattDCSReach6LinIdx], axis =0).copy()
stattDCSBackwardReachingsAsyIdx = np.concatenate([stattDCSReach2AsyIdx,stattDCSReach4AsyIdx,stattDCSReach6AsyIdx], axis =0).copy()
stattDCSBackwardReachingsSmoIdx = np.concatenate([stattDCSReach2SmoIdx,stattDCSReach4SmoIdx,stattDCSReach6SmoIdx], axis =0).copy()






"ALL REACHINGS"
statNormalAllReachingsLinIdx = np.concatenate([statNormalReach1LinIdx,statNormalReach2LinIdx,statNormalReach3LinIdx,statNormalReach4LinIdx,statNormalReach5LinIdx,statNormalReach6LinIdx], axis =0).copy()
statNormalAllReachingsAsyIdx = np.concatenate([statNormalReach1AsyIdx,statNormalReach2AsyIdx,statNormalReach3AsyIdx,statNormalReach4AsyIdx,statNormalReach5AsyIdx,statNormalReach6AsyIdx], axis =0).copy()
statNormalAllReachingsSmoIdx = np.concatenate([statNormalReach1SmoIdx,statNormalReach2SmoIdx,statNormalReach3SmoIdx,statNormalReach4SmoIdx,statNormalReach5SmoIdx,statNormalReach6SmoIdx], axis =0).copy()

statAtaxiaAllReachingsLinIdx = np.concatenate([statAtaxiaReach1LinIdx,statAtaxiaReach2LinIdx,statAtaxiaReach3LinIdx,statAtaxiaReach4LinIdx,statAtaxiaReach5LinIdx,statAtaxiaReach6LinIdx], axis =0).copy()
statAtaxiaAllReachingsAsyIdx = np.concatenate([statAtaxiaReach1AsyIdx,statAtaxiaReach2AsyIdx,statAtaxiaReach3AsyIdx,statAtaxiaReach4AsyIdx,statAtaxiaReach5AsyIdx,statAtaxiaReach6AsyIdx], axis =0).copy()
statAtaxiaAllReachingsSmoIdx = np.concatenate([statAtaxiaReach1SmoIdx,statAtaxiaReach2SmoIdx,statAtaxiaReach3SmoIdx,statAtaxiaReach4SmoIdx,statAtaxiaReach5SmoIdx,statAtaxiaReach6SmoIdx], axis =0).copy()

stattDCSAllReachingsLinIdx = np.concatenate([stattDCSReach1LinIdx,stattDCSReach2LinIdx,stattDCSReach3LinIdx,stattDCSReach4LinIdx,stattDCSReach5LinIdx,stattDCSReach6LinIdx], axis =0).copy()
stattDCSAllReachingsAsyIdx = np.concatenate([stattDCSReach1AsyIdx,stattDCSReach2AsyIdx,stattDCSReach3AsyIdx,stattDCSReach4AsyIdx,stattDCSReach5AsyIdx,stattDCSReach6AsyIdx], axis =0).copy()
stattDCSAllReachingsSmoIdx = np.concatenate([stattDCSReach1SmoIdx,stattDCSReach2SmoIdx,stattDCSReach3SmoIdx,stattDCSReach4SmoIdx,stattDCSReach5SmoIdx,stattDCSReach6SmoIdx], axis =0).copy()



"PROCESS DATA"


"REACH 1"
"LIN IDX mean&std in normal and ataxia"
meanNormalReach1LinIdx = np.mean(statNormalReach1LinIdx, axis =0) 
meanAtaxiaReach1LinIdx = np.mean(statAtaxiaReach1LinIdx, axis =0)
meantDCSReach1LinIdx = np.mean(stattDCSReach1LinIdx, axis =0)

stdNormalReach1LinIdx  = np.std(statNormalReach1LinIdx, axis =0)
stdAtaxiaReach1LinIdx  = np.std(statAtaxiaReach1LinIdx, axis =0)
stdtDCSReach1LinIdx  = np.std(stattDCSReach1LinIdx, axis =0)

"ASY IDX mean&std in normal and ataxia"
meanNormalReach1AsyIdx = np.mean(statNormalReach1AsyIdx, axis =0)
meanAtaxiaReach1AsyIdx= np.mean(statAtaxiaReach1AsyIdx, axis =0)
meantDCSReach1AsyIdx = np.mean(stattDCSReach1AsyIdx, axis =0)

stdNormalReach1AsyIdx  = np.std(statNormalReach1AsyIdx, axis =0)
stdAtaxiaReach1AsyIdx= np.std(statAtaxiaReach1AsyIdx, axis =0)
stdtDCSReach1AsyIdx  = np.std(stattDCSReach1AsyIdx, axis =0)

"SMO IDX mean&std in normal and ataxia"
meanNormalReach1SmoIdx = np.mean(statNormalReach1SmoIdx, axis =0)
meanAtaxiaReach1SmoIdx= np.mean(statAtaxiaReach1SmoIdx, axis =0)
meantDCSReach1SmoIdx = np.mean(stattDCSReach1SmoIdx, axis =0)

stdNormalReach1SmoIdx  = np.std(statNormalReach1SmoIdx, axis =0)
stdAtaxiaReach1SmoIdx= np.std(statAtaxiaReach1SmoIdx, axis =0)
stdtDCSReach1SmoIdx  = np.std(stattDCSReach1SmoIdx, axis =0)


"REACH 3"
"LIN IDX mean&std in normal and ataxia"
meanNormalReach3LinIdx = np.mean(statNormalReach3LinIdx, axis =0)
meanAtaxiaReach3LinIdx = np.mean(statAtaxiaReach3LinIdx, axis =0)
meantDCSReach3LinIdx = np.mean(statAtaxiaReach3LinIdx, axis =0)

stdNormalReach3LinIdx  = np.std(statNormalReach3LinIdx, axis =0)
stdAtaxiaReach3LinIdx  = np.std(statAtaxiaReach3LinIdx, axis =0)
stdtDCSReach3LinIdx  = np.std(stattDCSReach3LinIdx, axis =0)

"ASY IDX mean&std in normal and ataxia"
meanNormalReach3AsyIdx = np.mean(statNormalReach3AsyIdx, axis =0)
meanAtaxiaReach3AsyIdx= np.mean(statAtaxiaReach3AsyIdx, axis =0)
meantDCSReach3AsyIdx = np.mean(stattDCSReach3AsyIdx, axis =0)

stdNormalReach3AsyIdx  = np.std(statNormalReach3AsyIdx, axis =0)
stdAtaxiaReach3AsyIdx= np.std(statAtaxiaReach3AsyIdx, axis =0)
stdtDCSReach3AsyIdx  = np.std(stattDCSReach3AsyIdx, axis =0)

"SMO IDX mean&std in normal and ataxia"
meanNormalReach3SmoIdx = np.mean(statNormalReach3SmoIdx, axis =0)
meanAtaxiaReach3SmoIdx= np.mean(statAtaxiaReach3SmoIdx, axis =0)
meantDCSReach3SmoIdx = np.mean(stattDCSReach3SmoIdx, axis =0)

stdNormalReach3SmoIdx  = np.std(statNormalReach3SmoIdx, axis =0)
stdAtaxiaReach3SmoIdx= np.std(statAtaxiaReach3SmoIdx, axis =0)
stdtDCSReach3SmoIdx  = np.std(stattDCSReach3SmoIdx, axis =0)


"REACH 5"
"LIN IDX mean&std in normal and ataxia"
meanNormalReach5LinIdx = np.mean(statNormalReach5LinIdx, axis =0)
meanAtaxiaReach5LinIdx = np.mean(statAtaxiaReach5LinIdx, axis =0)
meantDCSReach5LinIdx = np.mean(stattDCSReach5LinIdx, axis =0)

stdNormalReach5LinIdx  = np.std(statNormalReach5LinIdx, axis =0)
stdAtaxiaReach5LinIdx  = np.std(statAtaxiaReach5LinIdx, axis =0)
stdtDCSReach5LinIdx  = np.std(stattDCSReach5LinIdx, axis =0)

"ASY IDX mean&std in normal and ataxia"
meanNormalReach5AsyIdx = np.mean(statNormalReach5AsyIdx, axis =0)
meanAtaxiaReach5AsyIdx = np.mean(statAtaxiaReach5AsyIdx, axis =0)
meantDCSReach5AsyIdx = np.mean(stattDCSReach5AsyIdx, axis =0)

stdNormalReach5AsyIdx  = np.std(statNormalReach5AsyIdx, axis =0)
stdAtaxiaReach5AsyIdx= np.std(statAtaxiaReach5AsyIdx, axis =0)
stdtDCSReach5AsyIdx  = np.std(stattDCSReach5AsyIdx, axis =0)

"SMO IDX mean&std in normal and ataxia"
meanNormalReach5SmoIdx = np.mean(statNormalReach5SmoIdx, axis =0)
meanAtaxiaReach5SmoIdx= np.mean(statAtaxiaReach5SmoIdx, axis =0)
meantDCSReach5SmoIdx = np.mean(stattDCSReach5SmoIdx, axis =0)

stdNormalReach5SmoIdx  = np.std(statNormalReach5SmoIdx, axis =0)
stdAtaxiaReach5SmoIdx= np.std(statAtaxiaReach5SmoIdx, axis =0)
stdtDCSReach5SmoIdx  = np.std(stattDCSReach5SmoIdx, axis =0)


"FORWARD REACHINGS"
"LIN IDX mean&std in normal and ataxia"
meanNormalForwardReachingsLinIdx = np.mean(statNormalForwardReachingsLinIdx, axis =0)
meanAtaxiaForwardReachingsLinIdx = np.mean(statAtaxiaForwardReachingsLinIdx, axis =0)
meantDCSForwardReachingsLinIdx = np.mean(stattDCSForwardReachingsLinIdx, axis =0)

stdNormalForwardReachingsLinIdx  = np.std(statNormalForwardReachingsLinIdx, axis =0)
stdAtaxiaForwardReachingsLinIdx  = np.std(statAtaxiaForwardReachingsLinIdx, axis =0)
stdtDCSForwardReachingsLinIdx  = np.std(stattDCSForwardReachingsLinIdx, axis =0)

"ASY IDX mean&std in normal and ataxia"
meanNormalForwardReachingsAsyIdx = np.mean(statNormalForwardReachingsAsyIdx, axis =0)
meanAtaxiaForwardReachingsAsyIdx = np.mean(statAtaxiaForwardReachingsAsyIdx, axis =0)
meantDCSForwardReachingsAsyIdx = np.mean(stattDCSForwardReachingsAsyIdx, axis =0)

stdNormalForwardReachingsAsyIdx  = np.std(statNormalForwardReachingsAsyIdx, axis =0)
stdAtaxiaForwardReachingsAsyIdx  = np.std(statAtaxiaForwardReachingsAsyIdx, axis =0)
stdtDCSForwardReachingsAsyIdx  = np.std(stattDCSForwardReachingsAsyIdx, axis =0)

"SMO IDX mean&std in normal and ataxia"
meanNormalForwardReachingsSmoIdx = np.mean(statNormalForwardReachingsSmoIdx, axis =0)
meanAtaxiaForwardReachingsSmoIdx = np.mean(statAtaxiaForwardReachingsSmoIdx, axis =0)
meantDCSForwardReachingsSmoIdx = np.mean(stattDCSForwardReachingsSmoIdx, axis =0)

stdNormalForwardReachingsSmoIdx = np.std(statNormalForwardReachingsSmoIdx, axis =0)
stdAtaxiaForwardReachingsSmoIdx = np.std(statAtaxiaForwardReachingsSmoIdx, axis =0)
stdtDCSForwardReachingsSmoIdx  = np.std(statAtaxiaForwardReachingsSmoIdx, axis =0)





"REACH 2"
"LIN IDX mean&std in normal and ataxia"
meanNormalReach2LinIdx = np.mean(statNormalReach2LinIdx, axis =0) 
meanAtaxiaReach2LinIdx = np.mean(statAtaxiaReach2LinIdx, axis =0)
meantDCSReach2LinIdx = np.mean(stattDCSReach2LinIdx, axis =0)


stdNormalReach2LinIdx  = np.std(statNormalReach2LinIdx, axis =0)
stdAtaxiaReach2LinIdx  = np.std(statAtaxiaReach2LinIdx, axis =0)
stdtDCSReach2LinIdx  = np.std(stattDCSReach2LinIdx, axis =0)

"ASY IDX mean&std in normal and ataxia"
meanNormalReach2AsyIdx = np.mean(statNormalReach2AsyIdx, axis =0)
meanAtaxiaReach2AsyIdx= np.mean(statAtaxiaReach2AsyIdx, axis =0)
meantDCSReach2AsyIdx = np.mean(stattDCSReach2AsyIdx, axis =0)

stdNormalReach2AsyIdx  = np.std(statNormalReach2AsyIdx, axis =0)
stdAtaxiaReach2AsyIdx  = np.std(statAtaxiaReach2AsyIdx, axis =0)
stdtDCSReach2AsyIdx  = np.std(stattDCSReach2AsyIdx, axis =0)

"SMO IDX mean&std in normal and ataxia"
meanNormalReach2SmoIdx = np.mean(statNormalReach2SmoIdx, axis =0)
meanAtaxiaReach2SmoIdx= np.mean(statAtaxiaReach2SmoIdx, axis =0)
meantDCSReach2SmoIdx = np.mean(stattDCSReach2SmoIdx, axis =0)

stdNormalReach2SmoIdx  = np.std(statNormalReach2SmoIdx, axis =0)
stdAtaxiaReach2SmoIdx= np.std(statAtaxiaReach2SmoIdx, axis =0)
stdtDCSReach2SmoIdx  = np.std(stattDCSReach2SmoIdx, axis =0)


"REACH 4"
"LIN IDX mean&std in normal and ataxia"
meanNormalReach4LinIdx = np.mean(statNormalReach4LinIdx, axis =0)
meanAtaxiaReach4LinIdx = np.mean(statAtaxiaReach4LinIdx, axis =0)
meantDCSReach4LinIdx = np.mean(stattDCSReach4LinIdx, axis =0)

stdNormalReach4LinIdx  = np.std(statNormalReach4LinIdx, axis =0)
stdAtaxiaReach4LinIdx  = np.std(statAtaxiaReach4LinIdx, axis =0)
stdtDCSReach4LinIdx  = np.std(stattDCSReach4LinIdx, axis =0)

"ASY IDX mean&std in normal and ataxia"
meanNormalReach4AsyIdx = np.mean(statNormalReach4AsyIdx, axis =0)
meanAtaxiaReach4AsyIdx= np.mean(statAtaxiaReach4AsyIdx, axis =0)
meantDCSReach4AsyIdx = np.mean(stattDCSReach4AsyIdx, axis =0)

stdNormalReach4AsyIdx  = np.std(statNormalReach4AsyIdx, axis =0)
stdAtaxiaReach4AsyIdx= np.std(statAtaxiaReach4AsyIdx, axis =0)
stdtDCSReach4AsyIdx  = np.std(stattDCSReach4AsyIdx, axis =0)

"SMO IDX mean&std in normal and ataxia"
meanNormalReach4SmoIdx = np.mean(statNormalReach4SmoIdx, axis =0)
meanAtaxiaReach4SmoIdx= np.mean(statAtaxiaReach4SmoIdx, axis =0)
meantDCSReach4SmoIdx = np.mean(stattDCSReach4SmoIdx, axis =0)

stdNormalReach4SmoIdx  = np.std(statNormalReach4SmoIdx, axis =0)
stdAtaxiaReach4SmoIdx= np.std(statAtaxiaReach4SmoIdx, axis =0)
stdtDCSReach4SmoIdx  = np.std(stattDCSReach4SmoIdx, axis =0)


"REACH 6"
"LIN IDX mean&std in normal and ataxia"
meanNormalReach6LinIdx = np.mean(statNormalReach6LinIdx, axis =0)
meanAtaxiaReach6LinIdx = np.mean(statAtaxiaReach6LinIdx, axis =0)
meantDCSReach6LinIdx = np.mean(stattDCSReach6LinIdx, axis =0)

stdNormalReach6LinIdx  = np.std(statNormalReach6LinIdx, axis =0)
stdAtaxiaReach6LinIdx  = np.std(statAtaxiaReach6LinIdx, axis =0)
stdtDCSReach6LinIdx  = np.std(stattDCSReach6LinIdx, axis =0)

"ASY IDX mean&std in normal and ataxia"
meanNormalReach6AsyIdx = np.mean(statNormalReach6AsyIdx, axis =0)
meanAtaxiaReach6AsyIdx= np.mean(statAtaxiaReach6AsyIdx, axis =0)
meantDCSReach6AsyIdx = np.mean(stattDCSReach6AsyIdx, axis =0)

stdNormalReach6AsyIdx  = np.std(statNormalReach6AsyIdx, axis =0)
stdAtaxiaReach6AsyIdx= np.std(statAtaxiaReach6AsyIdx, axis =0)
stdtDCSReach6AsyIdx  = np.std(stattDCSReach6AsyIdx, axis =0)

"SMO IDX mean&std in normal and ataxia"
meanNormalReach6SmoIdx = np.mean(statNormalReach6SmoIdx, axis =0)
meanAtaxiaReach6SmoIdx= np.mean(statAtaxiaReach6SmoIdx, axis =0)
meantDCSReach6SmoIdx = np.mean(stattDCSReach6SmoIdx, axis =0)

stdNormalReach6SmoIdx  = np.std(statNormalReach6SmoIdx, axis =0)
stdAtaxiaReach6SmoIdx  = np.std(statAtaxiaReach6SmoIdx, axis =0)
stdtDCSReach6SmoIdx  = np.std(stattDCSReach6SmoIdx, axis =0)


"BACKWARD REACHINGS"
"LIN IDX mean&std in normal and ataxia"
meanNormalBackwardReachingsLinIdx = np.mean(statNormalBackwardReachingsLinIdx, axis =0)
meanAtaxiaBackwardReachingsLinIdx = np.mean(statAtaxiaBackwardReachingsLinIdx, axis =0)
meantDCSBackwardReachingsLinIdx = np.mean(stattDCSBackwardReachingsLinIdx, axis =0)

stdNormalBackwardReachingsLinIdx  = np.std(statNormalBackwardReachingsLinIdx, axis =0)
stdAtaxiaBackwardReachingsLinIdx  = np.std(statAtaxiaBackwardReachingsLinIdx, axis =0)
stdtDCSBackwardReachingsLinIdx  = np.std(stattDCSBackwardReachingsLinIdx, axis =0)

"ASY IDX mean&std in normal and ataxia"
meanNormalBackwardReachingsAsyIdx = np.mean(statNormalBackwardReachingsAsyIdx, axis =0)
meanAtaxiaBackwardReachingsAsyIdx = np.mean(statAtaxiaBackwardReachingsAsyIdx, axis =0)
meantDCSBackwardReachingsAsyIdx = np.mean(stattDCSBackwardReachingsAsyIdx, axis =0)

stdNormalBackwardReachingsAsyIdx  = np.std(statNormalBackwardReachingsAsyIdx, axis =0)
stdAtaxiaBackwardReachingsAsyIdx  = np.std(statAtaxiaBackwardReachingsAsyIdx, axis =0)
stdtDCSBackwardReachingsAsyIdx  = np.std(stattDCSBackwardReachingsAsyIdx, axis =0)

"SMO IDX mean&std in normal and ataxia"
meanNormalBackwardReachingsSmoIdx = np.mean(statNormalBackwardReachingsSmoIdx, axis =0)
meanAtaxiaBackwardReachingsSmoIdx = np.mean(statAtaxiaBackwardReachingsSmoIdx, axis =0)
meantDCSBackwardReachingsSmoIdx = np.mean(stattDCSBackwardReachingsSmoIdx, axis =0)

stdNormalBackwardReachingsSmoIdx = np.std(statNormalBackwardReachingsSmoIdx, axis =0)
stdAtaxiaBackwardReachingsSmoIdx = np.std(statAtaxiaBackwardReachingsSmoIdx, axis =0)
stdtDCSBackwardReachingsSmoIdx  = np.std(stattDCSBackwardReachingsSmoIdx, axis =0)


"ALL REACHINGS"
"LIN IDX mean&std in normal and ataxia"
meanNormalAllReachingsLinIdx = np.mean(statNormalAllReachingsLinIdx, axis =0)
meanAtaxiaAllReachingsLinIdx = np.mean(statAtaxiaAllReachingsLinIdx, axis =0)
meantDCSAllReachingsLinIdx = np.mean(stattDCSAllReachingsLinIdx, axis =0)

stdNormalAllReachingsLinIdx  = np.std(statNormalAllReachingsLinIdx, axis =0)
stdAtaxiaAllReachingsLinIdx  = np.std(statAtaxiaAllReachingsLinIdx, axis =0)
stdtDCSAllReachingsLinIdx  = np.std(stattDCSAllReachingsLinIdx, axis =0)

"ASY IDX mean&std in normal and ataxia"
meanNormalAllReachingsAsyIdx = np.mean(statNormalAllReachingsAsyIdx, axis =0)
meanAtaxiaAllReachingsAsyIdx = np.mean(statAtaxiaAllReachingsAsyIdx, axis =0)
meantDCSAllReachingsAsyIdx = np.mean(stattDCSAllReachingsAsyIdx, axis =0)

stdNormalAllReachingsAsyIdx  = np.std(statNormalAllReachingsAsyIdx, axis =0)
stdAtaxiaAllReachingsAsyIdx  = np.std(statAtaxiaAllReachingsAsyIdx, axis =0)
stdtDCSAllReachingsAsyIdx  = np.std(stattDCSAllReachingsAsyIdx, axis =0)

"SMO IDX mean&std in normal and ataxia"
meanNormalAllReachingsSmoIdx = np.mean(statNormalAllReachingsSmoIdx, axis =0)
meanAtaxiaAllReachingsSmoIdx = np.mean(statAtaxiaAllReachingsSmoIdx, axis =0)
meantDCSAllReachingsSmoIdx = np.mean(stattDCSAllReachingsSmoIdx, axis =0)

stdNormalAllReachingsSmoIdx = np.std(statNormalAllReachingsSmoIdx, axis =0)
stdAtaxiaAllReachingsSmoIdx = np.std(statAtaxiaAllReachingsSmoIdx, axis =0)
stdtDCSAllReachingsSmoIdx  = np.std(stattDCSAllReachingsSmoIdx, axis =0)







"INIT PLOTTING"

"""
"FORWARD REACHING INDEX"
fig1   = plt.figure("FORWARD REACHING INDEX", figsize=(20,10))
reach1 = plt.figtext(.20, 0.90, "REACH 1" , style='normal', bbox={'facecolor':'orangered'})
reach2 = plt.figtext(.48, 0.90, "REACH 3" , style='normal', bbox={'facecolor':'orangered'})
reach3 = plt.figtext(.76, 0.90, "REACH 5" , style='normal', bbox={'facecolor':'orangered'})

LinIdx = plt.figtext(.01, 0.77, "LIN IDX" , style='normal', bbox={'facecolor':'orangered'})
Asyidx = plt.figtext(.01, 0.49, "ASY IDX" , style='normal', bbox={'facecolor':'orangered'})
SmoIdx = plt.figtext(.01, 0.21, "SMO IDX" , style='normal', bbox={'facecolor':'orangered'})



"PLOT REACH 1 LINEARITY INDEX mean&std in normal and ataxia"
reach1LinIdxPlot = fig1.add_subplot(331)

reach1LinIdxPlot.set_ylim([0,100.0])
reach1LinIdxPlot.set_xlim([0,(maxSeed *3) +1])
reach1LinIdxPlot.set_yticks(np.arange(0, 100.0, 10.0))
reach1LinIdxPlot.set_xticks(np.arange(1, (maxSeed *3) +1, 1))

reach1LinIdxPlot.errorbar(np.linspace(1,maxSeed,maxSeed), meanNormalReach1LinIdx, yerr = stdNormalReach1LinIdx , marker= '^' , mec ='black', mfc='white' , ecolor ='black', fmt = '')
meanAllSeedsNormalReach1LinIdx = patches.Rectangle((1,0), maxSeed -1,np.mean(meanNormalReach1LinIdx),linewidth=1,edgecolor='black',facecolor='lightgrey')
reach1LinIdxPlot.add_patch(meanAllSeedsNormalReach1LinIdx)

reach1LinIdxPlot.errorbar(np.linspace((maxSeed +1),maxSeed*2,maxSeed), meanAtaxiaReach1LinIdx,  yerr = stdAtaxiaReach1LinIdx, marker= '^', mec ='black', mfc='black', ecolor ='black', fmt = '')
meanAllSeedsAtaxiaReach1LinIdx = patches.Rectangle((maxSeed +1,0), maxSeed -1, np.mean(meanAtaxiaReach1LinIdx),linewidth=1,edgecolor='black',facecolor='lightgrey')
reach1LinIdxPlot.add_patch(meanAllSeedsAtaxiaReach1LinIdx)

reach1LinIdxPlot.errorbar(np.linspace((maxSeed*2 +1),maxSeed*3,maxSeed), meantDCSReach1LinIdx,  yerr = stdtDCSReach1LinIdx, marker= '^', mec ='red', mfc='black', ecolor ='black', fmt = '')
meanAllSeedstDCSReach1LinIdx = patches.Rectangle((maxSeed*2 +1,0), maxSeed -1, np.mean(meantDCSReach1LinIdx),linewidth=1,edgecolor='black',facecolor='lightgrey')
reach1LinIdxPlot.add_patch(meanAllSeedstDCSReach1LinIdx)


"PLOT REACH 3 LINEARITY INDEX mean&std in normal and ataxia"
reach3LinIdxPlot = fig1.add_subplot(332)

reach3LinIdxPlot.set_ylim([0,100.0])
reach3LinIdxPlot.set_xlim([0,(maxSeed *3) +1])
reach3LinIdxPlot.set_yticks(np.arange(0, 100.0, 10.0))
reach3LinIdxPlot.set_xticks(np.arange(1, (maxSeed *3) +1, 1))


reach3LinIdxPlot.errorbar(np.linspace(1,maxSeed,maxSeed), meanNormalReach3LinIdx, yerr = stdNormalReach3LinIdx , marker= '^' , mec ='black', mfc='white' , ecolor ='black', fmt = '')
meanAllSeedsNormalReach3LinIdx = patches.Rectangle((1,0), maxSeed -1,np.mean(meanNormalReach3LinIdx),linewidth=1,edgecolor='black',facecolor='lightgrey')
reach3LinIdxPlot.add_patch(meanAllSeedsNormalReach3LinIdx)

reach3LinIdxPlot.errorbar(np.linspace((maxSeed +1),maxSeed*2,maxSeed), meanAtaxiaReach3LinIdx,  yerr = stdAtaxiaReach3LinIdx, marker= '^', mec ='black', mfc='black', ecolor ='black', fmt = '')
meanAllSeedsAtaxiaReach3LinIdx = patches.Rectangle((maxSeed +1,0), maxSeed -1, np.mean(meanAtaxiaReach3LinIdx),linewidth=1,edgecolor='black',facecolor='lightgrey')
reach3LinIdxPlot.add_patch(meanAllSeedsAtaxiaReach3LinIdx)

reach3LinIdxPlot.errorbar(np.linspace((maxSeed*2 +1),maxSeed*3,maxSeed), meantDCSReach3LinIdx,  yerr = stdtDCSReach3LinIdx, marker= '^', mec ='red', mfc='black', ecolor ='black', fmt = '')
meanAllSeedstDCSReach3LinIdx = patches.Rectangle((maxSeed*2 +1,0), maxSeed -1, np.mean(meantDCSReach3LinIdx),linewidth=1,edgecolor='black',facecolor='lightgrey')
reach3LinIdxPlot.add_patch(meanAllSeedstDCSReach3LinIdx)


"PLOT REACH 5 LINEARITY INDEX mean&std in normal and ataxia"
reach5LinIdxPlot = fig1.add_subplot(333)

reach5LinIdxPlot.set_ylim([0,100.0])
reach5LinIdxPlot.set_xlim([0,(maxSeed *3) +1])
reach5LinIdxPlot.set_yticks(np.arange(0, 100.0, 10.0))
reach5LinIdxPlot.set_xticks(np.arange(1, (maxSeed *3) +1, 1))


reach5LinIdxPlot.errorbar(np.linspace(1,maxSeed,maxSeed), meanNormalReach5LinIdx, yerr = stdNormalReach5LinIdx , marker= '^' , mec ='black', mfc='white' , ecolor ='black', fmt = '')
meanAllSeedsNormalReach5LinIdx = patches.Rectangle((1,0), maxSeed -1,np.mean(meanNormalReach5LinIdx),linewidth=1,edgecolor='black',facecolor='lightgrey')
reach5LinIdxPlot.add_patch(meanAllSeedsNormalReach5LinIdx)

reach5LinIdxPlot.errorbar(np.linspace((maxSeed +1),maxSeed*2,maxSeed), meanAtaxiaReach5LinIdx,  yerr = stdAtaxiaReach5LinIdx, marker= '^', mec ='black', mfc='black', ecolor ='black', fmt = '')
meanAllSeedsAtaxiaReach5LinIdx = patches.Rectangle((maxSeed +1,0), maxSeed -1, np.mean(meanAtaxiaReach5LinIdx),linewidth=1,edgecolor='black',facecolor='lightgrey')
reach5LinIdxPlot.add_patch(meanAllSeedsAtaxiaReach5LinIdx)

reach5LinIdxPlot.errorbar(np.linspace((maxSeed*2 +1),maxSeed*3,maxSeed), meantDCSReach5LinIdx,  yerr = stdtDCSReach5LinIdx, marker= '^', mec ='red', mfc='black', ecolor ='black', fmt = '')
meanAllSeedstDCSReach5LinIdx = patches.Rectangle((maxSeed*2 +1,0), maxSeed -1, np.mean(meantDCSReach5LinIdx),linewidth=1,edgecolor='black',facecolor='lightgrey')
reach5LinIdxPlot.add_patch(meanAllSeedstDCSReach5LinIdx)




"PLOT REACH 1 ASYMMETRY INDEX mean&std in normal and ataxia"
reach1AsyIdxPlot = fig1.add_subplot(334)

reach1AsyIdxPlot.set_ylim([0,7.0])
reach1AsyIdxPlot.set_xlim([0,(maxSeed *3) +1])
reach1AsyIdxPlot.set_yticks(np.arange(0, 7.0, 0.5))
reach1AsyIdxPlot.set_xticks(np.arange(1, (maxSeed *3) +1, 1))


reach1AsyIdxPlot.errorbar(np.linspace(1,maxSeed,maxSeed), meanNormalReach1AsyIdx, yerr = stdNormalReach1AsyIdx , marker= '^' , mec ='black', mfc='white' , ecolor ='black', fmt = '')
meanAllSeedsNormalReach1AsyIdx = patches.Rectangle((1,0), maxSeed -1 , np.mean(meanNormalReach1AsyIdx) ,linewidth=1 , edgecolor='black' , facecolor='lightgrey')
reach1AsyIdxPlot.add_patch(meanAllSeedsNormalReach1AsyIdx)

reach1AsyIdxPlot.errorbar(np.linspace((maxSeed +1),maxSeed*2,maxSeed), meanAtaxiaReach1AsyIdx,  yerr = stdAtaxiaReach1AsyIdx, marker= '^', mec ='black', mfc='black', ecolor ='black', fmt = '')
meanAllSeedsAtaxiaReach1AsyIdx = patches.Rectangle((maxSeed +1,0), maxSeed -1, np.mean(meanAtaxiaReach1AsyIdx),linewidth=1,edgecolor='black',facecolor='lightgrey')
reach1AsyIdxPlot.add_patch(meanAllSeedsAtaxiaReach1AsyIdx)

reach1AsyIdxPlot.errorbar(np.linspace((maxSeed*2 +1),maxSeed*3,maxSeed), meantDCSReach1AsyIdx,  yerr = stdtDCSReach1AsyIdx, marker= '^', mec ='red', mfc='black', ecolor ='black', fmt = '')
meanAllSeedstDCSReach1AsyIdx = patches.Rectangle((maxSeed*2 +1,0), maxSeed -1, np.mean(meantDCSReach1AsyIdx),linewidth=1,edgecolor='black',facecolor='lightgrey')
reach1AsyIdxPlot.add_patch(meanAllSeedstDCSReach1AsyIdx)






"PLOT REACH 3 ASYMMETRY INDEX mean&std in normal and ataxia"
reach3AsyIdxPlot = fig1.add_subplot(335)

reach3AsyIdxPlot.set_ylim([0,7.0])
reach3AsyIdxPlot.set_xlim([0,(maxSeed *3) +1])
reach3AsyIdxPlot.set_yticks(np.arange(0, 7.0, 0.5))
reach3AsyIdxPlot.set_xticks(np.arange(1, (maxSeed *3) +1, 1))


reach3AsyIdxPlot.errorbar(np.linspace(1,maxSeed,maxSeed), meanNormalReach3AsyIdx, yerr = stdNormalReach3AsyIdx , marker= '^' , mec ='black', mfc='white' , ecolor ='black', fmt = '')
meanAllSeedsNormalReach3AsyIdx = patches.Rectangle((1,0), maxSeed -1 , np.mean(meanNormalReach3AsyIdx) ,linewidth=1 , edgecolor='black' , facecolor='lightgrey')
reach3AsyIdxPlot.add_patch(meanAllSeedsNormalReach3AsyIdx)

reach3AsyIdxPlot.errorbar(np.linspace((maxSeed +1),maxSeed*2,maxSeed), meanAtaxiaReach3AsyIdx,  yerr = stdAtaxiaReach3AsyIdx, marker= '^', mec ='black', mfc='black', ecolor ='black', fmt = '')
meanAllSeedsAtaxiaReach3AsyIdx = patches.Rectangle((maxSeed +1,0), maxSeed -1, np.mean(meanAtaxiaReach3AsyIdx),linewidth=1,edgecolor='black',facecolor='lightgrey')
reach3AsyIdxPlot.add_patch(meanAllSeedsAtaxiaReach3AsyIdx)

reach3AsyIdxPlot.errorbar(np.linspace((maxSeed*2 +1),maxSeed*3,maxSeed), meantDCSReach3AsyIdx,  yerr = stdtDCSReach3AsyIdx, marker= '^', mec ='red', mfc='black', ecolor ='black', fmt = '')
meanAllSeedstDCSReach3AsyIdx = patches.Rectangle((maxSeed*2 +1,0), maxSeed -1, np.mean(meantDCSReach3AsyIdx),linewidth=1,edgecolor='black',facecolor='lightgrey')
reach3AsyIdxPlot.add_patch(meanAllSeedstDCSReach3AsyIdx)






"PLOT REACH 5 ASYMMETRY INDEX mean&std in normal and ataxia"
reach5AsyIdxPlot = fig1.add_subplot(336)

reach5AsyIdxPlot.set_ylim([0,7.0])
reach5AsyIdxPlot.set_xlim([0,(maxSeed *3) +1])
reach5AsyIdxPlot.set_yticks(np.arange(0, 7.0, 0.5))
reach5AsyIdxPlot.set_xticks(np.arange(1, (maxSeed *3) +1, 1))


reach5AsyIdxPlot.errorbar(np.linspace(1,maxSeed,maxSeed), meanNormalReach5AsyIdx, yerr = stdNormalReach5AsyIdx , marker= '^' , mec ='black', mfc='white' , ecolor ='black', fmt = '')
meanAllSeedsNormalReach5AsyIdx = patches.Rectangle((1,0), maxSeed -1 , np.mean(meanNormalReach5AsyIdx) ,linewidth=1 , edgecolor='black' , facecolor='lightgrey')
reach5AsyIdxPlot.add_patch(meanAllSeedsNormalReach5AsyIdx)

reach5AsyIdxPlot.errorbar(np.linspace((maxSeed +1),maxSeed*2,maxSeed), meanAtaxiaReach5AsyIdx,  yerr = stdAtaxiaReach5AsyIdx, marker= '^', mec ='black', mfc='black', ecolor ='black', fmt = '')
meanAllSeedsAtaxiaReach5AsyIdx = patches.Rectangle((maxSeed +1,0), maxSeed -1, np.mean(meanAtaxiaReach5AsyIdx),linewidth=1,edgecolor='black',facecolor='lightgrey')
reach5AsyIdxPlot.add_patch(meanAllSeedsAtaxiaReach5AsyIdx)

reach5AsyIdxPlot.errorbar(np.linspace((maxSeed*2 +1),maxSeed*3,maxSeed), meantDCSReach5AsyIdx,  yerr = stdtDCSReach5AsyIdx, marker= '^', mec ='red', mfc='black', ecolor ='black', fmt = '')
meanAllSeedstDCSReach5AsyIdx = patches.Rectangle((maxSeed*2 +1,0), maxSeed -1, np.mean(meantDCSReach5AsyIdx),linewidth=1,edgecolor='black',facecolor='lightgrey')
reach5AsyIdxPlot.add_patch(meanAllSeedstDCSReach5AsyIdx)







"PLOT REACH 1 SMOOTHNESS INDEX mean&std in normal and ataxia"
reach1SmoIdxPlot = fig1.add_subplot(337)

reach1SmoIdxPlot.set_ylim([0,15])
reach1SmoIdxPlot.set_xlim([0,(maxSeed *3) +1])
reach1SmoIdxPlot.set_yticks(np.arange(0, 15, 5))
reach1SmoIdxPlot.set_xticks(np.arange(1, (maxSeed *3) +1, 1))


reach1SmoIdxPlot.errorbar(np.linspace(1,maxSeed,maxSeed), meanNormalReach1SmoIdx, yerr = stdNormalReach1SmoIdx , marker= '^' , mec ='black', mfc='white' , ecolor ='black', fmt = '')
meanAllSeedsNormalReach1SmoIdx = patches.Rectangle((1,0), maxSeed -1 , np.mean(meanNormalReach1SmoIdx) ,linewidth=1 , edgecolor='black' , facecolor='lightgrey')
reach1SmoIdxPlot.add_patch(meanAllSeedsNormalReach1SmoIdx)

reach1SmoIdxPlot.errorbar(np.linspace((maxSeed +1),maxSeed*2,maxSeed), meanAtaxiaReach1SmoIdx,  yerr = stdAtaxiaReach1SmoIdx, marker= '^', mec ='black', mfc='black', ecolor ='black', fmt = '')
meanAllSeedsAtaxiaReach1SmoIdx = patches.Rectangle((maxSeed +1,0), maxSeed -1, np.mean(meanAtaxiaReach1SmoIdx),linewidth=1,edgecolor='black',facecolor='lightgrey')
reach1SmoIdxPlot.add_patch(meanAllSeedsAtaxiaReach1SmoIdx)

reach1SmoIdxPlot.errorbar(np.linspace((maxSeed*2 +1),maxSeed*3,maxSeed), meantDCSReach1SmoIdx,  yerr = stdtDCSReach1SmoIdx, marker= '^', mec ='red', mfc='black', ecolor ='black', fmt = '')
meanAllSeedstDCSReach1SmoIdx = patches.Rectangle((maxSeed*2 +1,0), maxSeed -1, np.mean(meantDCSReach1SmoIdx),linewidth=1,edgecolor='black',facecolor='lightgrey')
reach1SmoIdxPlot.add_patch(meanAllSeedstDCSReach1SmoIdx)





"PLOT REACH 3 SMOOTHNESS INDEX mean&std in normal and ataxia"
reach3SmoIdxPlot = fig1.add_subplot(338)

reach3SmoIdxPlot.set_ylim([0,15])
reach3SmoIdxPlot.set_xlim([0,(maxSeed *3) +1])
reach3SmoIdxPlot.set_yticks(np.arange(0, 15, 5))
reach3SmoIdxPlot.set_xticks(np.arange(1, (maxSeed *3) +1, 1))

reach3SmoIdxPlot.errorbar(np.linspace(1,maxSeed,maxSeed), meanNormalReach3SmoIdx, yerr = stdNormalReach3SmoIdx , marker= '^' , mec ='black', mfc='white' , ecolor ='black', fmt = '')
meanAllSeedsNormalReach3SmoIdx = patches.Rectangle((1,0), maxSeed -1 , np.mean(meanNormalReach3SmoIdx) ,linewidth=1 , edgecolor='black' , facecolor='lightgrey')
reach3SmoIdxPlot.add_patch(meanAllSeedsNormalReach3SmoIdx)

reach3SmoIdxPlot.errorbar(np.linspace((maxSeed +1),maxSeed*2,maxSeed), meanAtaxiaReach3SmoIdx,  yerr = stdAtaxiaReach3SmoIdx, marker= '^', mec ='black', mfc='black', ecolor ='black', fmt = '')
meanAllSeedsAtaxiaReach3SmoIdx = patches.Rectangle((maxSeed +1,0), maxSeed -1, np.mean(meanAtaxiaReach3SmoIdx),linewidth=1,edgecolor='black',facecolor='lightgrey')
reach3SmoIdxPlot.add_patch(meanAllSeedsAtaxiaReach3SmoIdx)

reach3SmoIdxPlot.errorbar(np.linspace((maxSeed*2 +1),maxSeed*3,maxSeed), meantDCSReach3SmoIdx,  yerr = stdtDCSReach3SmoIdx, marker= '^', mec ='red', mfc='black', ecolor ='black', fmt = '')
meanAllSeedstDCSReach3SmoIdx = patches.Rectangle((maxSeed*2 +1,0), maxSeed -1, np.mean(meantDCSReach3SmoIdx),linewidth=1,edgecolor='black',facecolor='lightgrey')
reach3SmoIdxPlot.add_patch(meanAllSeedstDCSReach3SmoIdx)




"PLOT REACH 5 SMOOTHNESS INDEX mean&std in normal and ataxia"
reach5SmoIdxPlot = fig1.add_subplot(339)

reach5SmoIdxPlot.set_ylim([0,15])
reach5SmoIdxPlot.set_xlim([0,(maxSeed *3) +1])
reach5SmoIdxPlot.set_yticks(np.arange(0, 15, 5))
reach5SmoIdxPlot.set_xticks(np.arange(1, (maxSeed *3) +1, 1))



reach5SmoIdxPlot.errorbar(np.linspace(1,maxSeed,maxSeed), meanNormalReach5SmoIdx, yerr = stdNormalReach5SmoIdx , marker= '^' , mec ='black', mfc='white' , ecolor ='black', fmt = '')
meanAllSeedsNormalReach5SmoIdx = patches.Rectangle((1,0), maxSeed -1 , np.mean(meanNormalReach5SmoIdx) ,linewidth=1 , edgecolor='black' , facecolor='lightgrey')
reach5SmoIdxPlot.add_patch(meanAllSeedsNormalReach5SmoIdx)

reach5SmoIdxPlot.errorbar(np.linspace((maxSeed +1),maxSeed*2,maxSeed), meanAtaxiaReach5SmoIdx,  yerr = stdAtaxiaReach5SmoIdx, marker= '^', mec ='black', mfc='black', ecolor ='black', fmt = '')
meanAllSeedsAtaxiaReach5SmoIdx = patches.Rectangle((maxSeed +1,0), maxSeed -1, np.mean(meanAtaxiaReach5SmoIdx),linewidth=1,edgecolor='black',facecolor='lightgrey')
reach5SmoIdxPlot.add_patch(meanAllSeedsAtaxiaReach5SmoIdx)

reach5SmoIdxPlot.errorbar(np.linspace((maxSeed*2 +1),maxSeed*3,maxSeed), meantDCSReach5SmoIdx,  yerr = stdtDCSReach5SmoIdx, marker= '^', mec ='red', mfc='black', ecolor ='black', fmt = '')
meanAllSeedstDCSReach5SmoIdx = patches.Rectangle((maxSeed*2 +1,0), maxSeed -1, np.mean(meantDCSReach5SmoIdx),linewidth=1,edgecolor='black',facecolor='lightgrey')
reach5SmoIdxPlot.add_patch(meanAllSeedstDCSReach5SmoIdx)












"BACKWARD REACHING INDEX"
fig2   = plt.figure("BACKWARD REACHING INDEX", figsize=(20,10))
reach2 = plt.figtext(.20, 0.90, "REACH 2" , style='normal', bbox={'facecolor':'orangered'})
reach4 = plt.figtext(.48, 0.90, "REACH 4" , style='normal', bbox={'facecolor':'orangered'})
reach6 = plt.figtext(.76, 0.90, "REACH 6" , style='normal', bbox={'facecolor':'orangered'})

LinIdx = plt.figtext(.01, 0.77, "LIN IDX" , style='normal', bbox={'facecolor':'orangered'})
Asyidx = plt.figtext(.01, 0.49, "ASY IDX" , style='normal', bbox={'facecolor':'orangered'})
SmoIdx = plt.figtext(.01, 0.21, "SMO IDX" , style='normal', bbox={'facecolor':'orangered'})



"PLOT REACH 2 LINEARITY INDEX mean&std in normal and ataxia"
reach2LinIdxPlot = fig2.add_subplot(331)

reach2LinIdxPlot.set_ylim([0,100.0])
reach2LinIdxPlot.set_xlim([0,(maxSeed *3) +1])
reach2LinIdxPlot.set_yticks(np.arange(0, 100.0, 10.0))
reach2LinIdxPlot.set_xticks(np.arange(1, (maxSeed *3) +1, 1))


reach2LinIdxPlot.errorbar(np.linspace(1,maxSeed,maxSeed), meanNormalReach2LinIdx, yerr = stdNormalReach2LinIdx , marker= '^' , mec ='black', mfc='white' , ecolor ='black', fmt = '')
meanAllSeedsNormalReach2LinIdx = patches.Rectangle((1,0), maxSeed -1,np.mean(meanNormalReach2LinIdx),linewidth=1,edgecolor='black',facecolor='lightgrey')
reach2LinIdxPlot.add_patch(meanAllSeedsNormalReach2LinIdx)

reach2LinIdxPlot.errorbar(np.linspace((maxSeed +1),maxSeed*2,maxSeed), meanAtaxiaReach2LinIdx,  yerr = stdAtaxiaReach2LinIdx, marker= '^', mec ='black', mfc='black', ecolor ='black', fmt = '')
meanAllSeedsAtaxiaReach2LinIdx = patches.Rectangle((maxSeed +1,0), maxSeed -1, np.mean(meanAtaxiaReach2LinIdx),linewidth=1,edgecolor='black',facecolor='lightgrey')
reach2LinIdxPlot.add_patch(meanAllSeedsAtaxiaReach2LinIdx)

reach2LinIdxPlot.errorbar(np.linspace((maxSeed*2 +1),maxSeed*3,maxSeed), meantDCSReach2LinIdx,  yerr = stdtDCSReach2LinIdx, marker= '^', mec ='red', mfc='black', ecolor ='black', fmt = '')
meanAllSeedstDCSReach2LinIdx = patches.Rectangle((maxSeed*2 +1,0), maxSeed -1, np.mean(meantDCSReach2LinIdx),linewidth=1,edgecolor='black',facecolor='lightgrey')
reach2LinIdxPlot.add_patch(meanAllSeedstDCSReach2LinIdx)



"PLOT REACH 4 LINEARITY INDEX mean&std in normal and ataxia"
reach4LinIdxPlot = fig2.add_subplot(332)

reach4LinIdxPlot.set_ylim([0,100.0])
reach4LinIdxPlot.set_xlim([0,(maxSeed *3) +1])
reach4LinIdxPlot.set_yticks(np.arange(0, 100.0, 10.0))
reach4LinIdxPlot.set_xticks(np.arange(1, (maxSeed *3) +1, 1))

reach4LinIdxPlot.errorbar(np.linspace(1,maxSeed,maxSeed), meanNormalReach4LinIdx, yerr = stdNormalReach4LinIdx , marker= '^' , mec ='black', mfc='white' ,ecolor ='black',  fmt = '')
meanAllSeedsNormalReach4LinIdx = patches.Rectangle((1,0), maxSeed -1,np.mean(meanNormalReach4LinIdx),linewidth=1,edgecolor='black',facecolor='lightgrey')
reach4LinIdxPlot.add_patch(meanAllSeedsNormalReach4LinIdx)

reach4LinIdxPlot.errorbar(np.linspace((maxSeed +1),maxSeed*2,maxSeed), meanAtaxiaReach4LinIdx,  yerr = stdAtaxiaReach4LinIdx, marker= '^', mec ='black', mfc='black', ecolor ='black', fmt = '')
meanAllSeedsAtaxiaReach4LinIdx = patches.Rectangle((maxSeed +1,0), maxSeed -1, np.mean(meanAtaxiaReach4LinIdx),linewidth=1,edgecolor='black',facecolor='lightgrey')
reach4LinIdxPlot.add_patch(meanAllSeedsAtaxiaReach4LinIdx)

reach4LinIdxPlot.errorbar(np.linspace((maxSeed*2 +1),maxSeed*3,maxSeed), meantDCSReach4LinIdx,  yerr = stdtDCSReach4LinIdx, marker= '^', mec ='red', mfc='black', ecolor ='black', fmt = '')
meanAllSeedstDCSReach4LinIdx = patches.Rectangle((maxSeed*2 +1,0), maxSeed -1, np.mean(meantDCSReach4LinIdx),linewidth=1,edgecolor='black',facecolor='lightgrey')
reach4LinIdxPlot.add_patch(meanAllSeedstDCSReach4LinIdx)



"PLOT REACH 6 LINEARITY INDEX mean&std in normal and ataxia"
reach6LinIdxPlot = fig2.add_subplot(333)

reach6LinIdxPlot.set_ylim([0,100.0])
reach6LinIdxPlot.set_xlim([0,(maxSeed *3) +1])
reach6LinIdxPlot.set_yticks(np.arange(0, 100.0, 10.0))
reach6LinIdxPlot.set_xticks(np.arange(1, (maxSeed *3) +1, 1))

reach6LinIdxPlot.errorbar(np.linspace(1,maxSeed,maxSeed), meanNormalReach6LinIdx, yerr = stdNormalReach6LinIdx , marker= '^' , mec ='black', mfc='white' , ecolor ='black', fmt = '')
meanAllSeedsNormalReach6LinIdx = patches.Rectangle((1,0), maxSeed -1,np.mean(meanNormalReach6LinIdx),linewidth=1,edgecolor='black',facecolor='lightgrey')
reach6LinIdxPlot.add_patch(meanAllSeedsNormalReach6LinIdx)

reach6LinIdxPlot.errorbar(np.linspace((maxSeed +1),maxSeed*2,maxSeed), meanAtaxiaReach6LinIdx,  yerr = stdAtaxiaReach6LinIdx, marker= '^', mec ='black', mfc='black', ecolor ='black', fmt = '')
meanAllSeedsAtaxiaReach6LinIdx = patches.Rectangle((maxSeed +1,0), maxSeed -1, np.mean(meanAtaxiaReach6LinIdx),linewidth=1,edgecolor='black',facecolor='lightgrey')
reach6LinIdxPlot.add_patch(meanAllSeedsAtaxiaReach6LinIdx)

reach6LinIdxPlot.errorbar(np.linspace((maxSeed*2 +1),maxSeed*3,maxSeed), meantDCSReach6LinIdx,  yerr = stdtDCSReach6LinIdx, marker= '^', mec ='red', mfc='black', ecolor ='black', fmt = '')
meanAllSeedstDCSReach6LinIdx = patches.Rectangle((maxSeed*2 +1,0), maxSeed -1, np.mean(meantDCSReach6LinIdx),linewidth=1,edgecolor='black',facecolor='lightgrey')
reach6LinIdxPlot.add_patch(meanAllSeedstDCSReach6LinIdx)




"PLOT REACH 2 ASYMMETRY INDEX mean&std in normal and ataxia"
reach2AsyIdxPlot = fig2.add_subplot(334)

reach2AsyIdxPlot.set_ylim([0,7.0])
reach2AsyIdxPlot.set_xlim([0,(maxSeed *3) +1])
reach2AsyIdxPlot.set_yticks(np.arange(0, 7.0, 0.5))
reach2AsyIdxPlot.set_xticks(np.arange(1, (maxSeed *3) +1, 1))



reach2AsyIdxPlot.errorbar(np.linspace(1,maxSeed,maxSeed), meanNormalReach2AsyIdx, yerr = stdNormalReach2AsyIdx , marker= '^' , mec ='black', mfc='white' , ecolor ='black', fmt = '')
meanAllSeedsNormalReach2AsyIdx = patches.Rectangle((1,0), maxSeed -1 , np.mean(meanNormalReach2AsyIdx) ,linewidth=1 , edgecolor='black' , facecolor='lightgrey')
reach2AsyIdxPlot.add_patch(meanAllSeedsNormalReach2AsyIdx)

reach2AsyIdxPlot.errorbar(np.linspace((maxSeed +1),maxSeed*2,maxSeed), meanAtaxiaReach2AsyIdx,  yerr = stdAtaxiaReach2AsyIdx, marker= '^', mec ='black', mfc='black', ecolor ='black', fmt = '')
meanAllSeedsAtaxiaReach2AsyIdx = patches.Rectangle((maxSeed +1,0), maxSeed -1, np.mean(meanAtaxiaReach2AsyIdx),linewidth=1,edgecolor='black',facecolor='lightgrey')
reach2AsyIdxPlot.add_patch(meanAllSeedsAtaxiaReach2AsyIdx)

reach2AsyIdxPlot.errorbar(np.linspace((maxSeed*2 +1),maxSeed*3,maxSeed), meantDCSReach2AsyIdx,  yerr = stdtDCSReach2AsyIdx, marker= '^', mec ='red', mfc='black', ecolor ='black', fmt = '')
meanAllSeedstDCSReach2AsyIdx = patches.Rectangle((maxSeed*2 +1,0), maxSeed -1, np.mean(meantDCSReach2AsyIdx),linewidth=1,edgecolor='black',facecolor='lightgrey')
reach2AsyIdxPlot.add_patch(meanAllSeedstDCSReach2AsyIdx)

"PLOT REACH 3 ASYMMETRY INDEX mean&std in normal and ataxia"
reach4AsyIdxPlot = fig2.add_subplot(335)

reach4AsyIdxPlot.set_ylim([0,7.0])
reach4AsyIdxPlot.set_xlim([0,(maxSeed *3) +1])
reach4AsyIdxPlot.set_yticks(np.arange(0, 7.0, 0.5))
reach4AsyIdxPlot.set_xticks(np.arange(1, (maxSeed *3) +1, 1))


reach4AsyIdxPlot.errorbar(np.linspace(1,maxSeed,maxSeed), meanNormalReach4AsyIdx, yerr = stdNormalReach4AsyIdx , marker= '^' , mec ='black', mfc='white' , ecolor ='black', fmt = '')
meanAllSeedsNormalReach4AsyIdx = patches.Rectangle((1,0), maxSeed -1 , np.mean(meanNormalReach4AsyIdx) ,linewidth=1 , edgecolor='black' , facecolor='lightgrey')
reach4AsyIdxPlot.add_patch(meanAllSeedsNormalReach4AsyIdx)

reach4AsyIdxPlot.errorbar(np.linspace((maxSeed +1),maxSeed*2,maxSeed), meanAtaxiaReach4AsyIdx,  yerr = stdAtaxiaReach4AsyIdx, marker= '^', mec ='black', mfc='black', ecolor ='black', fmt = '')
meanAllSeedsAtaxiaReach4AsyIdx = patches.Rectangle((maxSeed +1,0), maxSeed -1, np.mean(meanAtaxiaReach4AsyIdx),linewidth=1,edgecolor='black',facecolor='lightgrey')
reach4AsyIdxPlot.add_patch(meanAllSeedsAtaxiaReach4AsyIdx)

reach4AsyIdxPlot.errorbar(np.linspace((maxSeed*2 +1),maxSeed*3,maxSeed), meantDCSReach4AsyIdx,  yerr = stdtDCSReach4AsyIdx, marker= '^', mec ='red', mfc='black', ecolor ='black', fmt = '')
meanAllSeedstDCSReach4AsyIdx = patches.Rectangle((maxSeed*2 +1,0), maxSeed -1, np.mean(meantDCSReach4AsyIdx),linewidth=1,edgecolor='black',facecolor='lightgrey')
reach4AsyIdxPlot.add_patch(meanAllSeedstDCSReach4AsyIdx)

"PLOT REACH 5 ASYMMETRY INDEX mean&std in normal and ataxia"
reach6AsyIdxPlot = fig2.add_subplot(336)

reach6AsyIdxPlot.set_ylim([0,7.0])
reach6AsyIdxPlot.set_xlim([0,(maxSeed *3) +1])
reach6AsyIdxPlot.set_yticks(np.arange(0, 7.0, 0.5))
reach6AsyIdxPlot.set_xticks(np.arange(1, (maxSeed *3) +1, 1))


reach6AsyIdxPlot.errorbar(np.linspace(1,maxSeed,maxSeed), meanNormalReach6AsyIdx, yerr = stdNormalReach6AsyIdx , marker= '^' , mec ='black', mfc='white' , ecolor ='black', fmt = '')
meanAllSeedsNormalReach6AsyIdx = patches.Rectangle((1,0), maxSeed -1 , np.mean(meanNormalReach6AsyIdx) ,linewidth=1 , edgecolor='black' , facecolor='lightgrey')
reach6AsyIdxPlot.add_patch(meanAllSeedsNormalReach6AsyIdx)

reach6AsyIdxPlot.errorbar(np.linspace((maxSeed +1),maxSeed*2,maxSeed), meanAtaxiaReach6AsyIdx,  yerr = stdAtaxiaReach6AsyIdx, marker= '^', mec ='black', mfc='black', ecolor ='black', fmt = '')
meanAllSeedsAtaxiaReach6AsyIdx = patches.Rectangle((maxSeed +1,0), maxSeed -1, np.mean(meanAtaxiaReach6AsyIdx),linewidth=1,edgecolor='black',facecolor='lightgrey')
reach6AsyIdxPlot.add_patch(meanAllSeedsAtaxiaReach6AsyIdx)

reach6AsyIdxPlot.errorbar(np.linspace((maxSeed*2 +1),maxSeed*3,maxSeed), meantDCSReach6AsyIdx,  yerr = stdtDCSReach6AsyIdx, marker= '^', mec ='red', mfc='black', ecolor ='black', fmt = '')
meanAllSeedstDCSReach6AsyIdx = patches.Rectangle((maxSeed*2 +1,0), maxSeed -1, np.mean(meantDCSReach6AsyIdx),linewidth=1,edgecolor='black',facecolor='lightgrey')
reach6AsyIdxPlot.add_patch(meanAllSeedstDCSReach6AsyIdx)





"PLOT REACH 2 SMOOTHNESS INDEX mean&std in normal and ataxia"
reach2SmoIdxPlot = fig2.add_subplot(337)

reach2SmoIdxPlot.set_ylim([0,15])
reach2SmoIdxPlot.set_xlim([0,(maxSeed *3) +1])
reach2SmoIdxPlot.set_yticks(np.arange(0, 15, 5))
reach2SmoIdxPlot.set_xticks(np.arange(1, (maxSeed *3) +1, 1))


reach2SmoIdxPlot.errorbar(np.linspace(1,maxSeed,maxSeed), meanNormalReach2SmoIdx, yerr = stdNormalReach2SmoIdx , marker= '^' , mec ='black', mfc='white' , ecolor ='black', fmt = '')
meanAllSeedsNormalReach2SmoIdx = patches.Rectangle((1,0), maxSeed -1 , np.mean(meanNormalReach2SmoIdx) ,linewidth=1 , edgecolor='black' , facecolor='lightgrey')
reach2SmoIdxPlot.add_patch(meanAllSeedsNormalReach2SmoIdx)

reach2SmoIdxPlot.errorbar(np.linspace((maxSeed +1),maxSeed*2,maxSeed), meanAtaxiaReach2SmoIdx,  yerr = stdAtaxiaReach2SmoIdx, marker= '^', mec ='black', mfc='black', ecolor ='black', fmt = '')
meanAllSeedsAtaxiaReach2SmoIdx = patches.Rectangle((maxSeed +1,0), maxSeed -1, np.mean(meanAtaxiaReach2SmoIdx),linewidth=1,edgecolor='black',facecolor='lightgrey')
reach2SmoIdxPlot.add_patch(meanAllSeedsAtaxiaReach2SmoIdx)

reach2SmoIdxPlot.errorbar(np.linspace((maxSeed*2 +1),maxSeed*3,maxSeed), meantDCSReach2SmoIdx,  yerr = stdtDCSReach2SmoIdx, marker= '^', mec ='red', mfc='black', ecolor ='black', fmt = '')
meanAllSeedstDCSReach2SmoIdx = patches.Rectangle((maxSeed*2 +1,0), maxSeed -1, np.mean(meantDCSReach2SmoIdx),linewidth=1,edgecolor='black',facecolor='lightgrey')
reach2SmoIdxPlot.add_patch(meanAllSeedstDCSReach2SmoIdx)

"PLOT REACH 4 SMOOTHNESS INDEX mean&std in normal and ataxia"
reach4SmoIdxPlot = fig2.add_subplot(338)

reach4SmoIdxPlot.set_ylim([0,15])
reach4SmoIdxPlot.set_xlim([0,(maxSeed *3) +1])
reach4SmoIdxPlot.set_yticks(np.arange(0, 15, 5))
reach4SmoIdxPlot.set_xticks(np.arange(1, (maxSeed *3) +1, 1))

reach4SmoIdxPlot.errorbar(np.linspace(1,maxSeed,maxSeed), meanNormalReach4SmoIdx, yerr = stdNormalReach4SmoIdx , marker= '^' , mec ='black', mfc='white' , ecolor ='black', fmt = '')
meanAllSeedsNormalReach4SmoIdx = patches.Rectangle((1,0), maxSeed -1 , np.mean(meanNormalReach4SmoIdx) ,linewidth=1 , edgecolor='black' , facecolor='lightgrey')
reach4SmoIdxPlot.add_patch(meanAllSeedsNormalReach4SmoIdx)

reach4SmoIdxPlot.errorbar(np.linspace((maxSeed +1),maxSeed*2,maxSeed), meanAtaxiaReach4SmoIdx,  yerr = stdAtaxiaReach4SmoIdx, marker= '^', mec ='black', mfc='black', ecolor ='black', fmt = '')
meanAllSeedsAtaxiaReach4SmoIdx = patches.Rectangle((maxSeed +1,0), maxSeed -1, np.mean(meanAtaxiaReach4SmoIdx),linewidth=1,edgecolor='black',facecolor='lightgrey')
reach4SmoIdxPlot.add_patch(meanAllSeedsAtaxiaReach4SmoIdx)

reach4SmoIdxPlot.errorbar(np.linspace((maxSeed*2 +1),maxSeed*3,maxSeed), meantDCSReach4SmoIdx,  yerr = stdtDCSReach4SmoIdx, marker= '^', mec ='red', mfc='black', ecolor ='black', fmt = '')
meanAllSeedstDCSReach4SmoIdx = patches.Rectangle((maxSeed*2 +1,0), maxSeed -1, np.mean(meantDCSReach4SmoIdx),linewidth=1,edgecolor='black',facecolor='lightgrey')
reach4SmoIdxPlot.add_patch(meanAllSeedstDCSReach4SmoIdx)

"PLOT REACH 6 SMOOTHNESS INDEX mean&std in normal and ataxia"
reach6SmoIdxPlot = fig2.add_subplot(339)

reach6SmoIdxPlot.set_ylim([0,15])
reach6SmoIdxPlot.set_xlim([0,(maxSeed *3) +1])
reach6SmoIdxPlot.set_yticks(np.arange(0, 15, 5))
reach6SmoIdxPlot.set_xticks(np.arange(1, (maxSeed *3) +1, 1))

reach6SmoIdxPlot.errorbar(np.linspace(1,maxSeed,maxSeed), meanNormalReach6SmoIdx, yerr = stdNormalReach6SmoIdx , marker= '^' , mec ='black', mfc='white' , ecolor ='black', fmt = '')
meanAllSeedsNormalReach6SmoIdx = patches.Rectangle((1,0), maxSeed -1 , np.mean(meanNormalReach6SmoIdx) ,linewidth=1 , edgecolor='black' , facecolor='lightgrey')
reach6SmoIdxPlot.add_patch(meanAllSeedsNormalReach6SmoIdx)

reach6SmoIdxPlot.errorbar(np.linspace((maxSeed +1),maxSeed*2,maxSeed), meanAtaxiaReach6SmoIdx,  yerr = stdAtaxiaReach6SmoIdx, marker= '^', mec ='black', mfc='black', ecolor ='black', fmt = '')
meanAllSeedsAtaxiaReach6SmoIdx = patches.Rectangle((maxSeed +1,0), maxSeed -1, np.mean(meanAtaxiaReach6SmoIdx),linewidth=1,edgecolor='black',facecolor='lightgrey')
reach6SmoIdxPlot.add_patch(meanAllSeedsAtaxiaReach6SmoIdx)

reach6SmoIdxPlot.errorbar(np.linspace((maxSeed*2 +1),maxSeed*3,maxSeed), meantDCSReach6SmoIdx,  yerr = stdtDCSReach6SmoIdx, marker= '^', mec ='red', mfc='black', ecolor ='black', fmt = '')
meanAllSeedstDCSReach6SmoIdx = patches.Rectangle((maxSeed*2 +1,0), maxSeed -1, np.mean(meantDCSReach6SmoIdx),linewidth=1,edgecolor='black',facecolor='lightgrey')
reach6SmoIdxPlot.add_patch(meanAllSeedstDCSReach6SmoIdx)











fig3   = plt.figure("FORWARD AVG INDEX", figsize=(20,10))

LinIdx = plt.figtext(.01, 0.77, "LIN IDX" , style='normal', bbox={'facecolor':'orangered'})
Asyidx = plt.figtext(.01, 0.49, "ASY IDX" , style='normal', bbox={'facecolor':'orangered'})
SmoIdx = plt.figtext(.01, 0.21, "SMO IDX" , style='normal', bbox={'facecolor':'orangered'})


"FORWARD AVG LIN IDX"
forwardLinIdxAvgPlot = fig3.add_subplot(311)
forwardLinIdxAvgPlot.set_ylim([0,100.0])
forwardLinIdxAvgPlot.set_xlim([0,(maxSeed *3) +1])
forwardLinIdxAvgPlot.set_yticks(np.arange(0, 100.0, 10.0))
forwardLinIdxAvgPlot.set_xticks(np.arange(1, (maxSeed *3) +1, 1))


forwardLinIdxAvgPlot.errorbar(np.linspace(1,maxSeed,maxSeed), meanNormalForwardReachingsLinIdx , yerr = stdNormalForwardReachingsLinIdx, marker= '^' , mec ='black', mfc='white' , ecolor ='black', fmt = '')
meanAllSeedsNormalForwardLinIdx = patches.Rectangle((1,0), maxSeed -1,np.mean(meanNormalForwardReachingsLinIdx),linewidth=1,edgecolor='black',facecolor='lightgrey')
forwardLinIdxAvgPlot.add_patch(meanAllSeedsNormalForwardLinIdx)

forwardLinIdxAvgPlot.errorbar(np.linspace((maxSeed +1),maxSeed*2,maxSeed), meanAtaxiaForwardReachingsLinIdx,  yerr = stdAtaxiaForwardReachingsLinIdx, marker= '^', mec ='black', mfc='black', ecolor ='black', fmt = '')
meanAllSeedsAtaxiaForwardLinIdx = patches.Rectangle((maxSeed +1,0), maxSeed -1, np.mean(meanAtaxiaForwardReachingsLinIdx),linewidth=1,edgecolor='black',facecolor='lightgrey')
forwardLinIdxAvgPlot.add_patch(meanAllSeedsAtaxiaForwardLinIdx)

forwardLinIdxAvgPlot.errorbar(np.linspace((maxSeed*2 +1),maxSeed*3,maxSeed), meantDCSForwardReachingsLinIdx,  yerr = stdtDCSForwardReachingsLinIdx, marker= '^', mec ='red', mfc='black', ecolor ='black', fmt = '')
meanAllSeedstDCSForwardLinIdx = patches.Rectangle((maxSeed*2 +1,0), maxSeed -1, np.mean(meantDCSForwardReachingsLinIdx),linewidth=1,edgecolor='black',facecolor='lightgrey')
forwardLinIdxAvgPlot.add_patch(meanAllSeedstDCSForwardLinIdx)


"FORWARD AVG ASY IDX"
forwardAsyIdxAvgPlot = fig3.add_subplot(312)

forwardAsyIdxAvgPlot.set_ylim([0,7.0])
forwardAsyIdxAvgPlot.set_xlim([0,(maxSeed *3) +1])
forwardAsyIdxAvgPlot.set_yticks(np.arange(0, 7.0, 0.5))
forwardAsyIdxAvgPlot.set_xticks(np.arange(1, (maxSeed *3) +1, 1))

forwardAsyIdxAvgPlot.errorbar(np.linspace(1,maxSeed,maxSeed), meanNormalForwardReachingsAsyIdx , yerr = stdNormalForwardReachingsAsyIdx, marker= '^' , mec ='black', mfc='white' , ecolor ='black', fmt = '')
meanAllSeedsNormalForwardAsyIdx = patches.Rectangle((1,0), maxSeed -1,np.mean(meanNormalForwardReachingsAsyIdx),linewidth=1,edgecolor='black',facecolor='lightgrey')
forwardAsyIdxAvgPlot.add_patch(meanAllSeedsNormalForwardAsyIdx)

forwardAsyIdxAvgPlot.errorbar(np.linspace((maxSeed +1),maxSeed*2,maxSeed), meanAtaxiaForwardReachingsAsyIdx,  yerr = stdAtaxiaForwardReachingsAsyIdx, marker= '^', mec ='black', mfc='black', ecolor ='black', fmt = '')
meanAllSeedsAtaxiaForwardAsyIdx = patches.Rectangle((maxSeed +1,0), maxSeed -1, np.mean(meanAtaxiaForwardReachingsAsyIdx),linewidth=1,edgecolor='black',facecolor='lightgrey')
forwardAsyIdxAvgPlot.add_patch(meanAllSeedsAtaxiaForwardAsyIdx)

forwardAsyIdxAvgPlot.errorbar(np.linspace((maxSeed*2 +1),maxSeed*3,maxSeed), meantDCSForwardReachingsAsyIdx,  yerr = stdtDCSForwardReachingsAsyIdx, marker= '^', mec ='red', mfc='black', ecolor ='black', fmt = '')
meanAllSeedstDCSForwardAsyIdx = patches.Rectangle((maxSeed*2 +1,0), maxSeed -1, np.mean(meantDCSForwardReachingsAsyIdx),linewidth=1,edgecolor='black',facecolor='lightgrey')
forwardAsyIdxAvgPlot.add_patch(meanAllSeedstDCSForwardAsyIdx)


"FORWARD AVG SMO IDX"
forwardSmoIdxAvgPlot = fig3.add_subplot(313)

forwardSmoIdxAvgPlot.set_ylim([0,15.0])
forwardSmoIdxAvgPlot.set_xlim([0,(maxSeed *3) +1])
forwardSmoIdxAvgPlot.set_yticks(np.arange(0, 15.0, 5))
forwardSmoIdxAvgPlot.set_xticks(np.arange(1, (maxSeed *3) +1, 1))

forwardSmoIdxAvgPlot.errorbar(np.linspace(1,maxSeed,maxSeed), meanNormalForwardReachingsSmoIdx , yerr = stdNormalForwardReachingsSmoIdx, marker= '^' , mec ='black', mfc='white' , ecolor ='black', fmt = '')
meanAllSeedsNormalForwardSmoIdx = patches.Rectangle((1,0), maxSeed -1,np.mean(meanNormalForwardReachingsSmoIdx),linewidth=1,edgecolor='black',facecolor='lightgrey')
forwardSmoIdxAvgPlot.add_patch(meanAllSeedsNormalForwardSmoIdx)

forwardSmoIdxAvgPlot.errorbar(np.linspace((maxSeed +1),maxSeed*2,maxSeed), meanAtaxiaForwardReachingsSmoIdx,  yerr = stdAtaxiaForwardReachingsSmoIdx, marker= '^', mec ='black', mfc='black', ecolor ='black', fmt = '')
meanAllSeedsAtaxiaForwardSmoIdx = patches.Rectangle((maxSeed +1,0), maxSeed -1, np.mean(meanAtaxiaForwardReachingsSmoIdx),linewidth=1,edgecolor='black',facecolor='lightgrey')
forwardSmoIdxAvgPlot.add_patch(meanAllSeedsAtaxiaForwardSmoIdx)

forwardSmoIdxAvgPlot.errorbar(np.linspace((maxSeed*2 +1),maxSeed*3,maxSeed), meantDCSForwardReachingsSmoIdx,  yerr = stdtDCSForwardReachingsSmoIdx, marker= '^', mec ='red', mfc='black', ecolor ='black', fmt = '')
meanAllSeedstDCSForwardSmoIdx = patches.Rectangle((maxSeed*2 +1,0), maxSeed -1, np.mean(meantDCSForwardReachingsSmoIdx),linewidth=1,edgecolor='black',facecolor='lightgrey')
forwardSmoIdxAvgPlot.add_patch(meanAllSeedstDCSForwardSmoIdx)





fig4   = plt.figure("BACKWARD AVG INDEX", figsize=(20,10))

LinIdx = plt.figtext(.01, 0.77, "LIN IDX" , style='normal', bbox={'facecolor':'orangered'})
Asyidx = plt.figtext(.01, 0.49, "ASY IDX" , style='normal', bbox={'facecolor':'orangered'})
SmoIdx = plt.figtext(.01, 0.21, "SMO IDX" , style='normal', bbox={'facecolor':'orangered'})


"BACKWARD AVG LIN IDX"
backwardLinIdxAvgPlot = fig4.add_subplot(311)
backwardLinIdxAvgPlot.set_ylim([0,100.0])
backwardLinIdxAvgPlot.set_xlim([0,(maxSeed *3) +1])
backwardLinIdxAvgPlot.set_yticks(np.arange(0, 100.0, 10.0))
backwardLinIdxAvgPlot.set_xticks(np.arange(1, (maxSeed *3) +1, 1))


backwardLinIdxAvgPlot.errorbar(np.linspace(1,maxSeed,maxSeed), meanNormalBackwardReachingsLinIdx , yerr = stdNormalBackwardReachingsLinIdx, marker= '^' , mec ='black', mfc='white' , ecolor ='black', fmt = '')
meanAllSeedsNormalBackwardLinIdx = patches.Rectangle((1,0), maxSeed -1,np.mean(meanNormalBackwardReachingsLinIdx),linewidth=1,edgecolor='black',facecolor='lightgrey')
backwardLinIdxAvgPlot.add_patch(meanAllSeedsNormalBackwardLinIdx)

backwardLinIdxAvgPlot.errorbar(np.linspace((maxSeed +1),maxSeed*2,maxSeed), meanAtaxiaBackwardReachingsLinIdx,  yerr = stdAtaxiaBackwardReachingsLinIdx, marker= '^', mec ='black', mfc='black', ecolor ='black', fmt = '')
meanAllSeedsAtaxiaBackwardLinIdx = patches.Rectangle((maxSeed +1,0), maxSeed -1, np.mean(meanAtaxiaBackwardReachingsLinIdx),linewidth=1,edgecolor='black',facecolor='lightgrey')
backwardLinIdxAvgPlot.add_patch(meanAllSeedsAtaxiaBackwardLinIdx)

backwardLinIdxAvgPlot.errorbar(np.linspace((maxSeed*2 +1),maxSeed*3,maxSeed), meantDCSBackwardReachingsLinIdx,  yerr = stdtDCSBackwardReachingsLinIdx, marker= '^', mec ='red', mfc='black', ecolor ='black', fmt = '')
meanAllSeedstDCSBackwardLinIdx = patches.Rectangle((maxSeed*2 +1,0), maxSeed -1, np.mean(meantDCSBackwardReachingsLinIdx),linewidth=1,edgecolor='black',facecolor='lightgrey')
backwardLinIdxAvgPlot.add_patch(meanAllSeedstDCSBackwardLinIdx)


"BACKWARD AVG ASY IDX"
backwardAsyIdxAvgPlot = fig4.add_subplot(312)

backwardAsyIdxAvgPlot.set_ylim([0,7.0])
backwardAsyIdxAvgPlot.set_xlim([0,(maxSeed *3) +1])
backwardAsyIdxAvgPlot.set_yticks(np.arange(0, 7.0, 0.5))
backwardAsyIdxAvgPlot.set_xticks(np.arange(1, (maxSeed *3) +1, 1))

backwardAsyIdxAvgPlot.errorbar(np.linspace(1,maxSeed,maxSeed), meanNormalBackwardReachingsAsyIdx , yerr = stdNormalBackwardReachingsAsyIdx, marker= '^' , mec ='black', mfc='white' , ecolor ='black', fmt = '')
meanAllSeedsNormalBackwardAsyIdx = patches.Rectangle((1,0), maxSeed -1,np.mean(meanNormalBackwardReachingsAsyIdx),linewidth=1,edgecolor='black',facecolor='lightgrey')
backwardAsyIdxAvgPlot.add_patch(meanAllSeedsNormalBackwardAsyIdx)

backwardAsyIdxAvgPlot.errorbar(np.linspace((maxSeed +1),maxSeed*2,maxSeed), meanAtaxiaBackwardReachingsAsyIdx,  yerr = stdAtaxiaBackwardReachingsAsyIdx, marker= '^', mec ='black', mfc='black', ecolor ='black', fmt = '')
meanAllSeedsAtaxiaBackwardAsyIdx = patches.Rectangle((maxSeed +1,0), maxSeed -1, np.mean(meanAtaxiaBackwardReachingsAsyIdx),linewidth=1,edgecolor='black',facecolor='lightgrey')
backwardAsyIdxAvgPlot.add_patch(meanAllSeedsAtaxiaBackwardAsyIdx)

backwardAsyIdxAvgPlot.errorbar(np.linspace((maxSeed*2 +1),maxSeed*3,maxSeed), meantDCSBackwardReachingsAsyIdx,  yerr = stdtDCSBackwardReachingsAsyIdx, marker= '^', mec ='red', mfc='black', ecolor ='black', fmt = '')
meanAllSeedstDCSBackwardAsyIdx = patches.Rectangle((maxSeed*2 +1,0), maxSeed -1, np.mean(meantDCSBackwardReachingsAsyIdx),linewidth=1,edgecolor='black',facecolor='lightgrey')
backwardAsyIdxAvgPlot.add_patch(meanAllSeedstDCSBackwardAsyIdx)


"BACKWARD AVG SMO IDX"
backwardSmoIdxAvgPlot = fig4.add_subplot(313)

backwardSmoIdxAvgPlot.set_ylim([0,15.0])
backwardSmoIdxAvgPlot.set_xlim([0,(maxSeed *3) +1])
backwardSmoIdxAvgPlot.set_yticks(np.arange(0, 15.0, 5))
backwardSmoIdxAvgPlot.set_xticks(np.arange(1, (maxSeed *3) +1, 1))

backwardSmoIdxAvgPlot.errorbar(np.linspace(1,maxSeed,maxSeed), meanNormalBackwardReachingsSmoIdx , yerr = stdNormalBackwardReachingsSmoIdx, marker= '^' , mec ='black', mfc='white' , ecolor ='black', fmt = '')
meanAllSeedsNormalBackwardSmoIdx = patches.Rectangle((1,0), maxSeed -1,np.mean(meanNormalBackwardReachingsSmoIdx),linewidth=1,edgecolor='black',facecolor='lightgrey')
backwardSmoIdxAvgPlot.add_patch(meanAllSeedsNormalBackwardSmoIdx)

backwardSmoIdxAvgPlot.errorbar(np.linspace((maxSeed +1),maxSeed*2,maxSeed), meanAtaxiaBackwardReachingsSmoIdx,  yerr = stdAtaxiaBackwardReachingsSmoIdx, marker= '^', mec ='black', mfc='black', ecolor ='black', fmt = '')
meanAllSeedsAtaxiaBackwardSmoIdx = patches.Rectangle((maxSeed +1,0), maxSeed -1, np.mean(meanAtaxiaBackwardReachingsSmoIdx),linewidth=1,edgecolor='black',facecolor='lightgrey')
backwardSmoIdxAvgPlot.add_patch(meanAllSeedsAtaxiaBackwardSmoIdx)

backwardSmoIdxAvgPlot.errorbar(np.linspace((maxSeed*2 +1),maxSeed*3,maxSeed), meantDCSBackwardReachingsSmoIdx,  yerr = stdtDCSBackwardReachingsSmoIdx, marker= '^', mec ='red', mfc='black', ecolor ='black', fmt = '')
meanAllSeedstDCSBackwardSmoIdx = patches.Rectangle((maxSeed*2 +1,0), maxSeed -1, np.mean(meantDCSBackwardReachingsSmoIdx),linewidth=1,edgecolor='black',facecolor='lightgrey')
backwardSmoIdxAvgPlot.add_patch(meanAllSeedstDCSBackwardSmoIdx)

"""





fig5  = plt.figure("ALL REACHINGS AVG INDEX", figsize=(20,10))

LinIdx = plt.figtext(.01, 0.77, "LIN IDX" , style='normal', bbox={'facecolor':'orangered'})
Asyidx = plt.figtext(.01, 0.49, "ASY IDX" , style='normal', bbox={'facecolor':'orangered'})
SmoIdx = plt.figtext(.01, 0.21, "SMO IDX" , style='normal', bbox={'facecolor':'orangered'})


"ALL REACHINGS AVG LIN IDX"
allLinIdxAvgPlot = fig5.add_subplot(311)
allLinIdxAvgPlot.set_ylim([0,100.0])
allLinIdxAvgPlot.set_xlim([0,(maxSeed *3) +1])
allLinIdxAvgPlot.set_yticks(np.arange(0, 100.0, 10.0))
allLinIdxAvgPlot.set_xticks(np.arange(1, (maxSeed *3) +1, 1))



allLinIdxAvgPlot.errorbar(np.linspace(1,maxSeed,maxSeed), meanNormalAllReachingsLinIdx , yerr = stdNormalAllReachingsLinIdx, marker= '^' , mec ='black', mfc='white' , ecolor ='black', fmt = '', linestyle='none')
meanAllSeedsNormalAllLinIdx = patches.Rectangle((1,0), maxSeed -1,np.mean(meanNormalAllReachingsLinIdx),linewidth=1,edgecolor='black',facecolor='lightgrey')
allLinIdxAvgPlot.add_patch(meanAllSeedsNormalAllLinIdx)

target_normal_lin_idx_plot = plt.plot([1.0, maxSeed] , [target_normal_lin_idx,target_normal_lin_idx], color = 'red', linestyle=':')



allLinIdxAvgPlot.errorbar(np.linspace((maxSeed +1),maxSeed*2,maxSeed), meanAtaxiaAllReachingsLinIdx,  yerr = stdAtaxiaAllReachingsLinIdx, marker= '^', mec ='black', mfc='black', ecolor ='black', fmt = '', linestyle='none')
meanAllSeedsAtaxiaAllLinIdx = patches.Rectangle((maxSeed +1,0), maxSeed -1, np.mean(meanAtaxiaAllReachingsLinIdx),linewidth=1,edgecolor='black',facecolor='lightgrey')
allLinIdxAvgPlot.add_patch(meanAllSeedsAtaxiaAllLinIdx)

target_ataxia_lin_idx_plot = plt.plot([maxSeed +1, maxSeed *2] , [target_ataxia_lin_idx,target_ataxia_lin_idx], color = 'red', linestyle=':')



allLinIdxAvgPlot.errorbar(np.linspace((maxSeed*2 +1),maxSeed*3,maxSeed), meantDCSAllReachingsLinIdx,  yerr = stdtDCSAllReachingsLinIdx, marker= '^', mec ='red', mfc='black', ecolor ='black', fmt = '', linestyle='none')
meanAllSeedstDCSAllLinIdx = patches.Rectangle((maxSeed*2 +1,0), maxSeed -1, np.mean(meantDCSAllReachingsLinIdx),linewidth=1,edgecolor='black',facecolor='lightgrey')
allLinIdxAvgPlot.add_patch(meanAllSeedstDCSAllLinIdx)


lin_idx_p_value = round(stats.f_oneway(statNormalAllReachingsLinIdx.ravel() ,statAtaxiaAllReachingsLinIdx.ravel())[1],10)
lin_idx_fisher = round(stats.f_oneway(statNormalAllReachingsLinIdx.ravel(),statAtaxiaAllReachingsLinIdx.ravel())[0],10)

lin_idx_p_value_box = plt.figtext(0.35, 0.85, "p_value=%s" % (lin_idx_p_value), style='normal', bbox={'facecolor':'lightgrey'})
lin_idx_fisher_box = plt.figtext(0.35, 0.82, "fisher=%s" % (lin_idx_fisher), style='normal', bbox={'facecolor':'lightgrey'})

"ALL REACHINGS AVG ASY IDX"
allAsyIdxAvgPlot = fig5.add_subplot(312)

allAsyIdxAvgPlot.set_ylim([0,7.0])
allAsyIdxAvgPlot.set_xlim([0,(maxSeed *3) +1])
allAsyIdxAvgPlot.set_yticks(np.arange(0, 7.0, 0.5))
allAsyIdxAvgPlot.set_xticks(np.arange(1, (maxSeed *3) +1, 1))


allAsyIdxAvgPlot.errorbar(np.linspace(1,maxSeed,maxSeed), meanNormalAllReachingsAsyIdx , yerr = stdNormalAllReachingsAsyIdx, marker= '^' , mec ='black', mfc='white' , ecolor ='black', fmt = '', linestyle='none')
meanAllSeedsNormalAllAsyIdx = patches.Rectangle((1,0), maxSeed -1,np.mean(meanNormalAllReachingsAsyIdx),linewidth=1,edgecolor='black',facecolor='lightgrey')
allAsyIdxAvgPlot.add_patch(meanAllSeedsNormalAllAsyIdx)

target_normal_asy_idx_plot = plt.plot([1.0, maxSeed] , [target_normal_asy_idx,target_normal_asy_idx], color = 'red', linestyle=':')

allAsyIdxAvgPlot.errorbar(np.linspace((maxSeed +1),maxSeed*2,maxSeed), meanAtaxiaAllReachingsAsyIdx,  yerr = stdAtaxiaAllReachingsAsyIdx, marker= '^', mec ='black', mfc='black', ecolor ='black', fmt = '', linestyle='none')
meanAllSeedsAtaxiaAllAsyIdx = patches.Rectangle((maxSeed +1,0), maxSeed -1, np.mean(meanAtaxiaAllReachingsAsyIdx),linewidth=1,edgecolor='black',facecolor='lightgrey')
allAsyIdxAvgPlot.add_patch(meanAllSeedsAtaxiaAllAsyIdx)

target_ataxia_asy_idx_plot = plt.plot([maxSeed +1, maxSeed *2] , [target_ataxia_asy_idx,target_ataxia_asy_idx], color = 'red', linestyle=':')

allAsyIdxAvgPlot.errorbar(np.linspace((maxSeed*2 +1),maxSeed*3,maxSeed), meantDCSAllReachingsAsyIdx,  yerr = stdtDCSAllReachingsAsyIdx, marker= '^', mec ='red', mfc='black', ecolor ='black', fmt = '', linestyle='none')
meanAllSeedstDCSAllAsyIdx = patches.Rectangle((maxSeed*2 +1,0), maxSeed -1, np.mean(meantDCSAllReachingsAsyIdx),linewidth=1,edgecolor='black',facecolor='lightgrey')
allAsyIdxAvgPlot.add_patch(meanAllSeedstDCSAllAsyIdx)

asy_idx_p_value = round(stats.f_oneway(statNormalAllReachingsAsyIdx.ravel() ,statAtaxiaAllReachingsAsyIdx.ravel())[1],5)
asy_idx_fisher = round(stats.f_oneway(statNormalAllReachingsAsyIdx.ravel() ,statAtaxiaAllReachingsAsyIdx.ravel())[0],5)

asy_idx_p_value_box = plt.figtext(0.35, 0.58, "p_value=%s" % (asy_idx_p_value), style='normal', bbox={'facecolor':'lightgrey'})
asy_idx_fisher_box = plt.figtext(0.35, 0.55, "fisher=%s" % (asy_idx_fisher), style='normal', bbox={'facecolor':'lightgrey'})


"ALL REACHINGS AVG SMO IDX"
allSmoIdxAvgPlot = fig5.add_subplot(313)

allSmoIdxAvgPlot.set_ylim([0,15.0])
allSmoIdxAvgPlot.set_xlim([0,(maxSeed *3) +1])
allSmoIdxAvgPlot.set_yticks(np.arange(0, 15.0, 5))
allSmoIdxAvgPlot.set_xticks(np.arange(1, (maxSeed *3) +1, 1))

allSmoIdxAvgPlot.errorbar(np.linspace(1,maxSeed,maxSeed), meanNormalAllReachingsSmoIdx , yerr = stdNormalAllReachingsSmoIdx, marker= '^' , mec ='black', mfc='white' , ecolor ='black', fmt = '', linestyle='none')
meanAllSeedsNormalAllSmoIdx = patches.Rectangle((1,0), maxSeed -1,np.mean(meanNormalAllReachingsSmoIdx),linewidth=1,edgecolor='black',facecolor='lightgrey')
allSmoIdxAvgPlot.add_patch(meanAllSeedsNormalAllSmoIdx)

target_normal_smo_idx_plot = plt.plot([1.0, maxSeed] , [target_normal_smo_idx,target_normal_smo_idx], color = 'red', linestyle=':')

allSmoIdxAvgPlot.errorbar(np.linspace((maxSeed +1),maxSeed*2,maxSeed), meanAtaxiaAllReachingsSmoIdx,  yerr = stdAtaxiaAllReachingsSmoIdx, marker= '^', mec ='black', mfc='black', ecolor ='black', fmt = '', linestyle='none')
meanAllSeedsAtaxiaAllSmoIdx = patches.Rectangle((maxSeed +1,0), maxSeed -1, np.mean(meanAtaxiaAllReachingsSmoIdx),linewidth=1,edgecolor='black',facecolor='lightgrey')
allSmoIdxAvgPlot.add_patch(meanAllSeedsAtaxiaAllSmoIdx)

target_ataxia_asy_idx_plot = plt.plot([maxSeed +1, maxSeed *2] , [target_ataxia_smo_idx,target_ataxia_smo_idx], color = 'red', linestyle=':')

allSmoIdxAvgPlot.errorbar(np.linspace((maxSeed*2 +1),maxSeed*3,maxSeed), meantDCSAllReachingsSmoIdx,  yerr = stdtDCSAllReachingsSmoIdx, marker= '^', mec ='red', mfc='black', ecolor ='black', fmt = '', linestyle='none')
meanAllSeedstDCSAllSmoIdx = patches.Rectangle((maxSeed*2 +1,0), maxSeed -1, np.mean(meantDCSAllReachingsSmoIdx),linewidth=1,edgecolor='black',facecolor='lightgrey')
allSmoIdxAvgPlot.add_patch(meanAllSeedstDCSAllSmoIdx)

smo_idx_p_value = round(stats.f_oneway(statNormalAllReachingsSmoIdx.ravel(),statAtaxiaAllReachingsSmoIdx.ravel())[1],5)
smo_idx_fisher = round(stats.f_oneway(statNormalAllReachingsSmoIdx.ravel(),statAtaxiaAllReachingsSmoIdx.ravel())[0],5)

smo_idx_p_value_box = plt.figtext(0.35, 0.31, "p_value=%s" % (smo_idx_p_value), style='normal', bbox={'facecolor':'lightgrey'})
smo_idx_fisher_box = plt.figtext(0.35, 0.28, "fisher=%s" % (smo_idx_fisher), style='normal', bbox={'facecolor':'lightgrey'})



