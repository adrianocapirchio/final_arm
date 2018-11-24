# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 16:00:31 2018

@author: Alex
"""

import os
import numpy as np
import copy as copy
import matplotlib.pyplot as plt

from games import armReaching6targets
from arm import Arm
from basal_ganglia import actorCritic
from cerebellum import Cerebellum 


import utilities as utils


shoulder_range = np.deg2rad(np.array([- 60.0, 150.0]))
elbow_range    = np.deg2rad(np.array([  0.0, 180.0]))

xVisionRange = np.array([-0.4, 0.2]) 
yVisionRange = np.array([ 0.0, 0.6])  
netGoalVision = np.zeros(2)
netAgentVision = np.zeros(2)
 

proprioception_input = True

vision_input = True
goal_vision_input = True
ef_vision_input = False

multinet = True
noise = True



cerebellum = False
sub_cicle = 10

ataxia = False
tDCS = False



max_seed = 10
training_epoch  = 2500
max_epoch = training_epoch + 60


max_step = 250


#ataxia_mag = 15

LMA = 1000

epsi = 0.3


Kp1 = 8.0
Kd1 = 1.0

Kp2 = 8.0
Kd2 = 1.0

goal_range = 0.03

start_plotting = training_epoch + 60
start_ataxia = max_epoch - 40
start_tDCS = max_epoch - 20 
avg_stats = 20

muscolar_noise = 0.008

load_weights = False
save_data = True

if __name__ == "__main__":
    
    
    
    
    
    
    for seed in xrange(0,max_seed):
        
        "set simulation seed"
        np.random.seed(seed)
        
        "recover ataxia and tDCS to next seed"
        ataxia = False
        tDCS == False
        
        ataxia_mag = np.random.normal(7.0,1.0,1)
        tDSC_mag = copy.deepcopy(ataxia_mag) * 10
        
        
        
        "init objects"
        game = armReaching6targets(max_step,max_epoch)
        arm = Arm(shoulder_range, elbow_range)
        bg = actorCritic(proprioception_input, vision_input, goal_vision_input, ef_vision_input, multinet, game.goal_list, LMA, max_step, game.training_trial)
        if cerebellum== True:
            cb = Cerebellum(multinet, bg.curr_state, game.training_trial)
         #   cb.init(multinet, bg.curr_state)
            
        if load_weights == True:
            
            training_epoch = 0
            max_epoch = training_epoch + 60
            start_ataxia = max_epoch - 40
            start_tDCS = max_epoch - 20 
            
            
            tDSC_mag = copy.deepcopy(ataxia_mag) * 10
            
            noise = False
            save_data = False
            start_plotting = 0
            epoch = copy.deepcopy(training_epoch)
            print epoch
            
            mydir = os.getcwd
            os.chdir("C:\Users/Alex/Desktop/final_arm/data/")
            
            if multinet == True:
                os.chdir(os.curdir + "/multinet/")
            else:
                os.chdir(os.curdir + "/uninet/")
                
            if cerebellum == True:
                os.chdir(os.curdir + "/cerebellum/actETA=%s_critETA=%s_cbETA=%s_proprioUnits=%s_LMA=%s/" % (bg.ACT_ETA,bg.CRIT_ETA,cb.ETA,bg.proprio_input_units, LMA))
                ataxia_mag = np.load("ataxiaMag_seed=%s.npy" % (seed))
                
                if multinet == False:
                    cb.w =     np.load("cerebellumWeights_seed=%s.npy" % (seed))
                    bg.w_act = np.load("actorWeights_seed=%s.npy" % (seed))
                    bg.w_crit = np.load("criticWeights_seed=%s.npy" % (seed))
                else:
                    cb.w_multi = np.load("cerebellumWeights_seed=%s.npy" % (seed))
                    bg.multi_w_act = np.load("actorWeights_seed=%s.npy" % (seed))
                    bg.multi_w_crit = np.load("criticWeights_seed=%s.npy" % (seed))
                    
            else:
                os.chdir(os.curdir + "/onlyGanglia/actETA=%s_critETA=%s_proprioUnits=%s_LMA=%s/" % (bg.ACT_ETA,bg.CRIT_ETA,bg.proprio_input_units,LMA))
    
                if multinet == False:
                    bg.w_act = np.load("actorWeights_seed=%s.npy" % (seed))
                    bg.w_crit = np.load("criticWeights_seed=%s.npy" % (seed))
                else:
                    bg.multi_w_act = np.load("actorWeights_seed=%s.npy" % (seed))
                    bg.multi_w_crit = np.load("criticWeights_seed=%s.npy" % (seed))
            
            
            
            
            
        

    
        "training stats"
        epoch_training_accuracy = np.zeros(game.training_trial)
        epoch_training_time= np.ones(game.training_trial) * max_step
        avg_epoch_training_accuracy  = np.zeros(1)
        avg_epoch_training_time= np.zeros(1)
        avg5_epoch_training_accuracy = np.zeros(avg_stats)
        avg5_epoch_training_time = np.ones(avg_stats) * max_step
        final_training_accuracy = np.zeros(max_epoch/avg_stats)
        final_training_time = np.ones(max_epoch/avg_stats) * max_step
    
        "test stats"                         
        epoch_test_accuracy = np.zeros(game.test_trial)
        epoch_test_time= np.ones(game.test_trial) * max_step
        avg_epoch_test_accuracy  = np.zeros(1)
        avg_epoch_test_time= np.zeros(1)
        avg5_epoch_test_accuracy = np.zeros(avg_stats)
        avg5_epoch_test_time = np.ones(avg_stats) * max_step
        final_test_accuracy = np.zeros(max_epoch/avg_stats)
        final_test_time = np.ones(max_epoch/avg_stats) * max_step
        
        
        
        for epoch in xrange(max_epoch):
            
            
            "start ataxia and tDCS"
            if cerebellum == True:
                if epoch > start_ataxia:
                    ataxia = True           
                if epoch > start_tDCS:
                    tDCS == True
            
            "reset epoch values"
            bg.epoch_reset(max_step)
            game.epoch_reset()
            arm.epochReset(shoulder_range,elbow_range)
            game.curr_position = np.array([arm.xEndEf,arm.yEndEf])             
            if cerebellum == True:
                cb.epoch_reset()
            
            
            "init netword output array"
            netOut = np.ones(2) * 0.5
            prv_netOut = netOut.copy()
            
            desiredAngles = np.ones(2) * 0.5
            desiredAngles[0] = utils.changeRange(desiredAngles[0], 0.,1.,shoulder_range[0],shoulder_range[1])
            desiredAngles[1] = utils.changeRange(desiredAngles[1], 0.,1., elbow_range[0],elbow_range[1])
            
            xDesAng = arm.L1*np.cos(desiredAngles[0]) + arm.L2*np.cos(desiredAngles[0]+desiredAngles[1])
            yDesAng = arm.L1*np.sin(desiredAngles[0]) + arm.L2*np.sin(desiredAngles[0]+desiredAngles[1])
            
            gangliaDesAng = np.ones(2) * 0.5
            gangliaDesAng[0] = utils.changeRange(bg.curr_act_I[0], 0.,1.,shoulder_range[0],shoulder_range[1])
            gangliaDesAng[1] = utils.changeRange(bg.curr_act_I[1], 0.,1.,elbow_range[0],elbow_range[1])
            
            xGangliaDes = arm.L1*np.cos(gangliaDesAng[0]) + arm.L2*np.cos(gangliaDesAng[0]+gangliaDesAng[1])
            yGangliaDes = arm.L1*np.sin(gangliaDesAng[0]) + arm.L2*np.sin(gangliaDesAng[0]+gangliaDesAng[1])
            
            if cerebellum == True:
                cerebDesAng = np.ones(2) * 0.5
                cerebDesAng[0] = utils.changeRange(cb.curr_I[0], 0.,1.,shoulder_range[0],shoulder_range[1])
                cerebDesAng[1] = utils.changeRange(cb.curr_I[1], 0.,1.,elbow_range[0],elbow_range[1])
                
                xCerebDes = arm.L1*np.cos(cerebDesAng[0]) + arm.L2*np.cos(cerebDesAng[0]+cerebDesAng[1])
                yCerebDes = arm.L1*np.sin(cerebDesAng[0]) + arm.L2*np.sin(cerebDesAng[0]+cerebDesAng[1])
            
            
            
            "init first plot"
            if epoch == start_plotting:
                                
                fig1   = plt.figure("Workspace", figsize=(9,8))
                
                text1 = plt.figtext(.02, .72, "epoch = %s" % (0), style='italic', bbox={'facecolor':'yellow'})
                text2 = plt.figtext(.02, .62, "trial = %s" % (0), style='italic', bbox={'facecolor':'lightblue'})
                text3 = plt.figtext(.02, .52, "step = %s" % (0), style='italic', bbox={'facecolor':'lightcoral'})
          #      text4 = plt.figtext(.02, .42, "reward = %s" % (0), style='italic', bbox={'facecolor':'lightgreen'})
                
                ax1 = fig1.add_subplot(111, aspect='equal')
                ax1.set_xlim([-1.0,1.0])
                ax1.set_ylim([-1.0,1.0])
                
                circle1 = plt.Circle((game.goal_list[0]), goal_range, color = 'yellow') 
                edgecircle1 = plt.Circle((game.goal_list[0]), goal_range, color = 'black', fill = False) 
                ax1.add_artist(circle1)
                ax1.add_artist(edgecircle1)
                
                circle2 = plt.Circle((game.goal_list[1]), goal_range, color = 'yellow') 
                edgecircle2 = plt.Circle((game.goal_list[1]), goal_range, color = 'black', fill = False) 
                ax1.add_artist(circle2)
                ax1.add_artist(edgecircle2)
                
                circle3 = plt.Circle((game.goal_list[2]), goal_range, color = 'yellow') 
                edgecircle3 = plt.Circle((game.goal_list[2]), goal_range, color = 'black', fill = False) 
                ax1.add_artist(circle3)
                ax1.add_artist(edgecircle3)
                
                circle4 = plt.Circle((game.goal_list[3]), goal_range, color = 'yellow') 
                edgecircle4 = plt.Circle((game.goal_list[3]), goal_range, color = 'black', fill = False) 
                ax1.add_artist(circle4)
                ax1.add_artist(edgecircle4)
                
    
                
                rewardCircle = plt.Circle((game.goal_list[0]), goal_range, color = 'red')  
                ax1.add_artist(rewardCircle)
                
                
                line1, = ax1.plot([0, arm.xElbow, arm.xEndEf], [0, arm.yElbow, arm.yEndEf], 'k-', color = 'pink', linewidth = 10)
                point, = ax1.plot([arm.xEndEf], [arm.yEndEf], 'o', color = 'black' , markersize= 20)
            
                desEnd, = ax1.plot([xDesAng], [yDesAng], 'o', color = 'green' , markersize= 10)
                gangliaOut, = ax1.plot([xGangliaDes], [yGangliaDes], 'o', color = 'blue' , markersize= 10)
            
                if cerebellum == True:
                    cerebOut, = ax1.plot([xCerebDes], [yCerebDes], 'o', color = 'orange' , markersize= 10)
            
            
            "update plotting in epoch"
            if epoch >= start_plotting:
                text1.set_text("epoch = %s" % (epoch))
                point.set_data([arm.xEndEf], [arm.yEndEf]) 
                line1.set_data([0, arm.xElbow, arm.xEndEf], [0, arm.yElbow, arm.yEndEf])
                plt.pause(0.1)
            
            
            
            "set training and test max trial"
            if epoch < training_epoch:
                max_trial = copy.deepcopy(game.training_trial)
            else:
                max_trial = copy.deepcopy(game.test_trial)
            
       #     bg.training_performance[epoch%LMA,:] *= 0. 
     #       print bg.training_performance[epoch%LMA,:] 
            
            
            for trial in xrange(max_trial):
                
                
                if epoch < training_epoch:
                    if trial % 2 == 0:
                        bg.prv_rew = 0
                    if trial % 2 == 1:
                        bg.prv_prv_rew = 0
                
                "reset trial values"
                bg.trial_reset(max_step)                
                if cerebellum == True:
                    cb.trial_reset()              
                if epoch< training_epoch:
                    game.trial_reset()
                
        

                game.set_test_goal(trial)  
                game.goalIndex1(trial)
                
                    
                
            #    "set goal position for training or test phase"
            #    if epoch < training_epoch:
            #        game.set_training_goals(trial)
            #        game.training_index(trial)
            #    else:
            #        game.set_test_goal(trial)  
                
                
                
                
                
                
                "update plotting during trial"
                if epoch >= start_plotting:                  
                    text2.set_text("trial = %s" % (trial))
                    rewardCircle.remove()
                    rewardCircle = plt.Circle((game.goal_position), goal_range, color = 'red') 
                    ax1.add_artist(rewardCircle)
                    plt.pause(0.1)
  
                
                
                "load current w in multinet"
                if multinet == True:           
                    bg.w_crit *= 0
                    bg.w_act*= 0
                    if cerebellum == True:
                        cb.w *= 0                      
               #     for i in xrange(len(game.goal_list)):                        
                #        if (game.goal_position == game.goal_list[i]).all():
                    bg.w_crit = bg.multi_w_crit[:,trial].copy()
                    bg.w_act = bg.multi_w_act[:,:,trial].copy()
                    if cerebellum == True:
                        cb.w = cb.w_multi[:,:,trial].copy()
                     #       break
                
                
                
                "compute T on the performance basis"
                
               # else:
            #    if trial == 1 or trial == 3 or trial == 5:
                if epoch < training_epoch:
            #        if epoch < LMA:    
            #            T = 1.0 
             #       else:
                    T = (1.0 - (np.sum(bg.training_performance[:, trial]) / LMA))
              #      else:
               #         temp = 1. *((bg.training_performance[:, 0] + bg.training_performance[:, 2] + bg.training_performance[:, 4] + bg.training_performance[:, 6]) > 0)
                #        T = (1.0 - (np.sum(temp) / LMA))
                
                "acquire goal vision state"
                if goal_vision_input == True:
                    netGoalVision[0] = utils.changeRange(game.goal_position[0], xVisionRange[0], xVisionRange[1], 0., 1.)
                    netGoalVision[1] = utils.changeRange(game.goal_position[1], yVisionRange[0], yVisionRange[1], 0., 1.)
                    bg.acquire_goal_state(netGoalVision)

                
                
                
                for step in xrange(max_step):
                    
                        
                    
                    "Initialize end-effector in a random position"
                    if step == 0:
                        
                        if epoch < training_epoch:
                            if trial == 0:
                                continue
                            elif trial % 2 == 1: #or trial == 3 or trial == 5:
                                if bg.prv_rew == 0:
                                    break
                            elif trial % 2== 0:# or trial == 4 or trial == 6:
                                if bg.prv_prv_rew == 0:
                                    break
                        
                     #   print "step", step, "trial" ,trial, "epoch", epoch
                        
                   #     if epoch < training_epoch:
                   #         randPos = np.random.uniform(0.0 , 1.0 ,2)                 
                   #         arm.setEffPosition(randPos,shoulder_range, elbow_range)
                   #         game.curr_position = np.array([arm.xEndEf,arm.yEndEf])
                            
                        "acquire proprioception input"    
                        if proprioception_input == True:
                            bg.acquire_proprio_state(np.array([utils.changeRange(arm.theta1, shoulder_range[0], shoulder_range[1], 0., 1.), utils.changeRange(arm.theta2, elbow_range[0], elbow_range[1], 0., 1.)]))

                                
                        "compute all inputs state"
                        if proprioception_input == True:
                            if goal_vision_input == True:
                                bg.curr_state = np.hstack([bg.proprio_state, bg.goal_vision_state])  
                            else:
                                bg.curr_state = bg.proprio_state.copy()                                
                        else:
                            if goal_vision_input == True:
                                bg.curr_state = bg.goal_vision_state.copy()
                    
                    
             
                        

                   
                    
                    
                    
                    "save last step dynamics"
                    game.prv_position = game.curr_position.copy() #save position
                    game.prv_velocity = game.curr_velocity.copy() #save velocity
                    game.prv_acceleration = game.curr_acceleration.copy() #save acceleration
                    
                    "save last step neural network values"
             #       if step > 0:
                        
                    bg.prv_state = bg.curr_state.copy() # save state
                    
                    bg.prv_crit_U = bg.curr_crit_U.copy() # save critic potential 
                    
                    bg.prv_act_U = bg.curr_act_U.copy() # save actor potential
                    bg.prv_act_I = bg.curr_act_I.copy() # save actor frequency
                    
                    bg.prv_noise = bg.curr_noise.copy() # save noise
                    prv_netOut = netOut.copy()
                                                     
                    if cerebellum == True:
                        cb.prv_U = cb.curr_U.copy()
                        if ataxia == False:
                            cb.prv_I = cb.curr_I.copy()  # save perceptron frequency
                        else:
                            cb.prv_I = cb.damage_I.copy()
                        

                    
                    "compute torques and arm movement"
                    Torque = arm.PD_controller(np.array([desiredAngles[0],desiredAngles[1]]), Kp1 , Kp2, Kd1, Kd2) # compute torques
                    arm.SolveDirectDynamics(Torque[0], Torque[1]) # move the arm
                    game.curr_position = np.array([arm.xEndEf,arm.yEndEf]) #save end-effector position 
                    
                    
                    
                    "compute dynamics"                     
                    if step > 0:
                        game.curr_velocity = (utils.distance(game.curr_position,game.prv_position)) / arm.dt #compute end-effector current velocity
                        game.curr_acceleration = (game.curr_velocity - game.prv_velocity) / arm.dt # compute  end-effector current acceleration
                        game.curr_jerk= (game.curr_acceleration - game.prv_acceleration) / arm.dt #compute  end-effector current jerk
                    
                    
                    "compute network desired angles in the task space"
                    gangliaDesAng[0] = utils.changeRange(bg.curr_act_I[0], 0.,1.,shoulder_range[0],shoulder_range[1])
                    gangliaDesAng[1] = utils.changeRange(bg.curr_act_I[1], 0.,1., elbow_range[0],elbow_range[1])
                    
                    if cerebellum == True:
                        if ataxia== False:
                            cerebDesAng[0] = utils.changeRange(cb.curr_I[0], 0.,1.,shoulder_range[0],shoulder_range[1])
                            cerebDesAng[1] = utils.changeRange(cb.curr_I[1], 0.,1., elbow_range[0],elbow_range[1])               
                        else:
                            cerebDesAng[0] = utils.changeRange(cb.damage_I[0], 0.,1.,shoulder_range[0],shoulder_range[1])
                            cerebDesAng[1] = utils.changeRange(cb.damage_I[1], 0.,1., elbow_range[0],elbow_range[1])
                    
                    
                    
                    
                    "save step data"
                    if save_data == True:
                        
                        if epoch < training_epoch:
                        
                            game.goal_position_training_data[:,trial,epoch] = game.goal_position.copy() 
                            
                            if trial == 0 or trial == 2 or trial == 4 or trial == 6:
                                game.goal_angles_training_data[:,step,trial,epoch] = np.array([ 0.64904199,  2.38552347])
                            elif trial == 1:
                                game.goal_angles_training_data[:,step,trial,epoch] = np.array([ 1.45202547,  1.18722554])
                            elif trial == 3:
                                game.goal_angles_training_data[:,step,trial,epoch] = np.array([0.62191676,  1.52726939])
                            elif trial == 5:
                                game.goal_angles_training_data[:,step,trial,epoch] = np.array([ 0.96953405,  1.50740439])
                                
                                
                            game.trajectories_training_data[:,step,trial,epoch] = game.curr_position.copy()
                            game.arm_angles_training_data[:,step,trial,epoch] = np.array([arm.theta1,arm.theta2]).copy()
                            game.ganglia_angles_training_data[:,step,trial,epoch]  = gangliaDesAng.copy()
                            game.jerk_training_data[step,trial,epoch] = game.curr_jerk.copy() 
                    
                            if cerebellum == True:
                                game.cereb_angles_training_data[:,step,trial,epoch] = cerebDesAng.copy() 
                    
                        else:
                        
                            game.goal_position_test_data[:,trial,epoch - training_epoch] = game.goal_position.copy() 
                            
                            if trial == 0 or trial == 2 or trial == 4 or trial == 6:
                                game.goal_angles_test_data[:,step,trial,epoch - training_epoch] = np.array([ 0.64904199,  2.38552347])
                            elif trial == 1:
                                game.goal_angles_test_data[:,step,trial,epoch - training_epoch] = np.array([ 1.45202547,  1.18722554])
                            elif trial == 3:
                                game.goal_angles_test_data[:,step,trial,epoch - training_epoch] = np.array([0.62191676 , 1.52726939])
                            elif trial == 5:
                                game.goal_angles_test_data[:,step,trial,epoch - training_epoch] = np.array([0.96953405,  1.50740439])
                                
                                
                            game.trajectories_test_data[:,step,trial,epoch - training_epoch] = game.curr_position.copy()
                            game.arm_angles_test_data[:,step,trial,epoch - training_epoch] = np.array([arm.theta1,arm.theta2]).copy()
                            game.ganglia_angles_test_data[:,step,trial,epoch - training_epoch]  = gangliaDesAng.copy()
                            game.jerk_test_data[step,trial,epoch - training_epoch] = game.curr_jerk.copy()
                            
                            if cerebellum == True:
                                game.cereb_angles_test_data[:,step,trial,epoch - training_epoch] = cerebDesAng.copy() 
                    
                    
                    
                    
                    "plot changes step by step"
                    if epoch >= start_plotting:                          
                        text3.set_text("step = %s" % (step))
                     #  text4.set_text("reward =%s" % (bg.rewardCounter[game.goalIdx]))
                        line1.set_data([0, arm.xElbow, arm.xEndEf], [0, arm.yElbow, arm.yEndEf])
                        point.set_data([arm.xEndEf], [arm.yEndEf]) 
                        
                        xDesAng = arm.L1*np.cos(desiredAngles[0]) + arm.L2*np.cos(desiredAngles[0]+desiredAngles[1])
                        yDesAng = arm.L1*np.sin(desiredAngles[0]) + arm.L2*np.sin(desiredAngles[0]+desiredAngles[1])
                        desEnd.set_data([xDesAng],[yDesAng])
                                                                     
                        xGangliaDes = arm.L1*np.cos(gangliaDesAng[0]) + arm.L2*np.cos(gangliaDesAng[0]+gangliaDesAng[1])
                        yGangliaDes = arm.L1*np.sin(gangliaDesAng[0]) + arm.L2*np.sin(gangliaDesAng[0]+gangliaDesAng[1])
                        gangliaOut.set_data( [xGangliaDes] , [yGangliaDes] )
                        
                        if cerebellum == True:
                            xCerebDes = arm.L1*np.cos(cerebDesAng[0]) + arm.L2*np.cos(cerebDesAng[0]+cerebDesAng[1])
                            yCerebDes = arm.L1*np.sin(cerebDesAng[0]) + arm.L2*np.sin(cerebDesAng[0]+cerebDesAng[1])
                            cerebOut.set_data([xCerebDes], [yCerebDes])
                                                
                        plt.pause(0.01)
                    
                    
                    "acquire proprioception input"    
                    if proprioception_input == True:
                        bg.acquire_proprio_state(np.array([utils.changeRange(arm.theta1, shoulder_range[0], shoulder_range[1], 0., 1.), utils.changeRange(arm.theta2, elbow_range[0], elbow_range[1], 0., 1.)]))
                                    
                    "compute all inputs state"
                    if proprioception_input== True:
                        if goal_vision_input == True:
                            bg.curr_state = np.hstack([bg.proprio_state, bg.goal_vision_state])  
                        else:
                            bg.curr_state = bg.proprio_state.copy()
                    else:
                        if goal_vision_input == True:
                            bg.curr_state = bg.goal_vision_state.copy()
                            
                    
                    
                    
                    
                    
                    
                    "compute goal distance and reward"        
                    game.compute_goal_distance()
                    
                    
                    if epoch < training_epoch:
                        if game.distance < goal_range:   
                            if trial == 1 or trial == 3 or trial == 5:
                                if step > 0:
                                    if bg.prv_rew == 1:
                                        bg.prv_prv_rew = 1
                                        bg.curr_rew = np.e**(-epsi * game.curr_velocity)                          
                                        bg.curr_crit_U = np.zeros(1)
                            elif trial == 2 or trial == 4 or trial == 6:
                                if step > 0:
                                    if bg.prv_prv_rew == 1:
                                        bg.prv_rew = 1
                                        bg.curr_rew = np.e**(-epsi * game.curr_velocity)                          
                                        bg.curr_crit_U = np.zeros(1)
                                        
                                        
                            else:
                                if step > 0:
                                    bg.prv_rew = 1
                                    bg.curr_rew = np.e**(-epsi * game.curr_velocity)                          
                                    bg.curr_crit_U = np.zeros(1)
                        
                        elif step == max_step -1:                        
                            bg.curr_crit_U = np.zeros(1)
                        
                        else: 
                            bg.compute_crit_U() 
                    
                    
                    else:
                    
                        if game.distance < goal_range:                        
                            if step > 0:
                                bg.curr_rew = np.e**(-epsi * game.curr_velocity)                           
                                bg.curr_crit_U = np.zeros(1)                   
                        
                        elif step == max_step -1:                        
                            bg.curr_crit_U = np.zeros(1)                       
                        
                        else: 
                            bg.compute_crit_U() 

                    
                    
                    "compute TD error"
                    if step > 0:
                        bg.compute_surprise()
                    
                    "stop arm if reached goal"
                    if bg.curr_rew > 0:
                        game.stop_move()
                        arm.stopMove() 
                        
                        
                        
                    "compute actor outputs"
                    bg.compute_actor_U()                            
                    bg.compute_actor_I() 
                    
                    if noise== True:
                        if epoch < training_epoch:
                            bg.compute_noise(T)
                        else:
                            bg.curr_noise *= 0
                            
                    "compute perceptron out"
                    if cerebellum == True:                        
                        cb.comp_U(bg.curr_state) #potential
                        
                        if ataxia == False:                        
                            cb.comp_I() # healthyfrequency          
                        else:                           
                            if tDCS == False:                   
                                cb.comp_I() # Healthy frequency
                            else:    
                                
                                cb.comp_I_tDCS(tDSC_mag) 
                    
                            cb.tau1 = copy.deepcopy(ataxia_mag)# np.random.normal(ataxia_mag, ataxia_mag /4.0 )# 1. + np.abs(np.random.normal(0.0, ataxia_mag)) ##                           
                            if cb.tau1 < 1.0:
                                cb.tau1 = 1.0
                            
                            cb.damage_I = ((1.0 - cb.dT / cb.tau1) * cb.damage_I + (cb.dT / cb.tau1) * (cb.curr_I )) + np.random.normal(0.0, ataxia_mag * 0.005, 2 )  #damaged frequency
                    
                    
                    "weightned sum of signals"
                    if cerebellum== True:
                        K = 1./2
                        if ataxia== False:    
                            netOut = K * bg.curr_act_I + K* cb.curr_I #+ np.random.normal(0.0, muscolar_noise, 2)
                        else:
                            netOut = K * bg.curr_act_I + K* cb.damage_I #+ np.random.normal(0.0, muscolar_noise, 2)
                    else:
                        K = 1.
                        netOut = K * bg.curr_act_I# + np.random.normal(0.0, muscolar_noise, 2)
                    
                    
                    "compute noised signals and convert to arm desired angles"
                    desiredAngles[0] = utils.changeRange(utils.limitRange(netOut[0] + bg.curr_noise[0] + np.random.normal(0.0, muscolar_noise, 1) , 0., 1.), 0., 1., shoulder_range[0] , shoulder_range[1])
                    desiredAngles[1] = utils.changeRange(utils.limitRange(netOut[1] + bg.curr_noise[1] + np.random.normal(0.0, muscolar_noise, 1),  0., 1.), 0., 1.,  elbow_range[0] , elbow_range[1])
                    
                    "load perceptron teaching input"
                    if cerebellum == True:                            
                        bg.act_I_buffer[:,step] = netOut.copy()
                        bg.state_buffer[:,step] = bg.curr_state.copy()
                    
                    
                    "training"
                    if epoch < training_epoch:
                        
                        "actor-critic training"
                        if step > 0:        
                            bg.train_crit()
                            bg.train_act(prv_netOut)
                        
                        "perceptron training"    
                        if cerebellum == True:
                            for i in xrange(sub_cicle):
                                if (bg.curr_rew > 0):  
                                    for i in xrange(step + 1): 
                                        cb.train(bg.state_buffer[:,i], bg.act_I_buffer[:,i], bg.curr_rew, T)
                            
                     
                                               
                    "unload current weights"
                    if multinet == True:               
                  #      for i in xrange(len(game.goal_list)):
                   #         if (game.goal_position == game.goal_list[i]).all():
                        bg.multi_w_crit[:,trial] = bg.w_crit.copy()
                        bg.multi_w_act[:,:,trial] = bg.w_act.copy()
                        if cerebellum == True:
                            cb.w_multi[:,:,trial] = cb.w.copy()
                     #           break
                    
                    
                    
                    "save epoch time and accuracy if success in training and test"
                    if epoch < training_epoch:
                        if bg.curr_rew > 0:
                            epoch_training_accuracy[trial%game.training_trial] = 1.
                            epoch_training_time[trial%game.training_trial] = step 
                            
                        #    if epoch > LMA:                   
                            bg.training_performance[epoch%LMA,trial] = 1.
                            print trial, game.goal_position, step, bg.curr_rew, np.array([arm.theta1,arm.theta2]), round(T,2)#round(1.0 - (np.sum(bg.training_performance[:, trial]) / LMA),2)
                        
                            break 
                    else:
                        
                        if bg.curr_rew > 0:
                            epoch_test_accuracy[trial%game.test_trial] = 1.
                            epoch_test_time[trial%game.test_trial] = step 
                            print trial, game.goal_position, step, bg.curr_rew#, np.array([arm.theta1,arm.theta2]), (1.0 - (np.sum(bg.training_performance[:, game.goalIdx]) / LMA))
                
                            break
                
                "save epoch time and accuracy if fail in training and test"
                if epoch < training_epoch:        
                    if bg.curr_rew  == 0:    
                        epoch_training_accuracy[trial%game.training_trial] = 0.          
                        epoch_training_time[trial%game.training_trial] = max_step                
                        
             #          if epoch > LMA:
                        bg.training_performance[epoch%LMA,trial] = 0. 
                else:
                    if bg.curr_rew == 0:    
                        epoch_test_accuracy[trial%game.test_trial] = 0.          
                        epoch_test_time[trial%game.test_trial] = max_step

            
            
         #   print bg.training_performance[epoch%LMA,:]  
            
            
            "compute average training and test accuracy and time in 10 epoch"
            if epoch < training_epoch:
                
                avg_epoch_training_accuracy  = (float(np.sum(epoch_training_accuracy)) / game.training_trial) * 100
                avg_epoch_training_time= (float(np.sum(epoch_training_time)) / game.training_trial)
        
                print "epoch" , epoch, "avarage steps", round(avg_epoch_training_time,2)  , "accurancy" , round(avg_epoch_training_accuracy,2), "%", "seed", seed
        
                avg5_epoch_training_accuracy[epoch%avg_stats] = avg_epoch_training_accuracy
                avg5_epoch_training_time[epoch%avg_stats] = avg_epoch_training_time
                         
                if epoch % avg_stats == (avg_stats -1):
                    final_training_accuracy[(epoch/avg_stats)%(max_epoch/avg_stats)] = np.sum(avg5_epoch_training_accuracy) / avg_stats
                    final_training_time[(epoch/avg_stats)%(max_epoch/avg_stats)] = np.sum(avg5_epoch_training_time) / avg_stats
                    print "******avg 5 epoch" , epoch, "avarage steps", round(np.sum(avg5_epoch_training_time) / avg_stats,2), "accurancy" , round(np.sum(avg5_epoch_training_accuracy) / avg_stats,2), "%"
            else:
                
                avg_epoch_test_accuracy  = (float(np.sum(epoch_test_accuracy)) / game.test_trial) * 100
                avg_epoch_test_time= (float(np.sum(epoch_test_time)) / game.test_trial)
        
                print "epoch" , epoch, "avarage steps", round(avg_epoch_test_time,2)  , "accurancy" , round(avg_epoch_test_accuracy,2), "%", "seed", seed
        
                avg5_epoch_test_accuracy[epoch%avg_stats] = avg_epoch_test_accuracy
                avg5_epoch_test_time[epoch%avg_stats] = avg_epoch_test_time
                         
                if epoch % avg_stats == (avg_stats -1):
                    final_test_accuracy[(epoch/avg_stats)%(max_epoch/avg_stats)] = np.sum(avg5_epoch_test_accuracy) / avg_stats
                    final_test_time[(epoch/avg_stats)%(max_epoch/avg_stats)] = np.sum(avg5_epoch_test_time) / avg_stats
                    print "******avg 5 epoch" , epoch, "avarage steps", round(np.sum(avg5_epoch_test_time) / avg_stats,2), "accurancy" , round(np.sum(avg5_epoch_test_accuracy) / avg_stats,2), "%"
        
        
        "plot training time"
        plt.figure(figsize=(120, 4), dpi=160)
        plt.title('average time in %s epoch' % (avg_stats))
        plt.xlim([0, max_epoch/avg_stats])
        plt.ylim([0, max_step])
        plt.xlabel("epochs")
        plt.ylabel("s")
        plt.xticks(np.arange(0,max_epoch/avg_stats, 10))
        if cerebellum == True:
            plt.plot(final_training_time, label ="cerebellum_ETA=%s_seed=%s_train" % (cb.ETA, seed))
            plt.plot(final_test_time, label ="cerebellum_ETA=%s_seed=%s_test" % (cb.ETA, seed))
        else:
            plt.plot(final_training_time, label ="Only_Ganglia_" + "actETA=_train" + str(bg.ACT_ETA) + "critETA=" + str(bg.CRIT_ETA) + "seed=" + str(seed))
            plt.plot(final_test_time, label ="Only_Ganglia_" + "actETA=_test" + str(bg.ACT_ETA) + "critETA=" + str(bg.CRIT_ETA) + "seed=" + str(seed))
        plt.legend(loc='lower left')
        
        
        "plot training accuracy"
        plt.figure(figsize=(120, 4), dpi=160)
        plt.title('training accurancy in %s epoch' % (avg_stats))
        plt.xlim([0, max_epoch/avg_stats])
        plt.ylim([0,101])
        plt.xlabel("epochs")
        plt.ylabel("accurancy %")
        plt.xticks(np.arange(0,max_epoch/avg_stats, 10))
        if cerebellum == True:
            plt.plot(final_training_accuracy, label ="cerebellum_ETA=%s_seed=%s_train" % (cb.ETA, seed))
            plt.plot(final_test_accuracy, label ="cerebellum_ETA=%s_seed=%s_test" % (cb.ETA, seed))
        else:
            plt.plot(final_training_accuracy, label ="Only_Ganglia_" + "actETA=_train" + str(bg.ACT_ETA) + "critETA=" + str(bg.CRIT_ETA) + "seed=" + str(seed))
            plt.plot(final_test_accuracy, label ="Only_Ganglia_" + "actETA=_test" + str(bg.ACT_ETA) + "critETA=" + str(bg.CRIT_ETA) + "seed=" + str(seed))
        plt.legend(loc='upper left') 
        
        
        
     #   "plot test time"
     #   plt.figure(figsize=(120, 4), dpi=160)
     #   plt.title('test average time in %s epoch' % (avg_stats))
     #   plt.xlim([0, max_epoch/avg_stats])
     #   plt.ylim([0, max_step])
     #   plt.xlabel("epochs")
     #   plt.ylabel("s")
     #   plt.xticks(np.arange(0,max_epoch/avg_stats, 10))
     #   if cerebellum == True:
            
     #   else:
     #               
     #   plt.legend(loc='lower left')
        
        
  #      "plot test accuracy"
  #      plt.figure(figsize=(120, 4), dpi=160)
  #      plt.title('test accurancy in %s epoch' % (avg_stats))
  #      plt.xlim([0, max_epoch/avg_stats])
  #      plt.ylim([0,101])
  #      plt.xlabel("epochs")
  #      plt.ylabel("accurancy %")
  #      plt.xticks(np.arange(0,max_epoch/avg_stats, 10))
  #      if cerebellum == True:
            
  #      else:
                
  #      plt.legend(loc='lower left') 
        
        
        
        
        "save data"
        if save_data == True:
            mydir = os.getcwd
            os.chdir("C:\Users/Alex/Desktop/final_arm/data/")
            
            if multinet == True:
                if not os.path.exists(os.curdir + "/multinet/"):
                    os.makedirs(os.curdir + "/multinet/")
                os.chdir(os.curdir + "/multinet/")
            else:
                if not os.path.exists(os.curdir + "/uninet/"):
                    os.makedirs(os.curdir + "/uninet/")
                os.chdir(os.curdir + "/uninet/")
            
            if cerebellum == True:
    
                if not os.path.exists(os.curdir + "/cerebellum/actETA=%s_critETA=%s_cbETA=%s_proprioUnits=%s_LMA=%s/" % (bg.ACT_ETA,bg.CRIT_ETA,cb.ETA,bg.proprio_input_units,LMA)):
                    os.makedirs(os.curdir + "/cerebellum/actETA=%s_critETA=%s_cbETA=%s_proprioUnits=%s_LMA=%s/" % (bg.ACT_ETA,bg.CRIT_ETA,cb.ETA,bg.proprio_input_units,LMA))
                os.chdir(os.curdir + "/cerebellum/actETA=%s_critETA=%s_cbETA=%s_proprioUnits=%s_LMA=%s/" % (bg.ACT_ETA,bg.CRIT_ETA,cb.ETA,bg.proprio_input_units,LMA))
                
    
                
                if multinet == False:
                    np.save("cerebellumWeights_seed=%s" % (seed), (cb.w))
                else:
                    np.save("cerebellumWeights_seed=%s" % (seed), (cb.w_multi))
                
                np.save("cerebAnglesTrainingData_seed=%s" % (seed), game.cereb_angles_training_data)
                np.save("cerebAnglesTestData_seed=%s" % (seed), game.cereb_angles_test_data)

                
                
            else:
                if not os.path.exists(os.curdir + "/onlyGanglia/actETA=%s_critETA=%s_proprioUnits=%s_LMA=%s/" % (bg.ACT_ETA,bg.CRIT_ETA, bg.proprio_input_units,LMA)):
                    os.makedirs(os.curdir + "/onlyGanglia/actETA=%s_critETA=%s_proprioUnits=%s_LMA=%s/" % (bg.ACT_ETA,bg.CRIT_ETA, bg.proprio_input_units,LMA))
                os.chdir(os.curdir + "/onlyGanglia/actETA=%s_critETA=%s_proprioUnits=%s_LMA=%s/" % (bg.ACT_ETA,bg.CRIT_ETA, bg.proprio_input_units,LMA))
            
            
            if multinet == False:
                np.save("actorWeights_seed=%s" % (seed), (bg.w_act))
                np.save("criticWeights_seed=%s" % (seed), (bg.w_crit))
            else:
                np.save("actorWeights_seed=%s" % (seed), (bg.multi_w_act))
                np.save("criticWeights_seed=%s" % (seed), (bg.multi_w_crit))
        
        
            np.save("goalPositionTrainingData_seed=%s" % (seed), game.goal_position_training_data)
            np.save("goalAnglesTrainingData_seed=%s" % (seed), game.goal_angles_training_data)
            np.save("trajectoriesTrainingData_seed=%s" % (seed), game.trajectories_training_data)
            np.save("armAnglesTrainingData_seed=%s" % (seed), game.arm_angles_training_data)
            np.save("gangliaAnglesTrainingData_seed=%s" % (seed), game.ganglia_angles_training_data)
            np.save("jerkTrainingData_seed=%s" % (seed), game.jerk_training_data)
            
            
            np.save("goalPositionTestData_seed=%s" % (seed), game.goal_position_test_data)
            np.save("goalAnglesTestData_seed=%s" % (seed), game.goal_angles_test_data)
            np.save("trajectoriesTestData_seed=%s" % (seed), game.trajectories_test_data)
            np.save("armAnglesTestData_seed=%s" % (seed), game.arm_angles_test_data)
            np.save("gangliaAnglesTestData_seed=%s" % (seed), game.ganglia_angles_test_data)
            np.save("jerkTestData_seed=%s" % (seed), game.jerk_test_data)
            
            np.save("finalTrainingTime_seed=%s" %(seed), final_training_time)
            np.save("finalTestTime_seed=%s" %(seed), final_test_time)
            
            np.save("finalTrainingAccuracy_seed=%s" %(seed), final_training_accuracy)
            np.save("finalTestAccuracy_seed=%s" %(seed), final_test_accuracy)
            np.save("ataxiaMag_seed=%s"%(seed), ataxia_mag) 
            
            
            