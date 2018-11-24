# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 15:14:40 2018

@author: Alex
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import utilities as utils


from arm import Arm
from games import armReaching6targets




shoulder_range = np.deg2rad(np.array([-60.0, 150.0]))
elbow_range    = np.deg2rad(np.array([  0.0, 180.0])) 


multinet = True
cerebellum = True

ataxia = False





actETA =  6.0 * 10 ** ( -1)
critETA = 6.0 * 10 ** ( -4)
cbETA   = 6.0 * 10 ** ( -1)
proprio_units = 101**2
LMA = 1000

goal_range = 0.03

seed = 0
training_epoch = 2

max_epoch = training_epoch + 60
max_step = 250
starting_epoch = training_epoch
start_plotting= training_epoch

if training_epoch > 20:
    ataxia = True
    ataxia_mag = 15.0
    
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
    

gameGoalPos = np.zeros(2)


trajectories              = np.load("trajectoriesTestData_seed=%s.npy" % (seed))
armAngles                 = np.load("armAnglesTestData_seed=%s.npy" % (seed))
gangliaAngles             = np.load("gangliaAnglesTestData_seed=%s.npy" % (seed))

if cerebellum== True:
    cerebAngles           = np.load("cerebAnglesTestData_seed=%s.npy" % (seed)) 

goalPosition              = np.load("goalPositionTestData_seed=%s.npy" % (seed))
goalAngles                = np.load("goalAnglesTestData_seed=%s.npy" % (seed))
jerk                      = np.load("jerkTestData_seed=%s.npy" % (seed))

if __name__ == "__main__":
    
    game = armReaching6targets(max_step, max_epoch)

    
    arm = Arm(shoulder_range, elbow_range)
    
    for epoch in xrange(training_epoch,max_epoch):
        
        if epoch == start_plotting:
            
                               
            fig1   = plt.figure("Workspace", figsize=(16     ,16))
            gs = plt.GridSpec(8,8)
            
            
            
            text1 = plt.figtext(.02, .72, "epoch = %s" % (0), style='italic', bbox={'facecolor':'yellow'})
            text2 = plt.figtext(.02, .62, "trial = %s" % (0), style='italic', bbox={'facecolor':'lightblue'})
            
            text3 = plt.figtext(.30, .20, "linearity index = %s" % (0), style='italic', bbox={'facecolor':'lightblue'})
            
            
            
            ax1 = fig1.add_subplot(gs[0:8, 0:5])
            
         #   line1, = ax1.plot([0, arm.xElbow, arm.xEndEf], [0, arm.yElbow, arm.yEndEf], 'k-', color = 'pink', linewidth = 10)
      #      point, = ax1.plot([arm.xEndEf], [arm.yEndEf], 'o', color = 'black' , markersize= 20)
       
            
            
            ax1.set_xlim([-0.7,0.5])
            ax1.set_ylim([-0.2,0.8])
            
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
            
        
            ax3 =  fig1.add_subplot(gs[0:1, 6:8])
            ax3.set_ylim([0,1.5])
            ax3.set_xlim([0,max_step])
            ax3.set_yticks(np.arange(0, 1.5, 0.5))
            ax3.set_xticks(np.arange(0, max_step, 20))
            title3 = plt.figtext(.79, 0.90, "VELOCITY" , style='normal', bbox={'facecolor':'orangered'})
            text5 = plt.figtext(.74, .75, "asimmetry index = %s" % (0), style='italic', bbox={'facecolor':'lightblue'})
            
            ax5 =  fig1.add_subplot(gs[2:3, 6:8])
            ax5.set_ylim([-2000,2000])
            ax5.set_xlim([0,max_step])
            ax5.set_yticks(np.arange(-2000, 2000, 1000))
            ax5.set_xticks(np.arange(0, max_step, 20))          
            title5 = plt.figtext(.8, .70, "JERK" , style='normal', bbox={'facecolor':'orangered'})
            text4 = plt.figtext(.72, .55, "smoothness index = %s" % (0), style='italic', bbox={'facecolor':'lightblue'})
            
            
            ax6 =fig1.add_subplot(gs[5:6, 6:8])
            ax6.set_xlim([0,100])
            ax6.set_xticks(np.arange(0, max_step, 20))
            ax6.set_ylim([-1.0, np.pi])
            
            title6 = plt.figtext(.76, .41, "SHOULDER ANGLE" , style='normal', bbox={'facecolor':'orangered'})
            
            ax7 =fig1.add_subplot(gs[7:8, 6:8])
            ax7.set_xlim([0,100])
            ax7.set_xticks(np.arange(0, max_step, 20))
            ax7.set_ylim([0.0, np.pi])
            
            title7 = plt.figtext(.77, .21, "ELBOW ANGLE" , style='normal', bbox={'facecolor':'orangered'})
            
            
            
            
        if epoch >= start_plotting:                
            text1.set_text("epoch = %s" % (epoch +1))
            
        for trial in xrange(1,game.test_trial):
            
            print trial, epoch
            
            gameGoalPos = goalPosition[:,trial,epoch].copy()
            
            
            trialArmAngles = armAngles[:,:,trial,epoch].copy()
            trialGoalAngles = goalAngles[:,:,trial,epoch].copy()
            trialGangliaAngles = gangliaAngles[:,:,trial,epoch].copy()
            
            if cerebellum == True:
                trialCerebAngles = cerebAngles[:,:,trial,epoch].copy()
                
            trialTraj = trajectories[:,:,trial,epoch].copy()
            
            
            
            
            
            
            
            if trialTraj[0,:].any() != 0 :
                trimmedTraj = utils.trimTraj(trialTraj) 
            
                minDistance = utils.distance(trimmedTraj[:,0], trimmedTraj[:,len(trimmedTraj[0,:]) -1])
                trajLen = utils.trajLen(trimmedTraj)
            
            
                trialTangVel = np.zeros(len(trimmedTraj[0,:]))
                
                for step in xrange(len(trimmedTraj[0,:])):
                    
                    if step > 0:
                        trialTangVel[step] = utils.distance(trimmedTraj[:,step],trimmedTraj[:,step -1]) / arm.dt
            
            
            
                trialTangJerk = np.trim_zeros(jerk[:,trial,epoch], 'b')
                
                trialdXdY = np.zeros([ 2, len(trimmedTraj[0,:]) ])
            
                trialdXdY[0,:] = np.ediff1d(trimmedTraj[0,:], to_begin = np.array([0]))  / arm.dt
                trialdXdY[1,:] = np.ediff1d(trimmedTraj[1,:], to_begin = np.array([0]))  / arm.dt
            
           
                trialddXddY = np.zeros([ 2, len(trimmedTraj[0,:]) ])
            
                trialddXddY[0,:] = np.ediff1d(trialdXdY[0,:], to_begin = np.array([0]))  / arm.dt
                trialddXddY[1,:] = np.ediff1d(trialdXdY[1,:], to_begin = np.array([0]))  / arm.dt
                       
            
                       
                       
                trialdddXdddY = np.zeros([ 2, len(trimmedTraj[0,:]) ])
            
                trialdddXdddY[0,:] = np.ediff1d(trialddXddY[0,:], to_begin = np.array([0]))  / arm.dt
                trialdddXdddY[1,:] = np.ediff1d(trialddXddY[1,:], to_begin = np.array([0]))  / arm.dt
                    
                
         #       trialJerk = np.mean(trialdddXdddY)
                
                linearityIndex = (utils.linearityIndex(trajLen, minDistance) -1.) * 100
              #  smoothnessIndex = np.log(np.sqrt(np.mean(trialTangJerk)**2 * (len(trimmedTraj)/ arm.dt)**6 / trajLen**2)/2) 
                
                smoothnessIndex = utils.smoothnessIndex(trialdddXdddY, trajLen, arm.dt)
            
            
                asimmetryIndex = utils.asimmetryIndex(trialTangVel)
                
            if epoch >= start_plotting:
                
                text2.set_text("trial = %s" % (trial +1))
                text3.set_text("linearity index = %s" % (linearityIndex))
                text4.set_text("smoothness index = %s" % (smoothnessIndex))
                text5.set_text("asimmetry index = %s" % (asimmetryIndex))
            
                ax1.cla()
                ax1.set_xlim([-0.7,0.5])
                ax1.set_ylim([-0.2,0.8])
            
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
                
                rewardCircle = plt.Circle((gameGoalPos), goal_range, color = 'red')  
                ax1.add_artist(rewardCircle)
    
    
                traj = plt.Line2D(trimmedTraj[0] , trimmedTraj[1], color = 'red')
                ax1.add_artist(traj)
                
                linearVelocityPlot, = ax3.plot(trialTangVel, color='blue')
                linearJerkPlot, = ax5.plot(trialTangJerk, color='blue')
                
                desiredShoulderAngle, = ax6.plot(np.trim_zeros(trialGoalAngles[0], 'b'), color='red')
                shoulderGangliaAngle, = ax6.plot(np.trim_zeros(trialGangliaAngles[0], 'b'), color='blue')
                currShoulderAngle, = ax6.plot(np.trim_zeros(trialArmAngles[0], 'b'), color='black')
                
            
                desiredElbowAngle, = ax7.plot(np.trim_zeros(trialGoalAngles[1], 'b'), color='red')
                elbowGangliaAngle = ax7.plot(np.trim_zeros(trialGangliaAngles[1], 'b'), color='blue')
                currElbowAngle, = ax7.plot(np.trim_zeros(trialArmAngles[1], 'b'), color='black')
                
                
            
                if cerebellum == True:
                    shoulderCerebAngle, = ax6.plot(np.trim_zeros(trialCerebAngles[0]), color='orange')
                    elbowCerebAngle, = ax7.plot(np.trim_zeros(trialCerebAngles[1]), color='orange')
                    
                plt.pause(2.0)
                
                
                ax3.cla()
                ax3.set_ylim([0,1.5])
                ax3.set_xlim([0,max_step])
                ax3.set_yticks(np.arange(0, 1.5, 0.5))
                ax3.set_xticks(np.arange(0, max_step, 50))
            
         #   ax4.cla()
        #    ax4.set_ylim([0,4])
        #    ax4.set_xlim([0,100])
          #  ax4.set_yticks(np.arange(0, 4, 0.8))
       #     ax4.set_xticks(np.arange(0, 100, 10))
            
                ax5.cla()
                ax5.set_ylim([-2000,2000])
                ax5.set_xlim([0,max_step])
                ax5.set_yticks(np.arange(-2000, 2000, 1000))
                ax5.set_xticks(np.arange(0, max_step, 50))
            
                ax6.cla()
                ax6.set_xlim([0,max_step])
                ax6.set_xticks(np.arange(0, max_step, 50))
                ax6.set_ylim([-1.0, np.pi])
            
                ax7.cla()
                ax7.set_xlim([0,max_step])
                ax7.set_xticks(np.arange(0, max_step, 50))
                ax7.set_ylim([0.0, np.pi])
                
             #   ax8.cla()
             #   ax8.set_xlim([0,max_step])
             #   ax8.set_xticks(np.arange(0, max_step, 20))
             #   ax8.set_ylim([np.deg2rad(-60), np.deg2rad(150)])
                
             #   ax9.cla()
             #   ax9.set_xlim([0,max_step])
              #  ax9.set_xticks(np.arange(0, max_step, 20))
              #  ax9.set_ylim([np.deg2rad(0), np.deg2rad(180)])
                """
                Trajectories_plot = plt.figure("Trajectories Plot", figsize=(16     ,16))
        #        plt.title('trajectories trial=%s ataxia=%s' % (trial, ataxia))
                trajectories_plot  =  Trajectories_plot.add_subplot(111)
                
                trajectories_plot.set_xlim([-0.7,0.5])
                trajectories_plot.set_ylim([-0.2,0.8])
                trajectories_plot.plot(trimmedTraj[0] , trimmedTraj[1], color = 'black')
                
                
                circle1_t = plt.Circle((game.goal_list[0]), goal_range, color = 'yellow') 
                edgecircle1_t = plt.Circle((game.goal_list[0]), goal_range, color = 'black', fill = False) 
                trajectories_plot.add_artist(circle1_t)
                trajectories_plot.add_artist(edgecircle1_t)
            
                circle2_t = plt.Circle((game.goal_list[1]), goal_range, color = 'yellow') 
                edgecircle2_t = plt.Circle((game.goal_list[1]), goal_range, color = 'black', fill = False) 
                trajectories_plot.add_artist(circle2_t)
                trajectories_plot.add_artist(edgecircle2_t)
            
                circle3_t = plt.Circle((game.goal_list[2]), goal_range, color = 'yellow') 
                edgecircle3_t = plt.Circle((game.goal_list[2]), goal_range, color = 'black', fill = False) 
                trajectories_plot.add_artist(circle3_t)
                trajectories_plot.add_artist(edgecircle3_t)
                
                circle4_t = plt.Circle((game.goal_list[3]), goal_range, color = 'yellow') 
                edgecircle4_t = plt.Circle((game.goal_list[3]), goal_range, color = 'black', fill = False) 
                trajectories_plot.add_artist(circle4_t)
                trajectories_plot.add_artist(edgecircle4_t)
                
                rewardCircle_t = plt.Circle((gameGoalPos), goal_range, color = 'red')  
                trajectories_plot.add_artist(rewardCircle_t)
                
                
             #   lin_index = plt.figtext(.30, .20, "linearity index = %s" % (linearityIndex), style='italic', bbox={'facecolor':'lightblue'})

                              
                plt.figure()
                plt.title('linear velocity trial=%s ataxia=%s' % (trial, ataxia))
                plt.xlim([0,max_step])
                plt.ylim([0,2.0])               
                plt.plot(trialTangVel, color='black')
                plt.xlabel("steps")
                plt.ylabel("linear velocity")
                asy_index = plt.figtext(.30, .70, "asimmetry index = %s" % (asimmetryIndex), style='italic', bbox={'facecolor':'lightblue'})
              
              
                
                plt.figure()
                plt.title('shoulder angles variation trial=%s ataxia=%s' % (trial, ataxia))
                plt.plot(np.trim_zeros(trialGoalAngles[0], 'b'), label="desired angles", color='red')                
                plt.plot(np.trim_zeros(trialGangliaAngles[0], 'b'),label="RL shoulder output", color='blue')                
                plt.plot(np.trim_zeros(trialArmAngles[0], 'b'), label="curr shoulder angle",color='black')
                if cerebellum == True:                    
                    plt.plot(np.trim_zeros(trialCerebAngles[0]), label="SL shoulder output",color='orange')
                plt.legend(loc='lower right')
                
                plt.figure()
                plt.title('elbow angles variation trial=%s ataxia=%s' % (trial, ataxia))
                plt.plot(np.trim_zeros(trialGoalAngles[1], 'b'), label="desired angles", color='red')
                plt.plot(np.trim_zeros(trialGangliaAngles[1], 'b'), label="RL elbow output",color='blue')
                plt.plot(np.trim_zeros(trialArmAngles[1], 'b'), label="curr elbow angle", color='black')
                if cerebellum == True:                 
                    plt.plot(np.trim_zeros(trialCerebAngles[1]), label="SL elbow output", color='orange')
                plt.legend(loc='lower right')
                """
                
