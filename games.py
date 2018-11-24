# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 16:00:34 2018

@author: Alex
"""

import numpy as np
import utilities as utils


class armReaching6targets():
    
    def __init__(self, max_step, max_epoch):
     
        self.goal_list = [np.array([-0.07 -.04, 0.16 +0.05]),
                         np.array([-0.24 -.04, 0.42 +0.05]),        
                         np.array([ 0.10 -.04, 0.42 +0.05]),
                         np.array([-0.07 -.04, 0.42 +0.05])]
        
        self.training_trial = 7
        self.test_trial = 7
        
        
        self.goal_position = self.goal_list[0].copy()
        self.prv_goal_position = self.goal_position.copy()
        
        self.curr_position = np.zeros(2)
        self.prv_position = np.zeros(2)
        
        self.curr_velocity = np.zeros(1)
        self.prv_velocity = np.zeros(1)
        self.curr_acceleration = np.zeros(1)
        self.prv_acceleration = np.zeros(1)
        self.curr_jerk = np.zeros(1)
        
        
        "init training data saving arrays"
        self.goal_position_training_data= np.zeros([2, self.training_trial, max_epoch])
        self.goal_angles_training_data = np.zeros([2, max_step, self.training_trial, max_epoch])
        
        self.arm_angles_training_data = np.zeros([2, max_step, self.training_trial, max_epoch])
        self.ganglia_angles_training_data = np.zeros([2, max_step, self.training_trial, max_epoch])
        
        self.cereb_angles_training_data  = np.zeros([2, max_step, self.training_trial, max_epoch])
        
        #self.trialmtxAngles = np.zeros([2, max_step, self.training_trial, max_epoch])
        
        self.trajectories_training_data = np.zeros([2, max_step, self.training_trial, max_epoch])
        self.velocity_training_data = np.zeros([max_step, self.training_trial, max_epoch])
        self.acceleration_training_data = np.zeros([max_step, self.training_trial, max_epoch])
        self.jerk_training_data = np.zeros([max_step, self.training_trial, max_epoch])
        
        "init test data saving arrays"
        self.goal_position_test_data= np.zeros([2, self.test_trial, 60])
        self.goal_angles_test_data = np.zeros([2, max_step, self.test_trial, 60])
        
        self.arm_angles_test_data = np.zeros([2, max_step, self.test_trial, 60])
        self.ganglia_angles_test_data = np.zeros([2, max_step, self.test_trial, 60])        
        self.cereb_angles_test_data  = np.zeros([2, max_step, self.test_trial, 60])
                
        self.trajectories_test_data = np.zeros([2, max_step, self.test_trial, 60])
        self.velocity_test_data = np.zeros([max_step, self.test_trial, 60])
        self.acceleration_test_data = np.zeros([max_step, self.test_trial, 60])
        self.jerk_test_data = np.zeros([max_step, self.test_trial, 60])
        
     
    def epoch_reset(self):
        
        self.curr_position = np.zeros(2)
        self.prv_position= np.zeros(2)
        
        self.curr_velocity= np.zeros(1)
        self.prv_velocity= np.zeros(1)
        self.curr_acceleration= np.zeros(1)
        self.prv_acceleration= np.zeros(1)
        self.curr_jerk= np.zeros(1)    
        
    def trial_reset(self):
        
        self.curr_position = np.zeros(2)
        self.prv_position= np.zeros(2)
        
        self.curr_velocity= np.zeros(1)
        self.prv_velocity= np.zeros(1)
        self.curr_acceleration= np.zeros(1)
        self.prv_acceleration= np.zeros(1)
        self.curr_jerk= np.zeros(1)
        
        
        
    
    def compute_goal_distance(self):
        self.distance = utils.distance(self.curr_position,self.goal_position) 
        
        
    def set_training_goals(self, trial):
        
        if trial == 0:
            self.goal_position = self.goal_list[0].copy()       
        elif trial == 1:
            self.goal_position= self.goal_list[1].copy()
        elif trial == 2:
            self.goal_position= self.goal_list[2].copy()
        elif trial == 3:
            self.goal_position= self.goal_list[3].copy()
            
            
    def set_test_goal(self, trial):
        
        if trial%2 == 0:
            self.goal_position= self.goal_list[0].copy()       
        elif trial == 1:
            self.goal_position= self.goal_list[1].copy()
        elif trial == 3:
            self.goal_position= self.goal_list[2].copy()
        elif trial == 5:
            self.goal_position= self.goal_list[3].copy()
    
        
        
    def training_index(self, trial):
        
        for i in xrange(self.training_trial):
            if trial == i:
                self.goalIdx = i


    def goalIndex1(self, trial):
        
        for i in xrange(len(self.goal_list)):
            if (self.goal_position == self.goal_list[i]).all():
                self.goalIdx = i
                
                
    def stop_move(self):
        
        self.curr_velocity= np.zeros(1)
        self.prv_velocity= np.zeros(1)
        self.curr_acceleration= np.zeros(1)
        self.prv_acceleration= np.zeros(1)
        self.curr_jerk= np.zeros(1)
        
        
        
      #  if trial%2 == 0:
      #      self.goalIdx = 0
      #  elif trial == 1:
      #      self.goalIdx = 1
      #  elif trial == 3:
      #      self.goalIdx = 2
      #  elif trial == 5:
      #      self.goalIdx = 3
        
                     
                
                
                