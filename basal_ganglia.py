# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 16:30:11 2018

@author: Alex
"""

import numpy as np
import utilities as utils

class actorCritic:
    
    def __init__(self, proprioception_input, vision, goal_vision_input, ef_vision_input, multinet, goal_list, LMA, max_step, max_trial, DOF=2):
        
        
        "time integration parameter"
        self.dt = 1.0/12
        self.TAU = 1.0 
        self.C1 = self.dt/ self.TAU
        self.C2 = 1. - self.C1
        
        self.noise_dt = 0.01
        self.noise_TAU = 1.0
        self.noise_C1 = self.noise_dt/ self.noise_TAU
        self.noise_C2 = 1. - self.noise_C1
        
        
        "noise param"
        self.noise_sigma = 4.0
        
        "learning parameters"
        self.ACT_ETA =  8.3 * 10 ** (-2)
        self.CRIT_ETA = 8.3 * 10 ** (-3)
        self.DISC_FACT = 0.99
        
        "init input state array"
        self.curr_state = np.array([])
        
        
        "init proprioception input"
        if proprioception_input == True:
            
            self.proprio_state = np.array([])
            
            self.proprio_input_units = 101**2
            self.proprio_intervals = int(np.sqrt(self.proprio_input_units)) -1
            self.proprio_sigma = 1. / (self.proprio_intervals * 2)
            self.proprio_grid = utils.buildGrid(0.0, 1.0, int(np.sqrt(self.proprio_input_units)), 0.0, 1.0, int(np.sqrt(self.proprio_input_units))).reshape(2,self.proprio_input_units)
            self.proprio_state = np.zeros(self.proprio_input_units)
            self.curr_state = np.zeros(len(self.curr_state)+len(self.proprio_state))
        
    #    if vision == True:
            
    #        self.vision_state = np.array([])
        
        "init goal vision"
        if goal_vision_input == True:
            
            self.goal_vision_state = np.array([])
                    
            self.goal_input_units = 51**2
            self.goal_vision_intervals = int(np.sqrt(self.goal_input_units)) -1  
            self.goal_vision_sigma= 1. / ((self.goal_vision_intervals)* 2)
            self.goal_vision_grid = utils.buildGrid(0.0,1.0,int(np.sqrt(self.goal_input_units)),0.0,1.0,int(np.sqrt(self.goal_input_units))).reshape(2,self.goal_input_units)
            self.goal_vision_state = np.zeros(self.goal_input_units)    
          #  self.vision_state = np.zeros(len(self.vision_state) + len(self.goal_vision_state))                      
            self.curr_state = np.zeros(len(self.curr_state)+len(self.goal_vision_state)) 
            
       #     "init end effector vision"
       #     if ef_vision_input == True:
                
        #        self.ef_vision_state = np.array([])
                        
         #       self.ef_input_units = 51**2
         #       self.ef_vision_intervals = int(np.sqrt(self.ef_input_units)) -1  
          #      self.ef_vision_sigma= 1. / ((self.ef_vision_intervals)* 2)
          #      self.ef_vision_grid = utils.buildGrid(0.0,1.0,int(np.sqrt(self.ef_input_units)),0.0,1.0,int(np.sqrt(self.ef_input_units))).reshape(2,self.ef_input_units)
           #     self.ef_vision_state = np.zeros(self.ef_input_units)    
           #     self.vision_state = np.zeros(len(self.vision_state) + len(self.ef_vision_state))
             
            
          #  self.curr_state = np.zeros(len(self.curr_state)+len(self.vision_state))
            
        "init past state array"    
        self.prv_state = self.curr_state.copy()
        
        "init weights"
        self.w_act = np.zeros([len(self.curr_state), DOF])# + np.random.uniform(0.0, 0.01, self.curr_state )
        self.w_crit= np.zeros(len(self.curr_state)) #+ np.random.uniform(0.0, 0.01, self.curr_state )
        
        if multinet == True:
            self.multi_w_act= np.zeros([len(self.curr_state), DOF, max_trial])# + np.random.uniform(0.0, 0.01, self.curr_state )        
            self.multi_w_crit= np.zeros([len(self.curr_state), max_trial])# + np.random.uniform(0.0, 0.01, self.curr_state )
         
        "init critic param"
        self.curr_rew = 0
        self.prv_rew = 0
        
        
        self.curr_crit_U = np.zeros(1)
        self.prv_crit_U = np.zeros(1)
        self.surp = np.zeros(1)
        
        
        "init actor param"
        self.curr_act_U = np.zeros(DOF)
        self.prv_act_U = self.curr_act_U.copy()
        
        self.curr_act_I = np.ones(DOF) * 0.5
        self.prv_act_I = self.curr_act_I.copy()
        
        self.des_noise = np.zeros(DOF)
        self.curr_noise = np.zeros(DOF)
        self.prv_noise = np.zeros(DOF)
        
        "init memory buffer"
        self.state_buffer = np.zeros([len(self.curr_state) , max_step])
        self.act_I_buffer = np.ones([DOF, max_step ]) * 0.5

        "init last epoch perfomance buffer"
        self.training_performance = np.zeros([LMA, 7])
        
    
    def epoch_reset(self, max_step, DOF = 2):
        
        "reset critic param"
        self.curr_rew = 0
        self.prv_rew = 0
        self.curr_crit_U = np.zeros(1)
        self.prv_crit_U = np.zeros(1)
        self.surp = np.zeros(1)
        
        "reset actor param"
        self.curr_act_U = np.zeros(DOF)
        self.prv_act_U = self.curr_act_U.copy()
        
        self.curr_act_I = np.ones(DOF) * 0.5
        self.prv_act_I = self.curr_act_I.copy()
        
        self.des_noise = np.zeros(DOF)
        self.curr_noise = np.zeros(DOF)
        self.prv_noise = np.zeros(DOF)
        
        "reset state"
        self.curr_state *= 0
        self.prv_state *= 0
        
        
        "reset memory buffer"
        self.state_buffer = np.zeros([len(self.curr_state) , max_step])
        self.act_I_buffer = np.ones([DOF, max_step ]) * 0.5

    
    def trial_reset(self, max_step, DOF = 2):
        
        "reset critic param"
        self.curr_rew = 0
        self.curr_crit_U = np.zeros(1)
        self.prv_crit_U = np.zeros(1)
        self.surp = np.zeros(1)
        
     #   self.des_noise = np.zeros(DOF)
     #   self.curr_noise = np.zeros(DOF)
     #   self.prv_noise = np.zeros(DOF)
        
        """
        "reset actor param"
        self.curr_act_U = np.zeros(2)
        self.prv_act_U = self.curr_act_U.copy()
        
        self.curr_act_I = np.ones(2) * 0.5
        self.prv_act_I = self.curr_act_I.copy()
        
        
        
        "reset state"
        self.curr_state *= 0
        self.prv_state *= 0
        
        "reset noise"
        self.des_noise = np.zeros(2)
        self.curr_noise = np.zeros(2)
        self.prv_noise = np.zeros(2)
        """
        
        
        "reset memory buffer"
        self.state_buffer = np.zeros([len(self.curr_state) , max_step])
        self.act_I_buffer = np.ones([DOF, max_step ]) * 0.5
    


    "acquire input states"                           
    def acquire_goal_state(self, goal_position):
        self.goal_vision_state = utils.rbf2D(goal_position, self.goal_vision_state,self.goal_vision_grid,self.goal_vision_sigma)

                
    def acquire_proprio_state(self, arm_angles):
        self.proprio_state= utils.rbf2D(arm_angles,self.proprio_state,self.proprio_grid,self.proprio_sigma)

    def acquire_ef_vision_state(self, ef_position):
        self.ef_vision_state = utils.rbf2D(ef_position, self.ef_vision_state,self.ef_vision_grid,self.ef_vision_sigma)

        
    "compute actor potential"
    def compute_actor_U(self):
        self.curr_act_U = self.C2 * self.curr_act_U + self.C1 * np.dot(self.w_act.T, self.curr_state)
    
    "compute actor frequency"
    def compute_actor_I(self):
        self.curr_act_I = utils.sigmoid(self.curr_act_U)
                     
    

    "compute crit potential"                 
    def compute_crit_U(self): 
        self.curr_crit_U = self.C2 * self.curr_crit_U + self.C1 * np.dot(self.w_crit, self.curr_state)
    
    "compute TD error"    
    def compute_surprise(self):
        self.surp = self.curr_rew + (self.DISC_FACT * self.curr_crit_U) - self.prv_crit_U
    

    


    "compute noise"
    def compute_noise(self,T, DOF = 2): 
        self.des_noise = np.random.normal(0.0, self.noise_sigma, DOF)
        self.curr_noise = utils.limitRange(self.noise_C2 * self.curr_noise + self.noise_C1 * self.des_noise, -1.0, 1.0) * T
        
    
    
    
    
    
    
    "training function"
    def train_crit(self):
        self.w_crit += self.CRIT_ETA * self.surp * self.prv_state
        
    def train_act(self, prv_netOut):    
        self.w_act += self.ACT_ETA * self.surp * np.outer(self.prv_state, prv_netOut + self.prv_noise - self.prv_act_I)  * self.prv_act_I * (1.- self.prv_act_I) 
        
               
                #netout +noise -outbg
        
                
    
            
                
            
            
            
        
        