# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 13:33:12 2018

@author: Alex
"""

import numpy as np
import utilities as utils

class Cerebellum():
    
    def __init__(self, multinet, stateBg, max_trial, DOF = 2):
               
        "time integration parameter"
        self.dT = 0.99# delta time
        self.tau  = 1.0 # potential tau 
        self.tau1 = 1.0 # frequency tau
        self.C1 = self.dT / self.tau # integration param
        self.C2 = 1. - self.C1 # integration param
        
        "learning rate"
        self.ETA = 9.9 * 10 ** (-1)
        
        "perceptron curr state"
        self.curr_state = np.zeros(len(stateBg))

        "init weights"
        self.w = np.zeros([len(self.curr_state), DOF]) #+ np.random.uniform(0.0, 0.01, self.curr_state )
        if multinet == True:
            self.w_multi = np.zeros([len(self.curr_state), DOF, max_trial]) #+ np.random.uniform(0.0, 0.01, self.curr_state )
            
        "output values"
        self.curr_U = np.zeros(DOF)
        self.prv_U = self.curr_U.copy()
        
        self.curr_I = np.ones(DOF) * 0.5
        self.prv_I = self.curr_I.copy()
        
        self.damage_I = np.ones(DOF) * 0.5
                               
        "training values"
        self.error_out = np.zeros(DOF)
        
        self.train_U = np.zeros(DOF)
        self.prv_train_U = self.train_U.copy() 
        
        self.train_I = np.ones(DOF) * 0.5
        self.prv_train_I = self.train_I.copy()                     
        
       
    def epoch_reset(self,DOF = 2):
        
        "output values"
        self.curr_U = np.zeros(DOF)
        self.prv_U = self.curr_U.copy()
        
        self.curr_I = np.ones(DOF) * 0.5
        self.prv_I = self.curr_I.copy()
        
        self.damage_I = np.ones(DOF) * 0.5
        
        "training values"
        self.error_out = np.zeros(DOF)
        
        self.train_U = np.zeros(DOF)
        self.prv_train_U = self.train_U.copy() 
        
        self.train_I = np.ones(DOF) * 0.5
        self.prv_train_I = self.train_I.copy()
    
    
    def trial_reset(self, DOF = 2):
        
        """
        "output values"
        self.curr_U = np.zeros(2)
        self.prv_U = self.curr_U.copy()
        
        self.curr_I = np.ones(2) * 0.5
        self.prv_I = self.curr_I.copy()
        
        self.damage_I = np.ones(2) * 0.5
        """
        
        
        
        "training values"
        self.error_out = np.zeros(DOF)
        
        self.train_U = np.zeros(DOF)
        self.prv_train_U = self.train_U.copy() 
        
        self.train_I = np.ones(DOF) * 0.5
        self.prv_train_I = self.train_I.copy()
        
        
        
    
    
    
    "compute perceptron potential"
    def comp_U(self, state):
        self.curr_U = self.C2 * self.curr_U + self.C1 * np.dot(self.w.T, state)
    
    "compute perceptron frequency"    
    def comp_I(self):
        self.curr_I = utils.sigmoid(self.curr_U)
        
    "compute perceptron frequency in tDCS"
    def comp_I_tDCS(self, tDCS_mag):
        self.curr_I = utils.sigmoid2(self.curr_U, tDCS_mag)
        
        
    "train perceptron"    
    def train(self,state, gangliaI, reward, T):
        self.train_U = np.dot(self.w.T, state) # #    self.C2 * self.train_U + self.C1 * 
        self.train_I = utils.sigmoid(self.train_U)
        self.error_out = gangliaI - self.train_I
        self.w +=  T * reward * self.ETA * np.outer(state, self.error_out) * self.train_I * (1. - self.train_I)
    