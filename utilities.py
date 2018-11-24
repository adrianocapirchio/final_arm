# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 16:09:02 2018

@author: Alex
"""

import numpy as np
import scipy as scy


def changeRange(old_value, old_min, old_max, new_min, new_max):
    return (((old_value - old_min) * (new_max - new_min)) / (old_max - old_min)) + new_min

def limitRange(x, x_min, x_high):
    return np.maximum(x_min, np.minimum(x_high,x))


def gaussian(x, mu, sig):
    return np.exp(-((x - mu)**2/ (2 * sig ** 2.)))


def rbf2D(position, activation, grid, sig):
    return  np.exp(-distance(position.reshape(-1, 1), grid, axis=0)**2 / (2 * sig ** 2.)) 
   
    #activation =  np.exp(-distance(position.reshape(-1, 1), grid, axis=0)**2 / (2 * sig ** 2.)) 
    # for i in xrange(len(grid[0])):
    #     activation[i] = np.exp(-distance(position,grid[:,i])**2 / (2 * sig ** 2.))        
    #     #activation[i] = np.exp(-np.linalg.norm(position - grid[:,i])**2 / (2 * sig ** 2.))        
    #return activation


def gaussian2D(curr2DPos, mu, sig, inputArray, intervals):
    for j in xrange(intervals +1):
        for i in xrange(intervals +1):
            inputArray[i,j] =  np.exp(- (np.sum((curr2DPos -mu[:,i,j])**2) / (2 * sig ** 2.)))       
    return inputArray

def gaussian3D(currAngles, mu, sig, wristRawPosState, interN):
    
    for k in xrange(interN +1):
        for j in xrange(interN +1):
            for i in xrange(interN +1):
                wristRawPosState[i,j,k] = np.exp(- np.sum(sqrDis(currAngles,mu[:,i,j,k]))  / (2 * sig ** 2.))
    return wristRawPosState








def clipped_exp(x):
    cx =np.clip(x, -700, 700)
    return np.exp(cx)

def sigmoid(x):
    return 1 / (1.0 + np.e**(-(1.0*x)))

def sigmoid2(x):
    return 1 / (1.0 + np.e**(-(1.0*x)))

def sigmoidtDCS(x, tDCSMag):
    I = (1 / (1.0 + np.e**(-(1.0*x)))) 
    if I > 0.5:
        I = I + tDCSMag/100
    else:
        I = I - tDCSMag/100
    return I




def distance(a,b,axis=None):
    return np.sqrt(np.sum((a - b)**2, axis=axis))



def build3DGrid(intervals, rangeMin, rangeMax):
    
    x = np.linspace(rangeMin[0],rangeMax[0],intervals)
    y = np.linspace(rangeMin[1],rangeMax[1],intervals)
    z = np.linspace(rangeMin[2],rangeMax[2],intervals)

    xxx, yyy, zzz = np.meshgrid(x,y,z)

    end = np.zeros([3,intervals,intervals,intervals])    
    end[0,:,:,:] = xxx
    end[1,:,:,:] = yyy
    end[2,:,:,:] = zzz
        
    return end

def build2DGrid(intervals, rangeMin, rangeMax):
    x = np.linspace(rangeMin, rangeMax ,intervals)
    y = np.linspace(rangeMin, rangeMax ,intervals)
    xx, yy = np.meshgrid(x, y)
    
    end = np.zeros([2,intervals,intervals])  
    end[0,:,:] = xx
    end[1,:,:] = yy
        
    return end

def buildGrid(minWidth, maxWidth, pointsWidth, minHeight, maxHeight, pointsHeight):
    xPoints = np.linspace(minWidth, maxWidth ,pointsWidth)
    yPoints = np.linspace(minHeight, maxHeight ,pointsHeight)
    xx, yy = np.meshgrid(xPoints, yPoints)
    
    end = np.zeros([2,pointsWidth,pointsHeight])  
    end[0,:,:] = xx
    end[1,:,:] = yy
     
    return end
    







def rotation2D(coord,angle):   
    rad = np.deg2rad(angle)
    rotCoords = np.zeros(2)

    rotCoords[0] = coord[0] * np.cos(rad) - coord[1] * np.sin(rad)
    rotCoords[1] = coord[0] * np.sin(rad) + coord[1] * np.cos(rad)
    
    return rotCoords


def skew(vec):    
    hat = np.zeros([len(vec),len(vec)])  
    hat[0,1] = -vec[2]
    hat[0,2] = vec[1]
    hat[1,0] = vec[2]
    hat[1,2] = -vec[0]
    hat[2,0] = -vec[1]
    hat[2,1] = vec[0]
    return hat

def rotationMatrix(currAngles):
       
    ex = np.array([1,0,0])
    exHat = skew(ex)
    
    ey = np.array([0,1,0])
    eyHat = skew(ey)
    
    ez = np.array([0,0,1])
    ezHat = skew(ez)
    
    return np.dot(np.dot(scy.linalg.expm(-exHat * currAngles[0]), scy.linalg.expm(ezHat * currAngles[2])),scy.linalg.expm(eyHat * currAngles[1]))

def rotationVector(R):
    
    theta = np.arccos((np.trace(R)- 1)/2)
    
  #  print theta
    
    r = (theta/ (2*np.sin(theta))) * np.array([ R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1] ])
    
    return r




def trimTraj(traj):
    
    trajdx = np.trim_zeros(traj[0,:], 'b')
    trajdy = np.trim_zeros(traj[1,:], 'b')
    
    trimmedTraj = np.vstack([trajdx, trajdy])
    
    return trimmedTraj


def trajLen(traj):
    
    trajLen = 0.
    
    trajdx = np.trim_zeros(traj[0,:], 'b')
    trajdy = np.trim_zeros(traj[1,:], 'b')
    
    trimmedTraj = np.vstack([trajdx, trajdy])
    
    for t in xrange(len(trimmedTraj[0,:])):
        if t > 0:
            trajLen += distance(trimmedTraj[:, t], trimmedTraj[:,t -1])
            
    return trajLen




def linearityIndex(trajLen, minDistance):
    
    return (trajLen / minDistance)




def smoothnessIndex(trajJerk, distance, armdt):
    
    mJerk = trajJerk[0,:]**2 + trajJerk[1,:]**2 * (((len(trajJerk[0,:]) * armdt) ** 5) / (distance **2))    
    totJerk = np.mean(mJerk)
    teuJerk = np.sqrt(totJerk)/2
 #   print J2
 #   raw_input()
   # J2int = np.sum(J2)
    logJ2int = np.log(teuJerk)
    
    return logJ2int

def asimmetryIndex(velPath):
    
    velPath = np.trim_zeros(velPath, 'b')
    
    prePeak = np.argmax(velPath)
 #   print prePeak
    
    postPeak = len(velPath[prePeak:])
    
    if prePeak != 0:
        return float(postPeak)/float(prePeak)
    

def smoothnessIndex1(jerkPath, distance):
    
    jerkPath = np.trim_zeros(jerkPath, 'b')
    normalizedJerkPath = jerkPath**2 * (len(jerkPath)**6 / distance**2)
  
    mJerk = np.linalg.norm(normalizedJerkPath)  
 #   print J2
 #   raw_input()
   # J2int = np.sum(J2)
    logJ2int = np.log(mJerk)
    
    return logJ2int


def grid_2D(x0,x1,y0,y1,xN,yN):
    rawGrid = np.meshgrid(np.linspace(x0,x1,xN),np.linspace(y0,y1,yN))
    grid = np.vstack(([rawGrid[0].T],[rawGrid[1].T])).T
    return grid

def angfrompos(arml, forarml, endeffX, endeffY, shoRange , elbRange):
    r2     = 0.0
    r      = 0.0
    Beta   = 0.0
    
    r2 = pow(endeffX,2) + pow(endeffY,2)
    r = np.sqrt(r2)
    
    theta2 = 3.14159 - np.arccos( (r2 - pow(arml,2) - pow(forarml,2) ) / ( (-2.0) * arml * forarml))
    Beta = np.arccos(( pow(forarml,2) - pow(arml,2) - r2) / (-2.0 * arml * r))
    theta1 = np.arctan2(endeffY, endeffX) - Beta
                       
    normTheta2 = changeRange(theta2, elbRange[0], elbRange[1], 0.0 , 1.0)
    normTheta1 = changeRange(theta1, shoRange[0], shoRange[1], 0.0 , 1.0)
    
    return np.array([normTheta1, normTheta2]), np.array([theta1, theta2])  
    
    
    
    
    
    
    
    
