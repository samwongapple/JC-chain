import numpy as np
import torch
from torch.autograd import Variable
import torch.optim as optim
from scipy.stats import unitary_group
import matplotlib.pyplot as plt
from quimb import *
from quimb.tensor import *
import sympy as sym

import scipy as sp
from scipy import linalg

"""This file includes functions related to the Jaynes-Cummings chain"""

### MPS part

###### Jaynes-cummings model
def JC_tensors(chi, t): #makes U above as a 4 tensor
    A0, A1, B0, B1 = [np.zeros((chi,chi), dtype = complex), np.zeros((chi,chi),dtype = complex), np.zeros((chi,chi),dtype = complex),np.zeros((chi,chi),dtype = complex)] 

    for n in range(chi-1):
        B1[n,n] = np.cos(t*np.sqrt(n))
        A1[n,n+1] = np.sin(t*np.sqrt(n+1))

        B0[n+1,n] = np.sin(t*np.sqrt(n+1))
        A0[n,n] = np.cos(t*np.sqrt(n+1))

    B1[-1,-1] = np.cos(t*np.sqrt(chi-1))
    B0[-1,-2] = np.sin(t*np.sqrt(chi-1))#this may be wrong

    A1[-2,-1] = np.sin(t*np.sqrt(chi-1))
    A0[-1,-1] = np.cos(t*np.sqrt(chi))
    # need to adjust bottom right corner of B0 and A1 further to make rotor-like unitary

    U = np.zeros((2,2,chi,chi), dtype = complex) #Tensor U_ijnm (2,2,chi,chi)
    U[0,0], U[1,0], U[0,1], U[1,1] = [A0, 1j*A1, 1j*B0, B1]

    return U #this is correct aside from lower block


#Use this function to form JC MPS
def MPS_JC_2site(chi,t): # makes MPS for 2 site unit cell
    U = JC_tensors(chi,t)
    A0,A1,B0,B1 = [U[0,0], U[1,0], U[0,1], U[1,1]] # standard single site MPS tensors

    C = np.zeros((4,chi,chi), dtype = complex) #0 = 00, 1 = 01, 2 = 10, 3 = 11
    C[0], C[1], C[2], C[3] = [np.einsum('ij,jk', A0, B0), np.einsum('ij,jk', A0, B1), np.einsum('ij,jk', A1, B0), np.einsum('ij,jk', A1, B1)]

    return C


####### Motzkin MPS
def MPS_Motzkin(chi, pL, pR):
    Aplus, A0, Aminus = [np.zeros((chi,chi)),np.zeros((chi,chi)),np.zeros((chi,chi))]

    for n in range(chi):
        A0[n,n] = np.sqrt(1-pL-pR)
        if n<chi-1:
            Aplus[n+1,n] = np.sqrt(pR) 

        if n>0:
            Aminus[n-1,n] = np.sqrt(pL)

    Aplus[1,0] = np.sqrt(pL+pR)

    return np.array([Aplus, A0, Aminus])


##########################################
##########################################
##########################################

#Stochastic matrix

# JC Matrix
def M_JC(chi,t): 
    M = np.zeros((chi,chi))

    for n in np.arange(chi):
        M[n, n] = (np.cos(t*np.sqrt(n+1))* np.cos(t*np.sqrt(n)))**2 + np.sin(t*np.sqrt(n+1))**4
        
        if n > 0:
            M[n-1,n] = (np.cos(t*np.sqrt(n+1))* np.sin(t*np.sqrt(n)))**2
        if n < chi-1:
            M[n+1,n] = (np.cos(t*np.sqrt(n+1))* np.sin(t*np.sqrt(n+1)))**2
        
    #modify M[chi-1,chi-1] so that sum of chi'th column =1
    M[-1,-1] += 1- M[:,chi-1].sum()

    return M

# Left, Stay, Right prob for JC (column entries of above matrix)
def Hop_prob(n, t):
    S = (np.cos(t*np.sqrt(n+1))* np.cos(t*np.sqrt(n)))**2 + np.sin(t*np.sqrt(n+1))**4
    L = (np.cos(t*np.sqrt(n+1))* np.sin(t*np.sqrt(n)))**2
    R = (np.cos(t*np.sqrt(n+1))* np.sin(t*np.sqrt(n+1)))**2

    return L, S, R



# Motzkin matrix
def M_randwalk(chi, p):
    M = np.zeros((chi,chi))

    for n in np.arange(1,chi):
        M[n, n] = 1-2*p
        
        if n > 0:
            M[n-1,n] = p
        if n < chi-1:
            M[n+1,n] = p

    M[1,0] = .5
    M[0,0] = .5
        
    #modify M[chi-1,chi-1] so that sum of chi'th column =1
    M[-1,-1] += 1- M[:,chi-1].sum()

    return M

##########################################
##########################################

#fcns on stoch matrix:

def Schmidt_weights(M_list, LA):
    L = len(M_list)
    chi = M_list[0].shape[0]

    ML,MR = [np.eye(chi), np.eye(chi)]
    L_list, R_list = [M_list[0:LA-1], M_list[LA:-1]]
    L_list.reverse()
    R_list.reverse()

    for M in L_list:
        ML = ML @ M

    for M in R_list:
        MR = MR @ M

    norm = (MR @ ML)[0,0]

    SW = []
    for n in range(chi):
        SW.append((MR[0,n]*ML[n,0])/norm)
    
    return np.array(SW)



def SE_fromM(M_list, LA):
    entropy = 0
    SW = Schmidt_weights(M_list, LA)
    for sw in SW:
        if sw > 10**-15:
            entropy -=sw * np.log(sw)/np.log(2)
    
    return entropy