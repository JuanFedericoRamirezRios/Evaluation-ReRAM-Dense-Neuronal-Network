"""
 * Python 3.14

 * GPL-3.0 license
"""
dllFile = "/ReRAM.dll"

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter # Smooth: Savitzky-Golay
import CLASSIC_DENSE_NN as nn
import ctypes as c
import os

# When import this python file:
path = os.path.dirname(os.path.realpath(__file__))
handle = c.CDLL(path + dllFile, winmode=0) # winmode=0: Use unicode.

handle.InitReRAMlayer.restype = c.c_int
handle.InitPotentiation.restype = c.POINTER(c.c_float)

def CreatePotentiationData(Gmin, Gmax, Pmax, m1, m2, noise):
    p = np.arange(0, Pmax+1) # [0,1,...,Pmax]
    a = (Pmax*(m1+m2)-2*(Gmax-Gmin))/(Pmax**3)
    b = (m2-m1-3*a*Pmax**2)/(2*Pmax)
    Gp = a*p**3 + b*p**2 + m1*p + Gmin
    GaussNoise = np.random.normal(0, noise, len(p))
    Gp = Gp + GaussNoise
    return Gp

def CreateDepressionData(Gmin, Gmax, Pmax, m1, m2, noise):
    p = np.arange(0, Pmax+1) # [0,1,...,Pmax]
    a = (Pmax*(m1+m2)-2*(Gmin-Gmax))/(Pmax**3)
    b = (m2-m1-3*a*Pmax**2)/(2*Pmax)
    Gd = a*p**3 + b*p**2 + m1*p + Gmax
    GaussNoise = np.random.normal(0, noise, len(p))
    Gd = Gd + GaussNoise
    return Gd

def Normalization(G, Wmin, Wmax):
    # assert pulses.shape == G.shape, "Error: dimensions of pulses and G"
    Gmin = G.min()
    Gmax = G.max()
    return ((Wmax-Wmin)/(Gmax-Gmin))*(G-Gmin) + Wmin

def SmoothSG(W, polyOrder = 3):
    """
    W: array of the data
    polyOder: of regression
    """
    window = W.shape[0] # window: Number of data used for the polyOrder regresion
    # window = int(np.sqrt(W.shape[0]) + 1) # For each square root of the total number of data points, use the polyOrder regression
    
    return savgol_filter(W, window, polyOrder)

def dWreram_pulses(Wsmooth):
    dWdp = np.diff(Wsmooth) # dW/dp discret difference: (#pulses-1)
    dWdp = np.append(dWdp,dWdp[-1]) # Same data at the final: (#pulses)
    return dWdp

# Structure RERAM_VALS
# class RERAM_VALS(c.Structure):
#     _fields_ = [
#         (""),
#         ("data", c.POINTER(c.POINTER(c.c_float))),
#         ("rows", c.c_int)
#     ]

def NpVectorToC(numpyList):
    return (c.c_float * numpyList.shape[0])(*numpyList)

def NpArrayToC(numpyArray):
    return (c.c_float * (numpyArray.shape[0] * numpyArray.shape[1]))(*numpyArray.flatten())

def CtoNpArray(cArray, rows, cols):
    # W = np.array(cArray)
    l = cArray[:(rows*cols)] # : -> pass to list
    npArray = np.zeros((rows, cols))
    for row in range(rows):
        for col in range(cols):
            npArray[row,col] = l[row*cols + col]
    
    print(npArray)
    return npArray

    

def FreeMemory():
    handle.FreeMemory()
    


class RERAM_LAYER(): # RERAM matrix
    def __init__(s, Gpot, Gdep, facStd=3, inSize=32, outSize=32, initialization = "Kaiming He", stdDev = 0.001):
        """
        Gpot: (#pulsesPotentiation)
        Gdep: (#pulsesDepression)
        facStd: Times of standard desviation for max and min normalization
        initialization: "constant" || "Kaiming He" || "Xavier" 
        WpotRe, WdepRe: Keys: "p", "Wexp", "Wsm", "dWdp"
        """
        
        s.std = stdDev
        if initialization == "constant":
            pass
        elif initialization == "Kaiming He":
            s.std = 1.0/np.sqrt(inSize/2)
        elif initialization == "Xavier":
            s.std = 1.0/np.sqrt(inSize)
        
        valsWpot = Normalization(Gpot, -facStd*s.std, facStd*s.std)
        valsWdep = Normalization(Gdep, -facStd*s.std, facStd*s.std)

        valsWpot_smooth = SmoothSG(valsWpot, 3)
        valsWdep_smooth = SmoothSG(valsWdep, 3)

        dWpot_dp = dWreram_pulses(valsWpot_smooth)
        dWdep_dp = dWreram_pulses(valsWdep_smooth)

        # print(valsWpot_smooth)
        # print(valsWpot)

        s.numLayer = handle.InitReRAMlayer(
            len(valsWpot), 
            NpVectorToC(valsWpot), 
            NpVectorToC(valsWpot_smooth), 
            len(valsWdep), 
            NpVectorToC(valsWdep), 
            NpVectorToC(valsWdep_smooth), 
            outSize, 
            inSize
        )


        # Join in dictionaries
        # s.Wpot = {"Wexp": valsWpot, "Wsm": valsWpot_smooth, "dWdp": dWpot_dp}
        # s.Wdep = {"Wexp": valsWdep, "Wsm": valsWdep_smooth, "dWdp": dWdep_dp}

        s.W = np.random.randn(outSize, inSize) * s.std

        # for row in range(outSize):
        #     for col in range(inSize):
        #         s.W[row][col] = row*inSize+col


        _ = s.InitPotentiation()
        CtoNpArray(_, outSize, inSize)

        s.W = s.W.view(nn.TENSOR)

        s.b = (np.zeros((outSize, 1))).view(nn.TENSOR) # (outSize, 1)

        s.Wmax = []; s.Wmin = []; s.Wmean = []; s.Wstd = []
    
    def PrintReRAMlayer(s):
        handle.PrintReRAMlayer(s.numLayer)
        

    def InitPotentiation(s):
        # print(CtypesArray(s.W))
        return handle.InitPotentiation(s.numLayer, NpArrayToC(s.W))


    def __call__(s, input): # Occur when: object(input = X). Forward through the layer.
        """
        input: (inSize, #samples)
        """
        z = s.W @ input + s.b # (outSize, #samples)
        return z.view(nn.TENSOR) # Return the scores during the forward.
    
    def Backward(s, input, z): # Backward through layer
        """
        input: (inSize, #samples)
        z: (outSize, #samples)
        """
        # W: (outSize, inSize)
        # dL/dinput = dL/dz * dz/dinput = dL/dz * W
        input.grad = s.W.T @ z.grad # grad will be an attribute of TENSOR class. 
        # (inSize, #samples) = (inSize, outSize)*(outSize, #samples) 

        # dL/dW = dL/dz * dz/dW = dL/dZ * input
        s.W.grad = z.grad @ input.T # Not /batch to normalize, the learningRate will take account the size of batch.  
        # (outSize, inSize) = (outSize, #samples) * (#samples, inSize)

        # z: (outSize, #samples)
        # dL/db = dL/dz * dz/db = dL/dz
        s.b.grad = np.sum(z.grad, axis = 1, keepdims=True) # Sum elements of each row (axis=1) and keep the (outSize) rows.
        # (outSize, 1)
    def Learning(s, learningRate):
        s.W = s.W - learningRate*s.W.grad # (outSize, inSize)
        s.b = s.b - learningRate*s.b.grad # (outSize, 1)

        s.Wmax.append(s.W.max())
        s.Wmin.append(s.W.min())
        s.Wmean.append(s.W.mean())
        s.Wstd.append(s.W.std())



    

    


    

