"""
 * Python 3.14

 * GPL-3.0 license
"""
import numpy as np
from GET_MNIST import ObtainImages, ObtainLabels
import matplotlib.pyplot as plt
import CLASSIC_DENSE_NN as nn
import RERAM_DENSE_NN as reram

##################################################################
np.random.seed(3)
##################################################################
mnistPath = "./MNIST/"
pathTrainingImages = mnistPath + "train-images.idx3-ubyte"
pathTrainingLabels = mnistPath + "train-labels.idx1-ubyte"
pathTestingImages = mnistPath + "t10k-images.idx3-ubyte"
pathTestingLabels = mnistPath + "t10k-labels.idx1-ubyte"

trainImages28x28 = ObtainImages(pathTrainingImages) # (60000, 28, 28)
trainLabelsList = ObtainLabels(pathTrainingLabels) # (60000,)
testImages28x28 = ObtainImages(pathTestingImages) # (10000, 28, 28)
testLabelsList = ObtainLabels(pathTestingLabels) # (10000,)

trainImages = trainImages28x28[:50000].reshape(50000, -1).astype(np.float32) # -1 is used to infer the dimension from the other dimensions. It is 28*28=784. (50000, 784)
trainLabels = trainLabelsList[:50000].reshape(50000, 1) # (50000, 1)

valiImages = trainImages28x28[50000:].reshape(10000, -1).astype(np.float32) # Validation set -> Fine tuning. (10000, 784)
valiLabels = trainLabelsList[50000:].reshape(10000, 1) # (10000, 1)

testImages = testImages28x28.reshape(10000, -1).astype(np.float32) # (10000, 784)
testLabels = testLabelsList.reshape(10000, 1) # (10000, 1)

# print(trainImages.min(), trainImages.max(), trainImages.mean(), trainImages.std())
#######################################################################
trainMean = trainImages.mean()
trainStd = trainImages.std()

# Using the trainMean and trainStd is statistically corret. Why???
trainImages = nn.Normalize(trainMean, trainStd, trainImages) # (50000, 784)
valiImages = nn.Normalize(trainMean, trainStd, valiImages) # (10000, 784)
testImages = nn.Normalize(trainMean, trainStd, testImages) # (10000, 784)
# trainImages.std() = 1.0
# trainImages.mean() = 0.0
# print(trainImages.mean(), trainImages.std())
#######################################################################
rnd = np.random.randint(len(testImages28x28))
# print(f'The correct label of image is {testLabelsList[rnd]}')
# plt.figure("exampleMNIST")
# nn.PlotImage(testImages28x28[rnd])
#######################################################################
Gmin = 5.6e-4
Gmax = 6.2e-4

# Potentiation
Ppot_max = 40
Mlin = (Gmax-Gmin)/Ppot_max
m1 = Mlin*3
m2 = Mlin*0.2
noise = (Gmax-Gmin)*0.02
Gpot = reram.CreatePotentiationData(Gmin, Gmax, Ppot_max, m1, m2, noise)
# plt.plot(Ppot, Gpot, ".")

# Depression
Pdep_max = 50
Mlin = (Gmin-Gmax)/Pdep_max
m1 = Mlin*2
m2 = Mlin*0
noise = (Gmax-Gmin)*0.02
Gdep = reram.CreateDepressionData(Gmin, Gmax, Pdep_max, m1, m2, noise)
# plt.plot(Pdep, Gdep, ".")
#######################################################################
# Charge Ppot (pulses) and Gpot <- Experimental data
#                .
#                .
#                .

valsWpot = reram.Normalization(Gpot, -0.01, 0.01)
valsWdep = reram.Normalization(Gdep, -0.01, 0.01)

valsWpot_smooth = reram.SmoothSG(valsWpot, 3)
valsWdep_smooth = reram.SmoothSG(valsWdep, 3)

dWpot_dp = reram.dWreram_pulses(valsWpot_smooth)
dWdep_dp = reram.dWreram_pulses(valsWdep_smooth)

# Join in dictionaries
valsWpotRe = {"Wexp": valsWpot, "Wsm": valsWpot_smooth, "dWdp": dWpot_dp}
valsWdepRe = {"Wexp": valsWdep, "Wsm": valsWdep_smooth, "dWdp": dWdep_dp}
#######################################################################
# plt.figure("Ws")
# plt.figure(figsize=(2,2)) 
# plt.plot(np.arange(len(valsWpotRe["Wexp"])), valsWpotRe["Wexp"], ".")
# plt.plot(np.arange(len(valsWpotRe["Wexp"])), valsWpotRe["Wsm"], "-")
# plt.plot(np.arange(len(valsWdepRe["Wexp"])), valsWdepRe["Wexp"], ".")
# plt.plot(np.arange(len(valsWdepRe["Wexp"])), valsWdepRe["Wsm"], "-")
# plt.show()
#######################################################################
# plt.figure(figsize=(2,2)) 
# plt.plot(np.arange(len(valsWpotRe["Wexp"])), valsWpotRe["dWdp"], ".")
# plt.show()
#######################################################################
# plt.figure(figsize=(2,2)) 
# plt.plot(np.arange(len(valsWdepRe["Wexp"])), valsWdepRe["dWdp"], ".")
# plt.show()

handle = 0
first = True
ex = 0
values = np.array([5, 7, 8, 3, 6, 8, 9, 6, 3, 2])

# array = (c.c_int * 10)(*values.tolist())

# print(ex)
# reram.InitReRAMlayer(9, array, array, 9, array, array, 10, 10)
# print(ex)
# reram.InitReRAMlayer(9, array, array, 9, array, array, 10, 10)
# print(ex)