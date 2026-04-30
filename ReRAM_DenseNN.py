"""
Python 3.14
"""


import matplotlib.pyplot as plt
import numpy as np

def Normalize(mean, std, data): 
    """
    data: (#samples, #pixels)
    """
    return (data-mean)/std # Normalize to mean=0 & std=1. (#samples, #pixels)

def PlotImage(image):
    plt.figure(figsize=(1,1)) # Figure size 1*100x1*100
    plt.imshow(image, cmap="gray")
    plt.show()

def GeneratorBatches(images, labels, batch=64, shuffle=True):  # The generators are iterable functions.
    """
    images: (#samples, #pixels)
    labels: (#samples, 1)
    """
    assert images.shape[0] == labels.shape[0], "Error: Images and labels must have the same number of samples." # assert is a debugging statement that will raise an error if the condition is false.
    numData = images.shape[0]
    if shuffle:
        indices = np.arange(numData) # indices = [0, 1, 2,..., numData-1]
        np.random.shuffle(indices) # indices = shuffled version of itself.
        images = images[indices]
        labels = labels[indices]
    return ((images[n:n+batch], labels[n:n+batch]) for n in range(0, numData, batch)) # Return a iterable object (List comprehension in python).

class TENSOR(np.ndarray): # Subclass derived from the np.ndarray
    """
    # Subclass derived from the np.ndarray.
    """
    pass

class LINEAR_LAYER(): # Identity activation function.
    def __init__(s, inSize, outSize, initialization = "Kaiming He"):
        """
        initialization: "Normal" || "Kaiming He" || "Xavier" 
        """
        s.factorInit = 0.001
        if initialization == "Normal":
            pass
        elif initialization == "Kaiming He":
            s.factorInit = 1.0/np.sqrt(inSize/2)
        elif initialization == "Xavier":
            s.factorInit = 1.0/np.sqrt(inSize)
        s.W = np.random.randn(outSize, inSize) * s.factorInit
        s.W = s.W.view(TENSOR)

        s.b = (np.zeros((outSize, 1))).view(TENSOR) # (outSize, 1)

        s.Wmax = []; s.Wmin = []

    def __call__(s, input): # Occur when: object(input = X). Forward through the layer.
        """
        input: (inSize, #samples)
        """
        z = s.W @ input + s.b # (outSize, #samples)
        return z.view(TENSOR) # Return the scores during the forward.
    
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
        




class RELU_LAYER(): # ReLU activation function.
    def __call__(s, z): # Forward through the activation function.
        """
        z: (inSize, #samples)
        """
        return np.maximum(0, z) # return a: (inSize, #samples)
    
    def Backward(s, z, a): # Backward through activation function.
        """
        z: (inSize, #samples)
        a: (inSize, #samples)
        """
        # dL/dz = dL/da * da/dz = dL/da * d(ReLU)
        z.grad = a.grad.copy() # (inSize, #samples)
        z.grad[z <= 0] = 0

class SEQUENTIAL(): # Through every layers
    def __init__(s, layers):
        '''
        layers: List of layers, type LINEAR or RELU
        '''
        s.layers = layers
        s.outputs = {} # It is a dictionary: L0, L1, L2, ....

    def __call__(s, Xbatch): # Forward through every layers.
        """
        Xbatch: (#pixels, #samples), inputs of first layer.
        """
        s.outputs['L0'] = Xbatch # L0 is the input layer: (#pixels, #samples)
        for n, layer in enumerate(s.layers, 1): # From n = 1 to len(s.layers), 0 is the Xbatch.
            s.outputs['L' + str(n)] = layer(s.outputs['L' + str(n-1)]) # __call__ of LAYER or RELU.
            # (outSize, #samples)            (inSize, #samples)
        return s.outputs['L' + str(len(s.layers))] # Return the output of last layer. 
    
    def Backward(s): # Backward through every layers.
        for n in reversed(range(len(s.layers))): # Ex len(s.layers) = 3: n = 2, 1, 0
            s.layers[n].Backward(s.outputs["L" + str(n)], s.outputs["L" + str(n+1)])
            #                     (inSize, #samples)        (outSize, #samples)
            # For LINEAR -> s.outputs["L" + str(n)].grad: (inSize, #samples)
            #               s.W.grad: (outSize, inSize)
            #               s.b.grad: (outSize, 1)
            # For RELU -> s.outputs["L" + str(n)].grad: (inSize, #samples)
    def Learning(s, learningRate=1e-3):
        for layer in s.layers:
            if isinstance(layer, RELU_LAYER): continue

            layer.Learning(learningRate)
            
    def GraphWmaxWmin(s):
        Wmax = []; Wmin = []
        
        
        for layer in s.layers:
            if isinstance(layer, RELU_LAYER): continue

            # print(len(layer.Wmax), end= " ")
            plt
            Wmax.append(layer.Wmax)
            # print(len(layer.Wmin))
            Wmin.append(layer.Wmin)
            
        Wmax = np.array(Wmax) # (#LinearLayers, #Learnings)
        Wmin = np.array(Wmin) # (#LinearLayers, #Learnings)
        # print()
        # print(Wmax.shape)
        # print(Wmin.shape)
        
        columns = 3
        rows = len(Wmax) // columns
        if (len(Wmax) % columns) > 0:
            rows = rows + 1
        for n, Ws in enumerate(Wmax):
            plt.subplot(rows,columns,n)
            


        
        
        



    
    def Predict(s, Xbatch): # Predict of a sample
        '''
        Xbatch: (#pixels, 1), data of the image.
        '''
        return np.argmax(s.__call__(Xbatch)) # s.__call__(Xbatch): (outSize, 1)

def SoftmaxXentropy(outLastLayer, labels):
    '''
    outLastLayer: (outSize, #samples)
    labels: (#samples, 1)
    '''
    batch = outLastLayer.shape[1]
    exp = np.exp(outLastLayer) # (outSize, #samples)
    # np.sum(exp, axis=0) # axis=0 -> Sum along columns: (#samples)
    probs = exp/np.sum(exp, axis=0) # Softmax: (outSize, #samples)

    # List of predictions
    eloss = probs[labels.squeeze(), np.arange(batch)] # labelsBatch.squeeze(): (#samples). eloss: (#samples)
    cost = np.sum(-np.log(eloss)) / batch # np.log(eloss): (#samples). cost is a number

    # Update the dL/d(outLastLayer)
    outLastLayer.grad = probs.copy()
    outLastLayer.grad[labels.squeeze(), np.arange(batch)] -= 1 # \hat{y}-y : (outSize, #samples)

    return probs, cost

def Accuracy(model, images, labels, batch=64):
    """
    model: Exm of a model: a SEQUENTIAL object.
    images: (#samples, #pixels)
    labels: (#samples, 1)
    """
    total = 0
    numCorrect = 0

    for (imagesBatch, labelsBatch) in GeneratorBatches(images, labels, batch):
        # FORWARD:
        outLastLayer = model(imagesBatch.T.view(TENSOR)) # __call__ of the object model -> __call__ of the SEQUENTIAL object.
        # outLastLayer: (outSize, batch)

        # np.argmax(outLastLayer, axis=0) # List of the index of maximum value of each column. (batch)
        numCorrect += np.sum(np.argmax(outLastLayer, axis=0) == labelsBatch.squeeze())
        total += outLastLayer.shape[1]

    return numCorrect/total

def Training(model, epochs, trainImages, trainLabels, valiImages, valiLabels, batch=128, learningRate=1e-3): 
    """
    :param model: Exm of a model: a SEQUENTIAL object.
    :param trainImages: (#trainSamples, #pixels)
    :param trainLabels: (#trainSamples, 1)
    :param valiImages: (#valiSamples, #pixels)
    :param valiLabels: (#valiSamples, 1)
    """
    for epoch in range(epochs):
        for n, (imagesBatch, labelsBatch) in enumerate(GeneratorBatches(trainImages, trainLabels, batch)):
            # imagesBatch.T: (#pixels, batch)
            # labelsBatch: (batch, 1)

            # FORWARD:
            outLastLayer = model(imagesBatch.T.view(TENSOR)) # __call__ of the object model -> __call__ of the SEQUENTIAL object.
            # outLastLayer: (outSize, batch) 
            probs, cost = SoftmaxXentropy(outLastLayer, labelsBatch) # probs: (outSize, batch)
            if n == 0:   
                firstCost = cost

            # BACKWARD:
            model.Backward()

            # LEARNING:
            model.Learning(learningRate)

            #Obtain array of Wmax and Wmin of each learning:



        print(f'Epoch: {epoch}, Cost of first batch of train images: {firstCost}, Accuracy validation images: {Accuracy(model, valiImages, valiLabels, batch)}')






    

    


    

