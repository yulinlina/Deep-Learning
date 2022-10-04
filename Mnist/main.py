import math, pickle
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from Mnist.utils import cost, accuracy, fc, bc

if __name__ == '__main__':
    # Step 1: Data Preparation
    m = loadmat("Mnist\mnist_small_matlab.mat")

    trainData, trainLabels = m['trainData'], m['trainLabels']
    testData, testLabels = m['testData'], m['testLabels']
    print(trainData.shape,trainLabels.shape)
    print(testData.shape,testLabels.shape)

    train_size = 10000
    X_train = trainData.reshape(-1, train_size)
    test_size = 2000
    X_test = testData.reshape(-1, test_size)
    print(X_train.shape, X_test.shape)


    alpha = 0.05  # initialize learning rate

    # Step 2: Network Architecture Design
    # define number of layers
    L = int(input("please input the number of layer:\n"))

    # define number of neurons in each layer 
    # - 1st column: external neurons 
    # - 2nd column: internal neurons
    if L == 5:
        layer_size = [784, # number of neurons in 1st layer
                    256, # number of neurons in 2nd layer
                    128, # number of neurons in 3rd layer
                    64,  # number of neurons in 4th layer
                    10]  # number of neurons in 5th layer
    elif  L == 2:
        layer_size = [784,10]
    elif  L == 3:
        layer_size  = [784,128,10]
    else: 
        layer_size =[784, # number of neurons in 1st layer
                    512,# number of neurons in 2nd layer
                    256, # number of neurons in 3nd layer
                    128, # number of neurons in 4rd layer
                    64,  # number of neurons in 5th layer
                    32,  # number of neurons in 6nd layer
                    16,  # number of neurons in 7nd layer
                    10]  # number of neurons in 8th layer
    
    # Step 3: Initializing Network Parameters
    # initialize weights
    w = {}
    for l in range(1, L):
        w[l] = 0.1 * np.random.randn(layer_size[l], layer_size[l-1])
   

    # Step 4,5 see utils.py

    # Step 6: Train the Network
    J = [] # array to store cost of each mini batch
    Acc = [] # array to store accuracy of each mini batch
    max_epoch = 200 # number of training epoch 200
    mini_batch = 100 # number of sample of each mini batch 100
    for epoch_num in range(max_epoch):
        idxs = np.random.permutation(train_size)
        for k in range(math.ceil(train_size/mini_batch)):
            start_idx = k*mini_batch 
            end_idx = min((k+1)*mini_batch, train_size) 

            a, z, delta = {}, {}, {}
            batch_indices = idxs[start_idx:end_idx]
            a[1] = X_train[:, batch_indices]
            y = trainLabels[:, batch_indices]
               
            # forward computation
            for l in range(1, L):
                a[l+1], z[l+1] = fc(w[l], a[l])

            delta[L] = (a[L] - y) * (a[L]*(1-a[L]))  
            # backward computation
            for l in range(L-1, 1, -1):
                # print('aaa')
                delta[l] = bc(w[l], z[l], delta[l+1])

            # update weights
            for l in range(1, L):
                grad_w = np.dot(delta[l+1], a[l].T)
                w[l] = w[l] - alpha*grad_w

            J.append(cost(a[L], y)/mini_batch)
            Acc.append(accuracy(a[L], y))
        
        # Step 7: Test the Network
        a[1] = X_test
        y = testLabels
        # forward computation 
        for l in range(1, L):
            a[l+1], z[l+1] = fc(w[l], a[l])

        print(epoch_num, "training acc:", Acc[-1], 'test acc:', accuracy(a[L], y))
    
    plt.figure()
    plt.plot(J)
    plt.savefig(f"J_{L}.png")
    plt.close()
    plt.figure()               
    plt.plot(Acc)
    plt.savefig(f"Acc_{L}.png")
    plt.close()
    # Step 8: Store the Network Parameters
    # save model
    model_name = f"model_{L}.pk"
    with open(model_name, 'wb') as f:
        pickle.dump([w, layer_size], f)
    print("model saved to {}".format(model_name))
