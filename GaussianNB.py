import numpy as np
"""
Implement Gaussian Naive Bayes algorithm for binary classification. Since dealing with
continuous dataset, the likelihood of the features is assumed to be Gaussian
   P(x_i|y) = (1/(sqrt(2*pi*sigma_y^2))*exp(-((x_i-mean_y)^2)/2*sigma_y ** 2)
x_i is the i-th feature of data point x and mean_y and sigma_y are the mean and variance
of x_i given label y. In binary classification case, y = 0 or 1

Procedure:
1. Compute P(y==0) and P(y==1)
2. Compute mean and variance of trainX for both y = 0 and y = 1
3. Compute P(x_i|y==0) and P(x_i|y==1)
4. Return train acc and test acc
"""

def GNBSolver(trainX, trainY, testX, testY):
    N_train = trainX.shape[0]
    N_test = testX.shape[0]
    M = trainX.shape[1]

    # TODO: Compute P(y==0) and P(y==1)
    py0 = (trainY == 0).astype(int).sum() / N_train
    py1 = (trainY == 1).astype(int).sum() / N_train

    # TODO: Compute mean and variance of trainX for both y = 0 and y = 1
    mean0 = trainX.iloc[trainY[trainY == 0].index.tolist(), :].mean()
    mean1 = trainX.iloc[trainY[trainY == 1].index.tolist(), :].mean()
    var0 = trainX.iloc[trainY[trainY == 0].index.tolist(), :].var()
    var1 = trainX.iloc[trainY[trainY == 1].index.tolist(), :].var()

    # TODO: Compute P(x_i|y==0) and P(x_i|y==1)
    trainX = np.array(trainX)
    testX = np.array(testX)
    mean0, mean1 = np.array(mean0), np.array(mean1)
    var0, var1 = np.array(var0), np.array(var1)
    Px_y0 = []
    Px_y1 = []
    Px_y0_test = []
    Px_y1_test = []
    for i in range(M):
        Px_y0.append((1 / np.sqrt(2 * np.pi * var0[i])) * np.exp(-np.power((trainX[:, i] - mean0[i]), 2) / 2 * var0[i]))
        Px_y1.append((1 / np.sqrt(2 * np.pi * var1[i])) * np.exp(-np.power((trainX[:, i] - mean1[i]), 2) / 2 * var1[i]))
        Px_y0_test.append(
            (1 / np.sqrt(2 * np.pi * var0[i])) * np.exp(-np.power((testX[:, i] - mean0[i]), 2) / 2 * var0[i]))
        Px_y1_test.append(
            (1 / np.sqrt(2 * np.pi * var1[i])) * np.exp(-np.power((testX[:, i] - mean1[i]), 2) / 2 * var1[i]))
    P0 = np.array(Px_y0).prod(axis=0) * py0
    P1 = np.array(Px_y1).prod(axis=0) * py1
    P0_test = np.array(Px_y0_test).prod(axis=0) * py0
    P1_test = np.array(Px_y1_test).prod(axis=0) * py1

    train_pred = np.where(P0 > P1, 0, 1)
    test_pred = np.where(P0_test > P1_test, 0, 1)

    # TODO: Return train acc and test acc
    trainY = np.array(trainY)
    testY = np.array(testY)
    train_acc = sum([trainY[i] == train_pred[i] for i in range(N_train)]) / N_train
    test_acc = sum([testY[i] == test_pred[i] for i in range(N_test)]) / N_test

    return train_acc, test_acc