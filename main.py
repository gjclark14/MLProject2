# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np
import matplotlib.pyplot as plt


class Boston(enumerate):
    CRIM = 0
    ZN = 1
    INDUS = 2
    CHAS = 3
    NOX = 4
    RM = 5
    AGE = 6
    DIS = 7
    RAD = 8
    TAX = 9
    PTRATIO = 10
    B = 11
    LSTAT = 12
    MEDV = 13


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    features = []
    target = []

    with np.load('boston.npz') as data:
        features = data['features']
        target = data['target'][:, None]

    # our validation set
    v_features = features[:102]
    v_target = target[:102]

    # our training set
    t_features = features[102:]
    t_target = target[102:]

    # TODO: Add labels to the scatterplot
    # NUMBER 3
    #plt.scatter(t_features[:, Boston.RM], t_target)
    #plt.show()

    # NUMBER FOUR
    x = t_features[:, Boston.RM][:, None]
    X = np.hstack((np.ones_like(x), x))
    t = t_target
    XX = np.dot(X.T, X)
    invXX = np.linalg.inv(XX)
    Xt = np.dot(X.T, t)
    w = np.dot(invXX, Xt)
    print(w)

    # NUMBER 5
    testx = np.linspace(4, 9, 2)[:, None]
    testX = np.hstack((np.ones_like(testx), testx))
    testt = np.dot(testX, w)
    plt.plot(x, t, 'ro')
    plt.plot(testx, testt, 'g')
    plt.show()



# See PyCharm help at https://www.jetbrains.com/help/pycharm/
