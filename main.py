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


# predicted_response(x) = w0 + w1x ?
def predicted_response(x, w):
    """
    :param x: feature vector
    :param w: weight vector
    :return: the dot product of x and w
    """
    return w[0] + w[1] * x


def four(t_features, feature):
    """
    Calculates the weight vector for a single feature.

    :param t_features: feature vector
    :param feature: feature to calculate for
    :return: weight vector for feature
    """
    x = t_features[:, feature][:, None]
    X = np.hstack((np.ones_like(x), x))
    t = t_target
    XX = np.dot(X.T, X)
    invXX = np.linalg.inv(XX)
    Xt = np.dot(X.T, t)
    return np.dot(invXX, Xt)


def six(t_target, t_features, feature):
    """
    Uses the Mean Squared Error aka Average Squared Loss algorithm:
    avg_loss = (1/n)sum((t[n] - predicted_response(feature(n)))^2)

    :param t_target: target vector
    :param t_features: feature vector
    :param feature: feature to calculate loss for
    :return: Mean Squared Error
    """
    sum = 0
    for n in range(0, len(t_target)):
        sum += pow(t_target[n] - predicted_response(t_features[n][feature], w), 2)

    avg_loss = (1 / len(t_target)) * sum
    return avg_loss

def get_loss(t, X, W):
    """
    Similar to six, calculates the loss this time using np and matrix operations:\n
    L = (1/N) (t-Xw)T dot (t-Xw)

    :param t: target vector
    :param X: feature matrix
    :param W: weight vector
    :return: loss
    """
    t_minus_Xw = np.subtract(t, np.dot(X, W))
    t_minus_Xw_transpose = np.transpose(t_minus_Xw)
    return np.dot(t_minus_Xw_transpose, t_minus_Xw) / len(t)

def get_w(x, t, feature = 0):
    """
    Returns the weight vector

    :param array x: feature vector
    :param array t: target vector
    :param int feature: optional feature to split on
    :return: weight vector
    """
    X = np.hstack((np.ones_like(x[:, feature][:, None]), x))
    XX = np.dot(X.T, X)
    invXX = np.linalg.inv(XX)
    Xt = np.dot(X.T, t)
    return np.dot(invXX, Xt)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    features = []
    target = []

    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

    with np.load('boston.npz') as data:
        features = data['features']
        target = data['target'][:, None]

    # our validation set, first 102 items
    v_features = features[:102]
    v_target = target[:102]

    # our training set, next 404 items
    t_features = features[102:]
    t_target = target[102:]

    # TODO: Add labels to the scatterplot
    # NUMBER 3
    plt.scatter(t_features[:, Boston.RM], t_target)
    plt.show()

    # NUMBER FOUR, this gives us w, a two item set, containing w0 and w1
    w = four(t_features, Boston.RM)
    print(f'four: w = <{w[0]},{w[1]}>')

    # NUMBER 5, this graphs the line
    testx = np.linspace(4, 9, 2)[:, None]
    testX = np.hstack((np.ones_like(testx), testx))
    testt = np.dot(testX, w)
    x = t_features[:, Boston.RM][:, None]
    t = t_target
    plt.plot(x, t, 'ro')
    plt.plot(testx, testt, 'g')
    plt.show()

    # NUMBER 6
    avg_loss = six(t_target, t_features, Boston.RM)
    print(f'six: {avg_loss}')

    # 7
    # Repeat experiment (6) for the validation set. How do the training and validation MSE values compare?
    # What accounts for the difference?
    avg_loss = six(v_target, v_features, Boston.RM)
    print(f'seven: {avg_loss}')

    # 8
    # the repetition of number four
    print('eight: (the repetition of four) w = ', end='')
    print(get_w(t_features, t_target))

    # the repetition of number 6
    X = np.hstack((np.ones_like(t_features[:, 0][:, None]), t_features))
    print(f'loss of the entire training set: {get_loss(t_target, X, get_w(t_features, t_target))}')

    # the repetition of number 7
    X = np.hstack((np.ones_like(v_features[:, 0][:, None]), v_features))
    print(f'loss of the entire validation set: {get_loss(v_target, X, get_w(v_features, v_target))}')

    # 9
    X = np.hstack((np.ones_like(t_features[:, 0][:, None]), t_features))
    w = get_w(t_features, t_target)
    for i in range(1, len(w)):
        print(f'${float(w[i])*1000:.2f}')



# See PyCharm help at https://www.jetbrains.com/help/pycharm/
