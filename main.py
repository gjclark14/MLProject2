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
    return w[0] + w[1] * x


def four(t_features, feature):
    x = t_features[:, feature][:, None]
    X = np.hstack((np.ones_like(x), x))
    t = t_target
    XX = np.dot(X.T, X)
    invXX = np.linalg.inv(XX)
    Xt = np.dot(X.T, t)
    w = np.dot(invXX, Xt)
    return w


def six(t_target, t_features, feature):
    sum = 0
    for n in range(0, len(t_target)):
        sum += pow(t_target[n] - predicted_response(t_features[n][feature], w), 2)

    avg_loss = (1 / len(t_target)) * sum
    return avg_loss


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    features = []
    target = []

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
    # plt.scatter(t_features[:, Boston.RM], t_target)
    # plt.show()

    # NUMBER FOUR, this gives us w, a two item set, containing w0 and w1
    # x = t_features[:, Boston.RM][:, None]
    # X = np.hstack((np.ones_like(x), x))
    # t = t_target
    # XX = np.dot(X.T, X)
    # invXX = np.linalg.inv(XX)
    # Xt = np.dot(X.T, t)
    # w = np.dot(invXX, Xt)
    w = four(t_features, Boston.RM)
    print(w)

    # NUMBER 5, this graphs the line
    testx = np.linspace(4, 9, 2)[:, None]
    testX = np.hstack((np.ones_like(testx), testx))
    testt = np.dot(testX, w)
    # plt.plot(x, t, 'ro')
    # plt.plot(testx, testt, 'g')
    # plt.show()

    # NUMBER 6
    # Use w to find the predicted response for each value of the RM attribute in the training set,
    # then compute the average loss 𝓛 for the model.
    # predicted_response(x, w) = f(x_n; w0, w1) = w0 + w1x ?
    # if avg_loss = squared loss then
    # avg_loss = (1/n)sum((t[n] - predicted_response(feature(n)))^2)

    # sum = 0
    # for n in range(0, len(t_target)):
    #     print(six(t_target, t_features, Boston.RM))
    #     sum += pow(t_target[n] - predicted_response(t_features[0][Boston.RM], w), 2)


    avg_loss = six(t_target, t_features, Boston.RM)
    print(avg_lossRepeat experiments (4), (6), and (7) using all 13 input features as X. How do the)

    # 7
    # Repeat experiment (6) for the validation set. How do the training and validation MSE values compare?
    # What accounts for the difference?
    avg_loss = six(v_target, v_features, Boston.RM)
    print(avg_loss)

    # 8
    print("\n\n\n")
    w_vec = []
    total_loss = 0
    for feature in range(0, 13):
        # four gives w for a feature
        w = four(t_features, feature)
        w_vec.append(w)
        print(f'{feature}th four: w = <{float(w[0])}, {float(w[1])}>')

        # six gives the average loss using the training set
        avg_loss = six(t_target, t_features, feature)
        total_loss += avg_loss
        print(f'{feature}th six: average loss is {avg_loss} using the training set')

        # 7 gives us the average loss using the validation set
        avg_loss = six(v_target, v_features, feature)
        print(f'{feature}th seven: average loss is {avg_loss} using the validation set\n')

    print(f'average loss for training set is {total_loss/13}')




# See PyCharm help at https://www.jetbrains.com/help/pycharm/
