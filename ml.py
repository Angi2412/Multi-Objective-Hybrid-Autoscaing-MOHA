import logging
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


def linear_regression_model(X: np.array, y: np.array) -> None:
    """
    Linear Regression model with given data.
    :param X: data
    :param y: targets
    :return: None
    """
    # split data in to train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

    # Create linear regression object
    regression = LinearRegression()

    # Train the model using the training sets
    regression.fit(X_train, y_train)

    # Make predictions using the testing set
    y_pred = regression.predict(X_test)

    # get metrics
    print("Linear Regression")
    get_metrics(y_test, y_pred)


def get_metrics(test: np.array, pred: np.array) -> None:
    """
    Prints mean squared error and r2 score.
    :param test: test data
    :param pred: predicted data
    :return: None
    """
    # The mean squared error
    print('Mean squared error: %.2f' % mean_squared_error(test, pred))
    # The coefficient of determination: 1 is perfect prediction
    print('Coefficient of determination: %.2f' % r2_score(test, pred))


def svr_model(X: np.array, y: np.array) -> None:
    """
    Several SVR models with different kernel functions from given data.
    :param X: data
    :param y: targets
    :return: None
    """
    # split data in to train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

    # SVRs with different kernels
    svr_rbf = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
    svr_lin = SVR(kernel='linear', C=100, gamma='auto')
    svr_poly = SVR(kernel='poly', C=100, gamma='auto', degree=2, epsilon=.1,
                   coef0=1)
    svrs = [svr_rbf, svr_lin, svr_poly]
    for ix, svr in enumerate(svrs):
        # Train the model using the training sets
        svr.fit(X_train, y_train.ravel())
        # Make predictions using the testing set
        y_pred = svr.predict(X_test)
        # print scores
        print("SVR: " + str(ix))
        get_metrics(y_test, y_pred)


def neural_network_model(X: np.array, y: np.array) -> None:
    """
    MLPRegressor neural network with given data.
    :param X: data
    :param y: target
    :return: None
    """
    # split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
    # train neural network
    mlp = make_pipeline(StandardScaler(),
                        MLPRegressor(hidden_layer_sizes=(100, 100),
                                     tol=1e-2, max_iter=10000, random_state=0))
    mlp.fit(X, y.ravel())
    # make predictions using the testing set
    y_pred = mlp.predict(X_test)
    # print scores
    print("Neural Network")
    print(mlp.score(X_test, y_test))
    get_metrics(y_test, y_pred)


def get_data(date: str) -> (np.array, np.array):
    """
    Gets filtered data and converts it to a numpy array.
    :param date: name of filtered data
    :return: X, y
    """
    filtered_path = os.path.join(os.getcwd(), "data", "filtered")
    for root, dirs, files in os.walk(filtered_path):
        for file in files:
            if date in file:
                data = pd.read_csv(os.path.join(filtered_path, file))
                data = data.dropna()
                X = data[['cpu limit', 'memory limit', 'number of pods']].to_numpy()
                y = data[['average response time']].to_numpy()
                return X, y
            else:
                logging.warning(f"No filtered file with name {date} found.")


if __name__ == '__main__':
    X_data, y_data = get_data("20210217-223253")
    linear_regression_model(X_data, y_data)
    svr_model(X_data, y_data)
    neural_network_model(X_data, y_data)
