import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import dump, load, numpy_pickle
from skcriteria import Data, MIN, MAX
from skcriteria.madm.closeness import TOPSIS
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


def linear_regression_model(X: np.array, y: np.array, name: str) -> None:
    """
    Linear Regression model with given data.
    :param name: name
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
    save_model(regression, name)


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


def svr_model(X: np.array, y: np.array, name: str) -> None:
    """
    Several SVR models with different kernel functions from given data.
    :param name: name
    :param X: data
    :param y: targets
    :return: None
    """
    # split data in to train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

    # SVRs with different kernels
    svr_rbf = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
    svr_lin = SVR(kernel='linear', C=100, gamma='auto', verbose=True)
    svr_poly = SVR(kernel='poly', C=100, gamma='auto', degree=2, epsilon=.1,
                   coef0=1, verbose=True)
    svrs = [svr_rbf, svr_lin, svr_poly]
    for ix, svr in enumerate(svrs):
        # Train the model using the training sets
        svr.fit(X_train, y_train.ravel())
        # Make predictions using the testing set
        y_pred = svr.predict(X_test)
        # print scores
        print("SVR: " + str(ix))
        get_metrics(y_test, y_pred)
    save_model(svr, name)


def neural_network_model(X: np.array, y: np.array, name: str) -> None:
    """
    MLPRegressor neural network with given data.
    :param name: name of model
    :param X: data
    :param y: target
    :return: None
    """
    # split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
    # train neural network
    mlp = make_pipeline(StandardScaler(),
                        MLPRegressor(hidden_layer_sizes=(1000, 1000),
                                     tol=1e-2, max_iter=10000, random_state=0))
    mlp.fit(X, y.ravel())
    # make predictions using the testing set
    y_pred = mlp.predict(X_test)
    # print scores
    print("Neural Network")
    print(mlp.score(X_test, y_test))
    get_metrics(y_test, y_pred)
    # save model
    save_model(mlp, name)


def get_data(date: str, target: str) -> (np.array, np.array):
    """
    Gets filtered data and converts it to a numpy array.
    :param target: name of target
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
                y = data[[target]].to_numpy()
                return X, y
    logging.warning(f"No filtered file with name {date} found.")


def save_model(model, name: str) -> None:
    """
    Saves a model under a given name.
    :param model: model
    :param name: model name
    :return: None
    """
    save_path = os.path.join(os.getcwd(), "data", "models", f"{name}.joblib")
    dump(model, save_path)


def load_model(name: str) -> numpy_pickle:
    """
    Loads a given model.
    :param name: model name
    :return: model
    """
    save_path = os.path.join(os.getcwd(), "data", "models", f"{name}.joblib")
    return load(save_path)


def get_best_parameters(cpu_limit: int, memory_limit: int, number_of_pods: int, window: int):
    """
    Chooses the best values for the parameters in a given window for a given status.
    :param cpu_limit: current cpu limit
    :param memory_limit: current memory limit
    :param number_of_pods: current number of pods
    :param window: size of window
    :return: pes parameters
    """
    # init arrays
    step = os.getenv("STEP")
    models = get_models()
    predict_window = np.empty(window * 2, dtype=[('cpu', np.int32), ('memory', np.int32), ('pods', np.int32)])
    predictions = np.empty((len(models), window*2))
    prediction_array = np.zeros(window * 2, dtype=[('cpu', np.int32), ('memory', np.int32), ('art', np.int32)])
    # get all possibilities in window
    for i in range(0, 2 * window):
        j = (i % window) + 1
        if i < window:
            predict_window[i] = ((cpu_limit - (j * step)), (memory_limit - (j * step)), (number_of_pods - j))
        else:
            predict_window[i] = ((cpu_limit + (j * step)), (memory_limit + (j * step)), (number_of_pods + j))
    # get predictions
    for i, model in enumerate(models):
        print(model.predict(predict_window.tolist()))
        predictions[i] = model.predict(predict_window.tolist())
    # format into array
    for i in range(0, 2 * window):
        prediction_array[i] = (predictions[0][i], predictions[1][i], predictions[2][i])
    # get index of best outcome
    best_outcome_index = choose_best(prediction_array.tolist())
    # get parameters of best outcome
    best_parameters = predict_window[best_outcome_index]
    return best_parameters


def choose_best(mtx: np.array) -> int:
    """
    Chooses the best alternative from given alternatives with multiple criteria.
    :param mtx: alternatives
    :return: index of best alternative
    """
    # min average response time, max cpu usage, max memory usage
    criteria = [MIN, MAX, MAX]
    # create DecisionMaker
    dm = TOPSIS()
    # create data object
    data = Data(mtx=mtx, criteria=criteria, cnames=["average response time", "cpu usage", "memory usage"])
    # make decision
    dec = dm.decide(data)
    # show result
    print(dec)
    data.plot("box")
    plt.show()
    return dec.best_alternative_


def get_models() -> list:
    """
    Imports all models.
    :return: list of models
    """
    targets = ["average response time", "cpu usage", "memory usage"]
    models_path = os.path.join(os.getcwd(), "data", "models")
    models = list()
    for t in targets:
        model = os.path.join(models_path, f"{t}.joblib")
        models.append(load(model))
    return models


def train_for_all_targets(date: str, kind: str) -> None:
    """
    Trains a given model for all targets.
    :param date: name of data
    :param kind: model type
    :return: None
    """
    targets = ["average response time", "cpu usage", "memory usage"]
    for t in targets:
        X, y = get_data(date, t)
        if kind == "neural":
            neural_network_model(X, y, t)
        elif kind == "linear":
            linear_regression_model(X, y, t)
        elif kind == "svr":
            svr_model(X, y, t)
        else:
            logging.warning("There is no model type: " + kind)
            return
    logging.info("All models are trained.")