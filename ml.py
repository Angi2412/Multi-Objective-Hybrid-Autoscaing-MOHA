import logging
import os
from time import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv, set_key
from joblib import dump, load, numpy_pickle
from skcriteria import Data, MIN, MAX
from skcriteria.madm.closeness import TOPSIS
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR


def linear_least_squares_model(target: str, save: bool) -> None:
    """
    Linear Regression model with given data.
    :param save: if should save
    :param target: target name
    :return: None
    """
    X_train, X_test, y_train, y_test = get_processed_data(target)

    # Create linear regression object
    regression = LinearRegression()

    tic = time()
    # Train the model using the training sets
    regression.fit(X_train, y_train)
    gsh_time = time() - tic
    print("Linear Least Squares Regression")
    print(f"Training time: {gsh_time}")
    # Make predictions using the testing set
    y_pred = regression.predict(X_test)

    # get metrics
    print("Metrics:")
    get_metrics(y_test, y_pred)
    # save model
    if save:
        save_model(regression, target)


def linear_bayesian_model(target: str, save: bool, search: bool) -> None:
    """
    Linear Regression model with given data.
    :param save: if should save
    :param target: target name
    :param search: grid search
    :return: None
    """
    X_train, X_test, y_train, y_test = get_processed_data(target)

    if search:
        params = {"lambda_1": np.logspace(-2, 10, 13, base=2), "lambda_2": np.logspace(-2, 10, 13, base=2)}
        tic = time()
        search = GridSearchCV(estimator=BayesianRidge(alpha_1=0.01, alpha_2=0.01), param_grid=params, verbose=1)
        search.fit(X_train, y_train.ravel())
        gsh_time = time() - tic
        print(f"Training time: {gsh_time}")
        print(f"Best params: {search.best_params_}")
    # Train the model using the training sets
    else:
        # Create linear regression object
        regression = BayesianRidge(alpha_1=0.01, alpha_2=0.01, lambda_1=1024, lambda_2=1.0)
        tic = time()
        regression.fit(X_train, y_train.ravel())
        gsh_time = time() - tic
        print("Linear Bayesian Regression")
        print(f"Training time: {gsh_time}")
        # Make predictions using the testing set
        y_pred = regression.predict(X_test)
        # get metrics
        print("Metrics:")
        get_metrics(y_test, y_pred)
        # save model
        if save:
            save_model(regression, target)


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


def svr_model(target: str, save: bool, search: bool) -> None:
    """
    Several SVR models with different kernel functions from given data.
    :param save: if should save
    :param target: target name
    :param search: search for hyper parameter
    :return: None
    """
    # split data in to train and test sets
    X_train, X_test, y_train, y_test = get_processed_data(target)
    if search:
        # SVRs with different kernels
        params = {"C": np.logspace(-2, 10, 13, base=2), "gamma": np.logspace(1, 3, 13, base=2)}
        tic = time()
        search = GridSearchCV(estimator=SVR(kernel="rbf", cache_size=8000, epsilon=0.1), param_grid=params, verbose=1)
        search.fit(X_train, y_train.ravel())
        gsh_time = time() - tic
        print(f"Training time: {gsh_time}")
        print(f"Best params: {search.best_params_}")
    else:
        svr = SVR(kernel="rbf", C=4.0, gamma=8.0, cache_size=8000)
        tic = time()
        svr.fit(X_train, y_train.ravel())
        gsh_time = time() - tic
        print("Support Vector Regression")
        print(f"Training time: {gsh_time}")
        # Make predictions using the testing set
        y_pred = svr.predict(X_test)
        # print scores
        print("Metrics:")
        get_metrics(y_test, y_pred)
        if save:
            save_model(search, target)


def neural_network_model(target: str, search: bool, save: bool) -> None:
    """
    MLPRegressor neural network with given data.
    :param search: use search
    :param save: should save
    :param target: target name
    :return: None
    """
    # split data
    X_train, X_test, y_train, y_test = get_processed_data(target)
    # train neural network
    mlp = None
    if search:
        # SVRs with different kernels
        params = {"solver": ["lbfgs", "adam", "sgd"], "activation": ["identity", "logistic", "tanh", "relu"],
                  "alpha": np.logspace(-2, 10, 13, base=2), "tol": np.logspace(1, 3, 13, base=2)}
        tic = time()
        search = GridSearchCV(estimator=MLPRegressor(learning_rate="adaptive", max_iter=1000), param_grid=params,
                              verbose=1)
        search.fit(X_train, y_train.ravel())
        gsh_time = time() - tic
        print(f"Training time: {gsh_time}")
        print(f"Best params: {search.best_params_}")
    else:
        mlp = MLPRegressor(activation="relu", alpha=0.25, solver="adam", tol=2.244924096618746)
        # make predictions using the testing set
        tic = time()
        mlp.fit(X_train, y_train.ravel())
        gsh_time = time() - tic
        print("Neural Network")
        print(f"Training time: {gsh_time}")
        y_pred = mlp.predict(X_test)
        # print scores
        print("Metrics:")
        print(mlp.score(X_test, y_test))
        get_metrics(y_test, y_pred)
    # save model
    if save:
        save_model(mlp, target)


def get_data(date: str, target: str, combined: bool) -> (np.array, np.array):
    """
    Gets filtered data and converts it to a numpy array.
    :param combined: combined or filtered
    :param target: name of target
    :param date: name of filtered data
    :return: X, y
    """
    # init path
    path = None
    if combined:
        path = os.path.join(os.getcwd(), "data", "combined")
    else:
        path = os.path.join(os.getcwd(), "data", "filtered")
    # get data
    for root, dirs, files in os.walk(path):
        for file in files:
            if date in file and "mean" not in file:
                data = pd.read_csv(os.path.join(path, file), delimiter=",")
                data = data.reset_index()
                X = data[['cpu limit', 'memory limit', 'number of pods']].to_numpy()
                y = data[[target]].to_numpy()
                logging.info(f"X: {X.shape} - y: {y.shape}")
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
    step = int(os.getenv("STEP"))
    models = get_models()
    predict_window = np.empty(window * 2, dtype=[('cpu', np.int32), ('memory', np.int32), ('pods', np.int32)])
    predictions = np.empty((len(models), window * 2))
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
        X, y = get_data(date, t, True)
        if kind == "neural":
            neural_network_model(t, False, True)
        elif kind == "linear":
            linear_least_squares_model(t, True)
            linear_bayesian_model(t, True, False)
        elif kind == "svr":
            svr_model(t, True, False)
        else:
            logging.warning("There is no model type: " + kind)
            return
    set_key(os.getenv(os.getcwd(), ".env"), "LAST_TRAINED_DATA", date)
    logging.info("All models are trained.")


def get_processed_data(target: str) -> (np.array, np.array, np.array, np.array):
    load_dotenv()
    X, y = get_data(os.getenv("LAST_DATA"), target, True)
    # scale dataset
    scaling = MinMaxScaler()
    X = scaling.fit_transform(X)
    y = scaling.fit_transform(y)
    # split data in to train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    load_dotenv()
    train_for_all_targets(os.getenv("LAST_DATA"), "svr")
