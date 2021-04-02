import datetime as dt
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

import benchmark
import k8s_tools


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
    print("Target: " + target)
    print(f"Training time: {gsh_time}")
    # Make predictions using the testing set
    tic = time()
    y_pred = regression.predict(X_test)
    pred_time = time() - tic
    print(f"Prediction time: {pred_time}")

    # get metrics
    print("Metrics:")
    get_metrics(y_test, y_pred)
    # save model
    if save:
        save_model(regression, target, "linear_lsq")


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
        params = {"lambda_1": np.logspace(-2, 10, 13, base=2), "lambda_2": np.logspace(-2, 10, 13, base=2),
                  "alpha_1": np.linspace(0.1, 1, 10), "alpha_2": np.linspace(0.1, 1, 10)}
        tic = time()
        search = GridSearchCV(estimator=BayesianRidge(), param_grid=params, verbose=1)
        search.fit(X_train, y_train.ravel())
        gsh_time = time() - tic
        print(f"Training time: {gsh_time}")
        print(f"Best params: {search.best_params_}")
    # Train the model using the training sets
    else:
        # Create linear regression object
        regression = BayesianRidge(alpha_1=1.0, alpha_2=0.1, lambda_1=1024, lambda_2=4.0)
        tic = time()
        regression.fit(X_train, y_train.ravel())
        gsh_time = time() - tic
        print("Linear Bayesian Regression")
        print("Target: " + target)
        print(f"Training time: {gsh_time}")
        # Make predictions using the testing set
        tic = time()
        y_pred = regression.predict(X_test)
        pred_time = time() - tic
        print(f"Prediction time: {pred_time}")
        # get metrics
        print("Metrics:")
        get_metrics(y_test, y_pred)
        # save model
        if save:
            save_model(regression, target, "linear_b")


def get_metrics(test: np.array, pred: np.array) -> None:
    """
    Prints mean squared error and r2 score.
    :param test: test data
    :param pred: predicted data
    :return: None
    """
    # The mean squared error
    print('Mean squared error: %.2f' % mean_squared_error(test, pred))
    # RMSE
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
        params = {"C": np.logspace(-2, 10, 13, base=2), "gamma": np.logspace(1, 3, 13, base=2),
                  "epsilon": np.linspace(0.1, 2.0, 10)}
        tic = time()
        search = GridSearchCV(estimator=SVR(kernel="rbf", cache_size=10000), param_grid=params, verbose=1)
        search.fit(X_train, y_train.ravel())
        gsh_time = time() - tic
        print(f"Training time: {gsh_time}")
        print(f"Best params: {search.best_params_}")
    else:
        svr = SVR(kernel="rbf", C=2.0, epsilon=0.1, gamma=2.0, cache_size=10000)
        tic = time()
        svr.fit(X_train, y_train.ravel())
        gsh_time = time() - tic
        print("Support Vector Regression")
        print("Target: " + target)
        print(f"Training time: {gsh_time}")
        # Make predictions using the testing set
        tic = time()
        y_pred = svr.predict(X_test)
        pred_time = time() - tic
        print(f"Prediction time: {pred_time}")
        # print scores
        print("Metrics:")
        get_metrics(y_test, y_pred)
        if save:
            save_model(svr, target, "svr")


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
        params = {"alpha": np.logspace(-2, 10, 13, base=2), "tol": np.logspace(1, 3, 13, base=2)}
        tic = time()
        search = GridSearchCV(
            estimator=MLPRegressor(solver="adam", activation="relu", learning_rate="adaptive", max_iter=100000),
            param_grid=params,
            verbose=1)
        search.fit(X_train, y_train.ravel())
        gsh_time = time() - tic
        print(f"Training time: {gsh_time}")
        print(f"Best params: {search.best_params_}")
    else:
        mlp = MLPRegressor(activation="relu", alpha=0.25, solver="adam", tol=2.8284271247461903)
        # make predictions using the testing set
        tic = time()
        mlp.fit(X_train, y_train.ravel())
        gsh_time = time() - tic
        print("Neural Network")
        print("Target: " + target)
        print(f"Training time: {gsh_time}")
        tic = time()
        y_pred = mlp.predict(X_test)
        pred_time = time() - tic
        print(f"Prediction time: {pred_time}")
        # print scores
        print("Metrics:")
        print(mlp.score(X_test, y_test))
        get_metrics(y_test, y_pred)
    # save model
    if save:
        save_model(mlp, target, "neural_network")


def get_data(date: str, target: str, combined: bool) -> (np.array, np.array):
    """
    Gets filtered data and converts it to a numpy array.
    :param combined: combined or filtered
    :param target: name of target
    :param date: name of filtered data
    :return: X, y
    """
    # init path
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
                X = data[['cpu limit', 'memory limit', 'number of pods', 'average rps']].to_numpy()
                y = data[[target]].to_numpy()
                logging.info(f"X: {X.shape} - y: {y.shape}")
                return X, y
    logging.warning(f"No filtered file with name {date} found.")


def save_model(model, name: str, alg: str) -> None:
    """
    Saves a model under a given name.
    :param alg: used algorithm
    :param model: model
    :param name: model name
    :return: None
    """
    save_path = os.path.join(os.getcwd(), "data", "models", alg)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    dump(model, os.path.join(save_path, f"{name}.joblib"))


def load_model(name: str, alg: str) -> numpy_pickle:
    """
    Loads a given model.
    :param alg: algorithm used
    :param name: model name
    :return: model
    """
    save_path = os.path.join(os.getcwd(), "data", "models", alg, f"{name}.joblib")
    return load(save_path)


def get_best_parameters(cpu_limit: int, memory_limit: int, number_of_pods: int, rps: float, window: int, alg: str):
    """
    Chooses the best values for the parameters in a given window for a given status.
    :param alg: algorithm to use
    :param cpu_limit: current cpu limit
    :param memory_limit: current memory limit
    :param number_of_pods: current number of pods
    :param rps: requests per second
    :param window: size of window
    :return: pes parameters
    """
    # init
    step = int(os.getenv("STEP"))
    models = get_models(alg)
    # get all possibilities in window
    parameter_variations = benchmark.parameter_variation("webui", cpu_limit - window * step, cpu_limit + window * step,
                                                         memory_limit - window * step, memory_limit + window * step,
                                                         number_of_pods - window, number_of_pods + window,
                                                         step, False, False, False, [int(rps)])
    # flatten possibilities
    predict_window_list = list(np.concatenate(parameter_variations).flat)
    # init arrays
    possibilities = len(predict_window_list)
    predictions = np.empty((len(models), possibilities))
    prediction_array = np.zeros([possibilities, len(models)], dtype=np.float64)
    # validate parameter variations
    for i, entry in enumerate(predict_window_list):
        predict_window_list[i] = validate_parameter(entry, rps)
    predict_window = np.array(predict_window_list, dtype=np.float64)
    # scale data
    scaler = MinMaxScaler()
    predict_window_scaled = scaler.fit_transform(predict_window)
    # get predictions for each model
    for i, model in enumerate(models):
        # predict
        predictions[i] = model.predict(predict_window_scaled)
    # load target scaler
    y_scalers = list()
    for t in ["average response time", "cpu usage", "memory usage"]:
        y_scalers.append(load(os.path.join(os.getcwd(), "data", "models", "data", f"y_scaler_{t}.gz")))
    # format into array
    for i in range(0, possibilities):
        for j in range(0, len(models)):
            prediction_array[i, j] = y_scalers[j].inverse_transform(predictions[j, i].reshape(1, -1))
    # concatenate targets and parameters
    prediction_array = np.concatenate((prediction_array, predict_window), axis=1)
    # delete rps parameter
    prediction_array = np.delete(prediction_array, -1, 1)
    # get index of best outcome
    best_outcome_index = choose_best(prediction_array.tolist())
    # get parameters of best outcome
    best_parameters = predict_window_list[best_outcome_index]
    print(
        f"Best Targets: {prediction_array[best_outcome_index, 0]}ms - {prediction_array[best_outcome_index, 1]}% - {prediction_array[best_outcome_index, 2]}%")
    return best_parameters


def validate_parameter(limits: tuple, rps: float) -> tuple:
    """
    Validates if the given limits are below the requested resources.
    :return: validated resources
    """
    cpu, memory, pods, tmp = limits
    request = k8s_tools.get_resource_requests()[os.getenv("SCALE_POD")]
    cpu_request = int(str(request["cpu"]).rstrip("m"))
    memory_request = int(str(request["memory"]).rstrip("Mi"))
    if cpu <= cpu_request:
        cpu = cpu_request
    if memory <= memory_request:
        memory = memory_request
    if pods <= 1:
        pods = 1
    return cpu, memory, pods, rps


def choose_best(mtx: np.array) -> int:
    """
    Chooses the best alternative from given alternatives with multiple criteria.
    :param mtx: alternatives
    :return: index of best alternative
    """
    # min average response time, max cpu usage, max memory usage, min cpu limit, min memory limit, min number of pods
    criteria = [MIN, MAX, MAX, MIN, MIN, MIN]
    weights = [0.3, 0.125, 0.125, 0.125, 0.125, 0.20]
    # create DecisionMaker
    dm = TOPSIS()
    # create data object
    data = Data(mtx=mtx, criteria=criteria, weights=weights,
                cnames=["average response time", "cpu usage", "memory usage", "cpu limit", "memory limit",
                        "number of pods"])
    # make decision
    dec = dm.decide(data)
    # print(f"index: {dec.best_alternative_}")
    return dec.best_alternative_


def get_models(alg: str) -> list:
    """
    Imports all models.
    :return: list of models
    """
    targets = ["average response time", "cpu usage", "memory usage"]
    models = list()
    for t in targets:
        model = os.path.join(os.getcwd(), "data", "models", alg, f"{t}.joblib")
        if os.path.exists(model):
            models.append(load(model))
        else:
            logging.error(f"No model found with name {t}")
    print(models)
    return models


def train_for_all_targets(kind: str) -> None:
    """
    Trains a given model for all targets.
    :param kind: model type
    :return: None
    """
    date = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    targets = ["average response time", "cpu usage", "memory usage"]
    for t in targets:
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
    d_path = os.path.join(os.getcwd(), "data", "models", "data", target)
    d = [None, None, None, None]
    for i in range(0, 4):
        d[i] = np.load(os.path.join(d_path, f"{i}.npy"))
    return d[0], d[1], d[2], d[3]


def processes_data() -> None:
    load_dotenv()
    for t in ["average response time", "cpu usage", "memory usage"]:
        X, y = get_data(os.getenv("LAST_DATA"), t, True)
        # scale dataset
        x_scaling = MinMaxScaler()
        y_scaling = MinMaxScaler()
        X = x_scaling.fit_transform(X)
        y = y_scaling.fit_transform(y)
        dump(y_scaling, os.path.join(os.getcwd(), "data", "models", "data", f"y_scaler_{t}.gz"))
        # split data in to train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
        logging.info(f"Training size: {X_train.shape}")
        logging.info(f"Test size: {X_test.shape}")
        d_path = os.path.join(os.getcwd(), "data", "models", "data", t)
        for i, d in enumerate([X_train, X_test, y_train, y_test]):
            np.save(os.path.join(d_path, str(i)), d)


if __name__ == '__main__':
    processes_data()