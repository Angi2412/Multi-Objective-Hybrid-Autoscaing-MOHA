import datetime as dt
import logging
import math
import os
from time import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from dotenv import load_dotenv, set_key
from joblib import dump, load, numpy_pickle
from skcriteria import Data, MIN, MAX
from skcriteria.madm.closeness import TOPSIS
from skcriteria.madm.simple import WeightedSum
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
        plot_prediction(y_test, y_pred, "linear", target)
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
        params = {"epsilon": np.arange(0.1, 1, 0.01)}
        tic = time()
        search = GridSearchCV(estimator=SVR(kernel="rbf", C=2.0, gamma=2.0, cache_size=12000), param_grid=params,
                              verbose=1)
        search.fit(X_train, y_train.ravel())
        gsh_time = time() - tic
        print(f"Training time: {gsh_time}")
        print("The best parameters are %s with a score of %0.2f"
              % (search.best_params_, search.best_score_))
    else:
        svr = SVR(kernel="rbf", C=2.0, gamma=2.0, cache_size=12000)
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
        plot_prediction(y_test, y_pred, "svr", target)
        if save:
            save_model(svr, target, "svr")


def plot_prediction(y_train, y_pred, alg, target):
    # regplot
    ax = sns.regplot(x=y_train, y=y_pred, scatter=True, fit_reg=True)
    ax.set_xlabel("Expected values")
    ax.set_ylabel("Predicted values")
    ax.figure.savefig(os.path.join(os.getcwd(), "data", "plots", f"{alg}_{target}.png"))
    plt.show()


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
        params = {"alpha": np.arange(0.1, 2, 0.01)}
        tic = time()
        search = GridSearchCV(
            estimator=MLPRegressor(solver="adam", tol=2.8284271247461903, activation="tanh", learning_rate="adaptive",
                                   max_iter=100000),
            param_grid=params,
            verbose=1)
        search.fit(X_train, y_train.ravel())
        gsh_time = time() - tic
        print(f"Training time: {gsh_time}")
        print(f"Best params: {search.best_params_}")
    else:
        mlp = MLPRegressor(solver="adam", alpha=0.49, tol=2.8284271247461903, activation="tanh",
                           learning_rate="adaptive",
                           max_iter=100000)
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
        get_metrics(y_test, y_pred)
        plot_prediction(y_test, y_pred, "neural", target)
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


def get_best_parameters_hpa(cpu_limit: int, memory_limit: int, number_of_pods: int, rps: float, alg: str,
                            response_time: float, cpu_usage: float, memory_usage: float, extrap: bool):
    # init
    models = get_models(alg)
    cpu_limits = list()
    memory_limits = list()
    pod_limits = list()
    # thresholds
    thresholds = [int(os.getenv("TARGET_RESPONSE")), (int(os.getenv("MAX_USAGE")) + int(os.getenv("MIN_USAGE"))) / 2,
                  (int(os.getenv("MAX_USAGE")) + int(os.getenv("MIN_USAGE"))) / 2]
    status = [response_time, cpu_usage, memory_usage]
    # if nan
    if math.isnan(cpu_limit):
        cpu_limit = 400
    if math.isnan(memory_limit):
        memory_limit = 400
    if math.isnan(number_of_pods):
        number_of_pods = 1
    # calculate possible limits
    for t, s in zip(thresholds, status):
        cpu_limits.append(math.ceil(cpu_limit * (s / t)))
        memory_limits.append(math.ceil(memory_limit * (s / t)))
        pod_limits.append(math.ceil(number_of_pods * (s / t)))
    cpu_limits.pop(-1)
    memory_limits.pop(1)
    if os.getenv("HPA") == "True":
        cpu_limits = [cpu_limit]
        memory_limits = [memory_limit]
    # make parameter variation
    parameter_variations = benchmark.parameter_variation_array(cpu_limits, memory_limits, pod_limits, rps)
    # flatten possibilities
    predict_window_list = list(np.concatenate(parameter_variations).flat)
    # validate parameter variations
    logging.info(predict_window_list)
    predict_window_list = validate_parameter(predict_window_list, rps)
    # init arrays
    possibilities = len(predict_window_list)
    predictions = np.empty((len(models), possibilities))
    prediction_array = np.zeros([possibilities, len(models)], dtype=np.float64)
    predict_window = np.array(predict_window_list, dtype=np.float64)
    if extrap:
        prediction_array = predict_extrap(predict_window_list)
    else:
        # load parameter scaler
        x_scaler = load(os.path.join(os.getcwd(), "data", "models", "data", f"x_scaler_average response time.gz"))
        # scale data
        if predict_window.size != 0:
            if predict_window.size == 1:
                predict_window = predict_window.reshape(1, -1)
            predict_window_scaled = x_scaler.transform(predict_window)
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
        else:
            return None
    logging.info(prediction_array)
    # concatenate targets and parameters
    prediction_array = np.concatenate((prediction_array, predict_window), axis=1)
    # validate targets
    if len(prediction_array) > 1:
        # delete rps parameter
        prediction_array = np.delete(prediction_array, -1, 1)
        # get index of best outcome
        # if horizontal scaling only delete cpu and memory limit
        best_outcome_index = choose_best(prediction_array.tolist(), False)
        # get parameters of best outcome
        best_parameters = prediction_array[best_outcome_index]
        logging.info(
            f"Best Parameter: {prediction_array[best_outcome_index, 3]}m - {prediction_array[best_outcome_index, 4]}Mi - {prediction_array[best_outcome_index, 5]}")
        logging.info(
            f"Best Targets: {prediction_array[best_outcome_index, 0]}ms - {prediction_array[best_outcome_index, 1]}% - {prediction_array[best_outcome_index, 2]}%")
        return best_parameters
    elif len(prediction_array) == 1:
        best_outcome_index = 0
        best_parameters = prediction_array[best_outcome_index]
        logging.info(
            f"Best Parameter: {prediction_array[best_outcome_index, 3]}m - {prediction_array[best_outcome_index, 4]}Mi - {prediction_array[best_outcome_index, 5]}")
        logging.info(
            f"Best Targets: {prediction_array[best_outcome_index, 0]}ms - {prediction_array[best_outcome_index, 1]}% - {prediction_array[best_outcome_index, 2]}%")
        return best_parameters
    else:
        return None


def predict_extrap(parameters: list) -> np.array:
    predicted = list()
    for c, m, p, rps in parameters:
        response_time = 25714.32539236546 - 2502.3032054616065 * math.log(c, 2) - 222.9076387765515 * math.log(m,
                                                                                                               2) - 877.3839295358297 * math.log(
            p, 2) + 82.33195028589898 * math.pow(math.log(rps, 2), 2)
        cpu_usage = 207.24043369071322 - 18.358570566154665 * math.log(c, 2) - 15.099670589384738 * math.log(p,
                                                                                                             2) + 2.224919774965972 * math.pow(
            math.log(rps, 2), 3 / 2)
        memory_usage = 453.9128240678697 - 44.393604901442515 * math.log(m, 2) + 5.716589705683245 * math.log(p,
                                                                                                              1 / 2) + 0.44049840226376347 * math.pow(
            math.log(rps, 2), 2)
        predicted.append((response_time, cpu_usage, memory_usage))
    logging.info(predicted)
    return np.array(predicted, dtype=np.float64)


def get_best_parameters_window(cpu_limit: int, memory_limit: int, number_of_pods: int, rps: float, window: int,
                               alg: str, hpa: bool, response_time: float, cpu_usage: float,
                               memory_usage: float) -> np.array:
    """
    Chooses the best values for the parameters in a given window for a given status.
    :param memory_usage: current memory usage
    :param cpu_usage: current cpu usage
    :param response_time: current average response time
    :param hpa: only horizontal scaling
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
    current_status = np.array((cpu_limit, memory_limit, number_of_pods, rps), dtype=np.float64)
    parameter_variations = benchmark.parameter_variation("webui", cpu_limit - window * step, cpu_limit + window * step,
                                                         memory_limit - window * step, memory_limit + window * step,
                                                         number_of_pods - window, number_of_pods + window,
                                                         step, False, False, False, [int(rps)])
    # flatten possibilities
    predict_window_list = list(np.concatenate(parameter_variations).flat)
    # validate parameter variations
    predict_window_list = validate_parameter(predict_window_list, rps)
    # init arrays
    possibilities = len(predict_window_list)
    predictions = np.empty((len(models), possibilities))
    prediction_array = np.zeros([possibilities, len(models)], dtype=np.float64)
    predict_window = np.array(predict_window_list, dtype=np.float64)
    current_status_predicted = np.empty((len(models), 3))
    status_prediction = np.zeros([3, len(models)], dtype=np.float64)
    # load parameter scaler
    x_scaler = load(os.path.join(os.getcwd(), "data", "models", "data", f"x_scaler_average response time.gz"))
    # scale data
    predict_window_scaled = x_scaler.transform(predict_window)
    # get predictions for each model
    current_status_scaled = x_scaler.transform(current_status.reshape(1, -1))
    for i, model in enumerate(models):
        # predict
        predictions[i] = model.predict(predict_window_scaled)
        current_status_predicted[i] = model.predict(current_status_scaled)
    # load target scaler
    y_scalers = list()
    for t in ["average response time", "cpu usage", "memory usage"]:
        y_scalers.append(load(os.path.join(os.getcwd(), "data", "models", "data", f"y_scaler_{t}.gz")))
    # format into array
    for i in range(0, possibilities):
        for j in range(0, len(models)):
            prediction_array[i, j] = y_scalers[j].inverse_transform(predictions[j, i].reshape(1, -1))

    for j in range(0, len(models)):
        status_prediction[j] = y_scalers[j].inverse_transform(current_status_predicted[j, 0].reshape(1, -1))
    # concatenate targets and parameters
    prediction_array = np.concatenate((prediction_array, predict_window), axis=1)
    # validate targets
    logging.info(f"Targets: {len(prediction_array)}")
    prediction_array = validate_targets(prediction_array, status_prediction,
                                        np.array((response_time, cpu_usage, memory_usage), dtype=np.float64))
    # logging.info(f"Validated Targets: {len(prediction_array)}")
    if len(prediction_array) > 1:
        # delete rps parameter
        prediction_array = np.delete(prediction_array, -1, 1)
        # get index of best outcome
        # if horizontal scaling only delete cpu and memory limit
        if hpa:
            prediction_array_mod = np.delete(prediction_array, 3, 1)
            prediction_array_mod = np.delete(prediction_array_mod, 3, 1)
            best_outcome_index = choose_best(prediction_array_mod.tolist(), True)
        else:
            best_outcome_index = choose_best(prediction_array.tolist(), True)
        # get parameters of best outcome
        best_parameters = prediction_array[best_outcome_index]
        logging.info(
            f"Best Parameter: {prediction_array[best_outcome_index, 3]}m - {prediction_array[best_outcome_index, 4]}Mi - {prediction_array[best_outcome_index, 5]}")
        logging.info(
            f"Best Targets: {prediction_array[best_outcome_index, 0]}ms - {prediction_array[best_outcome_index, 1]}% - {prediction_array[best_outcome_index, 2]}%")
        return best_parameters
    elif len(prediction_array) == 1:
        best_outcome_index = 0
        best_parameters = prediction_array[best_outcome_index]
        logging.info(
            f"Best Parameter: {prediction_array[best_outcome_index, 3]}m - {prediction_array[best_outcome_index, 4]}Mi - {prediction_array[best_outcome_index, 5]}")
        logging.info(
            f"Best Targets: {prediction_array[best_outcome_index, 0]}ms - {prediction_array[best_outcome_index, 1]}% - {prediction_array[best_outcome_index, 2]}%")
        return best_parameters
    else:
        return None


def validate_targets(predictions: np.ndarray, curr_pred: np.array, curr: np.array) -> np.array:
    # init
    load_dotenv()
    i, j = predictions.shape
    validated = list()
    # calculate difference
    print(curr)
    print(curr_pred)
    r_diff = curr[0] - curr_pred[0, 0]
    c_diff = curr[1] - curr_pred[1, 0]
    m_diff = curr[2] - curr_pred[2, 0]
    logging.info(f"Diff: {r_diff}ms - {c_diff}% - {m_diff}%")
    # check every entry
    for ix in range(0, i):
        logging.info(f"Before: {predictions[ix]}")
        v = True
        for jx in range(0, j):
            # average response time:
            if jx == 0:
                predictions[ix, jx] = predictions[ix, jx] + r_diff
                if predictions[ix, jx] > curr[0]:
                    v = False
            # cpu usage
            elif jx == 1:
                predictions[ix, jx] = predictions[ix, jx] + c_diff
            # memory usage
            elif jx == 2:
                predictions[ix, jx] = predictions[ix, jx] + m_diff
        # append validated predictions
        logging.info(f"After: {predictions[ix]}")
        if v:
            validated.append(predictions[ix])
    # return list as array
    return np.array(validated)


def validate_parameter(data: list, rps: float) -> list:
    """
    Validates if the given limits are below the requested resources.
    :return: validated resources
    """
    # get requests
    request = k8s_tools.get_resource_requests()[os.getenv("SCALE_POD")]
    cpu_request = int(str(request["cpu"]).rstrip("m"))
    memory_request = int(str(request["memory"]).rstrip("Mi"))
    cpu_limit = 700
    memory_limit = 700
    # init
    validated = list()
    for entry in data:
        v = True
        cpu, memory, pods, tmp = entry
        if cpu < cpu_request:
            cpu = cpu_request
        if cpu > cpu_limit:
            cpu = cpu_limit
        if memory > memory_limit:
            memory = memory_limit
        if memory < memory_request:
            memory = memory_request
        if pods < 1:
            pods = 1
        if pods > int(os.getenv("MAX_PODS")):
            pods = int(os.getenv("MAX_PODS"))
        if v:
            validated.append((cpu, memory, pods, rps))
    # remove duplicates
    validated = list(dict.fromkeys(validated))
    return validated


def choose_best(mtx: np.array, method: bool) -> int:
    """
    Chooses the best alternative from given alternatives with multiple criteria.
    :param method: which mcdm method to use
    :param mtx: alternatives
    :return: index of best alternative
    """
    # min average response time, max cpu usage, max memory usage, min cpu limit, min memory limit, min number of pods
    load_dotenv()
    criteria = [MIN, MAX, MAX, MIN, MIN, MIN]
    # b is default
    if os.getenv("WEIGHTS") == "t":
        weights = [0.9, 0.02, 0.02, 0.02, 0.02, 0.02]
    elif os.getenv("WEIGHTS") == "r":
        weights = [0.1, 0.18, 0.18, 0.18, 0.18, 0.18]
    else:
        weights = [0.5, 0.1, 0.1, 0.1, 0.1, 0.1]
    # create DecisionMaker
    if method:
        dm = TOPSIS()
    else:
        dm = WeightedSum()
    # create data object
    data = Data(mtx=mtx, criteria=criteria, weights=weights)
    # make decision
    dec = dm.decide(data)
    logging.info(f"decisions: {dec}")
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
        # split data in to train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
        # scale dataset
        x_scaling = MinMaxScaler()
        y_scaling = MinMaxScaler()
        X_train = x_scaling.fit_transform(X_train)
        y_train = y_scaling.fit_transform(y_train)
        X_test = x_scaling.transform(X_test)
        y_test = y_scaling.transform(y_test)
        # save scaler
        dump(y_scaling, os.path.join(os.getcwd(), "data", "models", "data", f"y_scaler_{t}.gz"))
        dump(x_scaling, os.path.join(os.getcwd(), "data", "models", "data", f"x_scaler_{t}.gz"))
        # Save data
        logging.info(f"Training size: {X_train.shape}")
        logging.info(f"Test size: {X_test.shape}")
        d_path = os.path.join(os.getcwd(), "data", "models", "data", t)
        for i, d in enumerate([X_train, X_test, y_train, y_test]):
            np.save(os.path.join(d_path, str(i)), d)


def test_plot(alg: str):
    d = list()
    for p in range(1, 6):
        d.append((300, 400, p, 10))
    data = np.array(d, dtype=np.float64)
    # get models & scaler
    m = get_models(alg)
    scaler_path = os.path.join(os.getcwd(), "data", "models", "data")
    x_scaler = load(os.path.join(scaler_path, "x_scaler_average response time.gz"))
    y_scaler = load(os.path.join(scaler_path, "y_scaler_average response time.gz"))
    # predict
    data_scaled = x_scaler.transform(data)
    data_predicted = m[0].predict(data_scaled)
    prediction_inversed = y_scaler.inverse_transform(data_predicted.reshape(-1, 1))
    plt.scatter(data[:, 2], prediction_inversed)
    plt.show()