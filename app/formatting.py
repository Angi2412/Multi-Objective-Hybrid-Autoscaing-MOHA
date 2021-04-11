import logging
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from dotenv import load_dotenv

# init
load_dotenv()
logging.getLogger().setLevel(logging.INFO)


def get_all_data() -> list:
    """
    Gets all metric tables between two dates.
    :return: list of metric data
    """
    # init
    all_data = list()
    for d in get_directories():
        p_data, c_data, l_data = get_data(d)
        # append to list
        all_data.append([p_data, c_data, l_data])
    return all_data


def get_data(directory: str) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    """
    Gets data from prometheus.
    :return: prometheus data
    """
    # config
    load_dotenv(override=True)
    prometheus_data = None
    prometheus_custom_data = None
    locust_data = None
    # check if folder exists
    data_path = os.path.join(os.getcwd(), "data", "raw", directory)
    if os.path.exists(data_path):
        # search for prometheus metric files
        logging.info(f"Gets data from {directory}.")
        for (dir_path, dir_names, filenames) in os.walk(data_path):
            for file in filenames:
                if "metrics" in file and "custom_metrics" not in file:
                    i = int(str(file).split("_")[1].rstrip(".csv"))
                    prometheus_data = get_data_helper(prometheus_data, file, i, directory)
                elif "custom_metrics" in file:
                    j = int(str(file).split("_")[2].rstrip(".csv"))
                    print(j)
                    prometheus_custom_data = get_data_helper(prometheus_custom_data, file, j, directory)
                elif "locust" in file and "stats" in file and "history" not in file:
                    l = int(str(file).split("_")[2].rstrip(".csv"))
                    locust_data = get_data_helper(locust_data, file, l, directory)
    return prometheus_data, prometheus_custom_data, locust_data


def get_data_helper(data: pd.DataFrame, file: str, iteration: int, directory: str) -> pd.DataFrame:
    """
    Connects two dataframes.
    :param data: given dataframe
    :param file: data frame in file
    :param iteration: number of iteration
    :param directory: date
    :return: connected data frame
    """
    data_path = os.path.join(os.getcwd(), "data", "raw", directory)
    # concat metrics
    tmp_data = pd.read_csv(filepath_or_buffer=os.path.join(data_path, file), delimiter=',')
    tmp_data.insert(0, 'Iteration', iteration)
    if data is None:
        data = tmp_data
    else:
        data = pd.concat([data, tmp_data])
    return data


def get_directories() -> list:
    """
    Gets all directory names between the first and last data date.
    :return: list of directory names
    """
    load_dotenv()
    first_date = int(str(os.getenv("FIRST_DATA")).replace('-', "").strip())
    last_date = int(str(os.getenv("LAST_DATA")).replace('-', "").strip())
    base_path = os.path.join(os.getcwd(), "data", "raw")
    dirs = list()
    # get data from each run
    for (dir_path, dir_names, filenames) in os.walk(base_path):
        for c_dir in dir_names:
            if "dataset" not in c_dir:
                c_date = int(str(c_dir).replace('-', "").strip())
                if last_date >= c_date >= first_date:
                    dirs.append(c_dir)
    return dirs


def get_filtered_data(directory: str) -> pd.DataFrame:
    """
    Returns a data frame of a given filtered data.
    :param directory: date
    :return: data frame of filtered data
    """
    base_path = os.path.join(os.getcwd(), "data", "filtered")
    for (dir_path, dir_names, filenames) in os.walk(base_path):
        for c_file in filenames:
            if directory in c_file:
                df = pd.read_csv(os.path.join(base_path, c_file))
                return df


def get_all_filtered_data() -> list:
    """
    Reads all filtered data between two dates.
    :return: list of filtered data
    """
    load_dotenv()
    first_date = int(str(os.getenv("FIRST_DATA")).replace('-', "").strip())
    last_date = int(str(os.getenv("LAST_DATA")).replace('-', "").strip())
    base_path = os.path.join(os.getcwd(), "data", "filtered")
    files = list()
    # get data from each run
    for (dir_path, dir_names, filenames) in os.walk(base_path):
        for c_file in filenames:
            if str(c_file).endswith(".csv"):
                c_date = int(str(c_file).replace('-', "").replace(".csv", "").strip())
                if last_date >= c_date >= first_date:
                    files.append(pd.read_csv(os.path.join(base_path, c_file)))
    return files


def filter_all_data() -> None:
    """
    Filters all data between two dates.
    :return: None
    """
    # init
    i = 1
    dirs = get_directories()
    for d in dirs:
        # filter data in directory
        logging.info(f"Filtering data: {d} {i}/{len(dirs)}")
        filter_data(d)
        i = i + 1


def get_variation_matrix(directory: str) -> np.array:
    """
    Reads all variation matrices of a directory and puts them in a list.
    :param directory: current directory
    :return: variation matrix
    """
    dir_path = os.path.join(os.getcwd(), "data", "raw", directory)
    # find variation files
    for (dir_path, dir_names, filenames) in os.walk(dir_path):
        for file in filenames:
            if "variation" in file:
                # filter name
                name = str(file).split("-")[1].split("_")[0]
                # read variation file
                file_path = os.path.join(dir_path, file)
                res = pd.read_csv(filepath_or_buffer=file_path, delimiter=',')
                # edit table
                res.insert(0, 'pod', name)
                res.rename(columns={"Unnamed: 0": "Iteration"}, inplace=True)
                res.reset_index()
                return res


def filter_data(directory: str) -> pd.DataFrame:
    """
    Filters data from prometheus.
    :return: filtered data
    """
    normal, custom, locust = get_data(directory)
    # filter for namespace
    filtered_data = pd.concat(objs=[normal[normal.namespace.eq(os.getenv("NAMESPACE"))]])
    # read variation matrices
    variation = get_variation_matrix(directory)
    # filter for pod name
    filtered_data["pod"] = filtered_data["pod"].str.split("-", n=2).str[1]
    custom["pod"] = custom["pod"].str.split("-", n=2).str[1]
    # count data points per iteration
    filtered_data['datapoint'] = filtered_data.groupby(["Iteration"]).cumcount() + 1
    custom['datapoint'] = custom.groupby(["Iteration"]).cumcount() + 1
    custom['pod'].fillna("webui", inplace=True)
    # create pivot tables
    filtered_data = pd.pivot_table(filtered_data, index=["Iteration", "pod", "datapoint"], columns=["__name__"],
                                   values="value").reset_index()
    filtered_custom_data = pd.pivot_table(custom, index=["Iteration", "pod", "datapoint"], columns=["metric"],
                                          values="value").reset_index()
    # calculate median values
    filtered_data = filtered_data.groupby(["Iteration", "pod"]).mean().reset_index()
    filtered_custom_data = filtered_custom_data.groupby(["Iteration", "pod"]).mean().reset_index()
    filtered_custom_data.rename(columns={"rps": "average rps"}, inplace=True)
    # outliers
    filtered_custom_data["median_latency"] = np.where(
        filtered_custom_data["median_latency"] < filtered_custom_data["median_latency"].quantile(0.10),
        filtered_custom_data["median_latency"].quantile(0.10),
        filtered_custom_data['median_latency'])
    filtered_custom_data["median_latency"] = np.where(
        filtered_custom_data["median_latency"] > filtered_custom_data["median_latency"].quantile(0.90),
        filtered_custom_data["median_latency"].quantile(0.90),
        filtered_custom_data['median_latency'])
    # merge all tables
    res_data = pd.merge(filtered_data, filtered_custom_data, how='left', on=["Iteration", "pod"])
    res_data = pd.merge(res_data, variation, how='left', on=["Iteration", "pod"])
    # erase stuff
    res_data.drop(columns=["kube_deployment_spec_replicas", "kube_pod_container_resource_limits_cpu_cores",
                           "kube_pod_container_resource_limits_memory_bytes",
                           "kube_pod_container_resource_requests_cpu_cores",
                           "kube_pod_container_resource_requests_memory_bytes",
                           "datapoint_x", "datapoint_y"], inplace=True)
    res_data.rename(
        columns={"cpu": "cpu usage", "memory": "memory usage", "CPU": "cpu limit", "Memory": "memory limit",
                 "Pods": "number of pods", "container_cpu_cfs_throttled_seconds_total": "cpu throttled total",
                 "response_time": "average response time", "median_latency": "median latency"},
        inplace=True)
    # ratio
    res_data["rps delta"] = (res_data["average rps"] - res_data["RPS"]) / res_data["RPS"]
    res_data["ratio response time"] = res_data["median latency"] * (1 - res_data["rps delta"])
    res_data["ratio cpu usage"] = res_data["cpu usage"] * (1 - res_data["rps delta"])
    res_data["ratio memory usage"] = res_data["memory usage"] * (1 - res_data["rps delta"])
    # filter for webui pod
    res_data = res_data.loc[(res_data['pod'] == "webui")]
    res_data.reset_index(inplace=True)
    res_data.drop(columns=["index"], inplace=True)
    # save
    save_data(res_data, directory, "filtered")
    return res_data


def save_data(data: pd.DataFrame, directory: str, mode: str) -> None:
    """
    Saves a given data frame in a given folder.
    :param data: data frame
    :param directory: name of file
    :param mode: name of directory
    :return: None
    """
    save_path = os.path.join(os.getcwd(), "data", mode, f"{directory}.csv")
    if not os.path.exists(save_path):
        data.to_csv(path_or_buf=save_path)
    else:
        logging.warning("Filtered data already exists.")


def plot_filtered_data(data: pd.DataFrame, name: str) -> None:
    """
    Plots a given metric from filtered data from prometheus.
    :return: None
    """
    # create directory
    dir_path = os.path.join(os.getcwd(), "data", "plots", f"{name}")
    os.mkdir(dir_path)
    # init x- and y-axis
    x_axis = ["cpu limit", "memory limit", "number of pods"]
    y_axis = ["ratio response time", "ratio cpu usage",
              "ratio memory usage"]
    # functions
    functions = [pd.DataFrame.min, pd.DataFrame.median, pd.DataFrame.max]
    # create and save plots
    plot = None
    for fn in functions:
        for y in y_axis:
            for x in x_axis:
                if x == "number of pods":
                    data_pods = data.loc[(data['memory limit'] == fn(data['memory limit'])) & (
                            data['cpu limit'] == fn(data['cpu limit']))]
                    plot = sns.lineplot(data=data_pods, x=x, y=y)
                elif x == "memory limit":
                    data_memory = data.loc[(data['cpu limit'] == fn(data['cpu limit']))]
                    plot = sns.lineplot(data=data_memory, x=x, y=y, hue="number of pods")
                elif x == "cpu limit":
                    data_cpu = data.loc[(data['memory limit'] == fn(data['memory limit']))]
                    plot = sns.lineplot(data=data_cpu, x=x, y=y, hue="number of pods")
                name = f"{x}_{y}_{fn.__name__}.png"
                plot.figure.savefig(os.path.join(dir_path, name))
                plot.figure.clf()
    # Requests per second
    fig, ax = plt.subplots()
    ax.plot(data["Iteration"], data["average rps"])
    ax.plot(data["Iteration"], data["RPS"])
    fig.savefig(os.path.join(dir_path, "rps.png"))
    fig.clf()


def format_for_extra_p() -> None:
    """
    Formats a given benchmark for extra-p.
    :return: None
    """
    # init
    load_dotenv()
    save_path = os.path.join(os.getcwd(), "data", "formatted", os.getenv("LAST_DATA"))
    # create directory if not existing
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    # get variation matrix
    variation = get_variation_matrix(os.getenv("LAST_DATA"))
    m_max = variation["Memory"].nunique()
    # parameter and metrics
    parameter = ["cpu limit", "memory limit", "number of pods", "rps"]
    targets = ["average response time", "cpu usage", "memory usage"]
    # get all filtered data
    filtered_data = list()
    for f in get_all_filtered_data():
        filtered_data.append(f)
    # write in txt file
    for metric in targets:
        m_name = (re.sub('[^a-zA-Z0-9 _]', '', metric)).rstrip().replace(' ', '_').lower()
        with open(os.path.join(save_path, f"{os.getenv('LAST_DATA')}_{m_name}_extra-p.txt"), "x") as file:
            # write parameters
            for par in parameter:
                file.write(f"PARAMETER {(re.sub('[^a-zA-Z0-9 _]', '', par)).rstrip().replace(' ', '_').lower()}\n")
            file.write("\n")
            # write coordinates
            # for every iteration
            for i, v in enumerate(variation.index):
                if i % m_max == 0:
                    file.write("\n")
                    file.write("POINTS ")
                row = variation.iloc[[i]]
                file.write(
                    f"( {row.iloc[0]['CPU']} {row.iloc[0]['Memory']} {row.iloc[0]['Pods']} {row.iloc[0]['RPS']} ) ")
            file.write("\n")
            file.write("\n")
            file.write(f"REGION {os.getenv('APP_NAME')}\n")
            file.write(f"METRIC {m_name}\n")
            file.write("\n")
            # write data
            # for every datapoint
            for i in range(0, (filtered_data[0].index.max() + 1)):
                logging.info(f"format data: {i + 1}/{(filtered_data[0].index.max() + 1)}")
                file.write("DATA ")
                # for test purposes
                for f in filtered_data:
                    x = f.loc[f.index == i, metric].iloc[0]
                    file.write(f"{x} ")
                file.write("\n")


def correlation_coefficient_matrix() -> None:
    """
    Calculates and plots the correlation coefficient matrix for a given dataframe.
    :return: None
    """
    dir_path = os.path.join(os.getcwd(), "data", "correlation", os.getenv("LAST_DATA"))
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    df = pd.read_csv(os.path.join(os.getcwd(), "data", "combined", f"{os.getenv('LAST_DATA')}.csv"))
    df = df[
        ["cpu limit", "memory limit", "number of pods", "ratio response time", "ratio cpu usage",
         "ratio memory usage"]]
    df.dropna()
    corr = df.corr(method="pearson")
    save_data(corr, os.getenv("LAST_DATA"), os.path.join("correlation", os.getenv("LAST_DATA")))
    # plot correlation
    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))
    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))
    # Generate a custom diverging colormap
    cmap = sns.color_palette("coolwarm", as_cmap=True)
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0, square=True, linewidths=.5,
                cbar_kws={"shrink": .5},
                annot=True, fmt=".2f")
    f.savefig(os.path.join(dir_path, f"{os.getenv('LAST_DATA')}.png"), bbox_inches='tight')
    plt.show()


def combine_runs() -> None:
    """
    Combines data from all runs.
    :return: None
    """
    data = get_directories()
    data_path = os.path.join(os.getcwd(), "data", "filtered")
    tmp = list()
    for i, file in enumerate(data):
        tmp_data = pd.read_csv(filepath_or_buffer=os.path.join(data_path, f"{file}.csv"), delimiter=',')
        tmp_data.insert(0, 'run', i + 1)
        tmp_data = tmp_data.loc[(tmp_data['pod'] == "webui")]
        tmp.append(tmp_data)
    result = pd.concat(tmp)
    save_data(result, os.getenv("LAST_DATA"), "combined")


def combine_data(data: list, name: str) -> None:
    """
    Combines data from all runs.
    :return: None
    """
    data_path = os.path.join(os.getcwd(), "data", "filtered")
    tmp = list()
    for i, file in enumerate(data):
        tmp_data = pd.read_csv(filepath_or_buffer=os.path.join(data_path, f"{file}.csv"), delimiter=',')
        tmp_data.insert(0, 'run', i + 1)
        tmp_data = tmp_data.loc[(tmp_data['pod'] == "webui")]
        tmp.append(tmp_data)
    result = pd.concat(tmp)
    save_data(result, name, "combined")


def filter_run() -> None:
    """
    Filters all data from a run.
    :return: None
    """
    path = os.path.join(os.getcwd(), "data", "combined", f"{os.getenv('LAST_DATA')}.csv")
    data = pd.read_csv(path, delimiter=",")
    data = data.groupby(["Iteration", "pod"]).median()
    save_data(data, f"{os.getenv('LAST_DATA')}_median", "combined")


def plot_run() -> None:
    """
    Plots all data from a run.
    :return: None
    """
    # plot combined run
    path = os.path.join(os.getcwd(), "data", "combined", f"{os.getenv('LAST_DATA')}.csv")
    data = pd.read_csv(path, delimiter=",")
    plot_filtered_data(data, f"{os.getenv('LAST_DATA')}_combined")
    # plot mean run
    path = os.path.join(os.getcwd(), "data", "combined", f"{os.getenv('LAST_DATA')}_median.csv")
    data = pd.read_csv(path, delimiter=",")
    plot_filtered_data(data, f"{os.getenv('LAST_DATA')}_combined_median")


def plot_all_data():
    """
    Plots data from all runs.
    :return: None
    """
    directory = os.path.join(os.getcwd(), "data", "plots", os.getenv("LAST_DATA"))
    # creates folder if does not exist
    if not os.path.exists(directory):
        os.mkdir(directory)
    for i, file in enumerate(get_all_filtered_data()):
        plot_filtered_data(file, os.path.join(os.getenv("LAST_DATA"), str(i)))


def plot_targets_4d(data: pd.DataFrame, name: str) -> None:
    """
    Plot all parameters and each target in a 4D plot.
    :param data: dataset
    :param name: save name
    :return: None
    """
    targets = ["average response time", "cpu usage", "memory usage"]
    for t in targets:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        x = data["cpu limit"]
        y = data["memory limit"]
        z = data["number of pods"]
        c = data[t]

        img = ax.scatter(x, y, z, c=c, cmap=plt.jet())
        fig.colorbar(img)
        plt.show()
        # save figure
        save_path = os.path.join(os.getcwd(), "data", "plots", name)
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        fig.savefig(os.path.join(save_path, f"{t}_3D.png"))
        fig.clf()


def get_combined_data() -> pd.DataFrame:
    # get last combined run
    path = os.path.join(os.getcwd(), "data", "combined", f"{os.getenv('LAST_DATA')}.csv")
    data = pd.read_csv(path, delimiter=",")
    return data


def histogram() -> None:
    sns.set_style("whitegrid")
    data = get_combined_data()
    ax = sns.histplot(data, x="average response time",
                      bins=50, binrange=(250, 5000))
    ax.set_xlabel("Average Response Time [ms]")
    plt.show()


def boxplot() -> None:
    data = get_combined_data()
    ax = sns.boxplot(data=data[["cpu usage", "memory usage"]])
    ax.set_ylabel("Usage [%]")
    plt.show()


def stats(column: str) -> None:
    data = get_combined_data()
    print(data[column].describe())


def process_run() -> None:
    """
    Processes one single run.
    :return: None
    """
    filter_data(os.getenv("LAST_DATA"))
    plot_filtered_data(get_filtered_data(os.getenv("LAST_DATA")), os.getenv("LAST_DATA"))


def process_all_runs() -> None:
    """
    Processes all runs between start- and last data.
    :return: None
    """
    filter_all_data()
    plot_all_data()
    combine_runs()
    filter_run()
    plot_run()
    format_for_extra_p()
    correlation_coefficient_matrix()


def formatting_evaluation(date: str) -> (pd.DataFrame, pd.DataFrame):
    # get data
    dir_path = os.path.join(os.getcwd(), "data", "raw", f"{date}")
    custom = pd.read_csv(os.path.join(dir_path, "custom_metrics_0.csv"), delimiter=',')
    normal = pd.read_csv(os.path.join(dir_path, "metrics_0.csv"), delimiter=',')
    # filter for namespace
    filtered_data = pd.concat(objs=[normal[normal.namespace.eq(os.getenv("NAMESPACE"))]])
    # filter for pod name
    filtered_data["pod"] = filtered_data["pod"].str.split("-", n=2).str[1]
    custom["pod"] = custom["pod"].str.split("-", n=2).str[1]
    custom['pod'].fillna("webui", inplace=True)
    # timestamps in seconds
    filtered_data["timestamp"] = pd.to_datetime(filtered_data["timestamp"], unit='s', origin='unix')
    filtered_data["timestamp"] = (filtered_data["timestamp"] - filtered_data["timestamp"].min())
    custom["timestamp"] = pd.to_datetime(custom["timestamp"], unit='s', origin='unix')
    custom["timestamp"] = (custom["timestamp"] - custom["timestamp"].min())
    filtered_data["timestamp"] = (filtered_data.timestamp / np.timedelta64(1, 'm'))
    custom["timestamp"] = (custom.timestamp / np.timedelta64(1, 'm'))
    filtered_data["timestamp"] = filtered_data["timestamp"] + 0.5
    custom["timestamp"] = custom["timestamp"] + 0.5
    # filter for pod
    filtered_data_a = filtered_data[filtered_data['deployment'] == "teastore-webui"]
    filtered_data_b = filtered_data[filtered_data['pod'] == "webui"]
    filtered_data = pd.concat([filtered_data_a, filtered_data_b])
    custom = custom[(custom['pod'] == "webui")]
    # pivot
    filtered_data = pd.pivot_table(filtered_data, index=["timestamp"], columns=["__name__"],
                                   values="value").reset_index()
    filtered_custom_data = pd.pivot_table(custom, index=["timestamp"], columns=["metric"],
                                          values="value").reset_index()
    filtered_custom_data.set_index("timestamp")
    # fix units
    filtered_data["kube_pod_container_resource_limits_cpu_cores"] = filtered_data[
                                                                        "kube_pod_container_resource_limits_cpu_cores"] * 1000
    filtered_data["kube_pod_container_resource_limits_memory_bytes"] = filtered_data[
                                                                           "kube_pod_container_resource_limits_memory_bytes"] / 1048576
    filtered_data.to_csv(os.path.join(dir_path, "filtered.csv"))
    filtered_custom_data.to_csv(os.path.join(dir_path, "filtered_custom.csv"))
    return filtered_data, filtered_custom_data, dir_path


def plot_evaluation(date: str):
    # format raw data
    n, c, dir_path = formatting_evaluation(date)

    # kube metrics
    nfig, (nax1, nax2, nax3) = plt.subplots(nrows=3, ncols=1)
    # cpu limit
    sns.lineplot(data=n, x="timestamp", y="kube_pod_container_resource_limits_cpu_cores", ax=nax1)
    nax1.set_ylabel("CPU limit [m]")
    # memory limit
    sns.lineplot(data=n, x="timestamp", y="kube_pod_container_resource_limits_memory_bytes", ax=nax2)
    nax2.set_ylabel("Memory limit [Mi]")
    # number of pods
    sns.lineplot(data=n, x="timestamp", y="kube_deployment_spec_replicas", ax=nax3)
    nax3.set_ylabel("Number of pods")
    nfig.savefig(os.path.join(dir_path, "parameters.png"))
    nfig.clf()

    # custom metrics
    cfig, (cax1, cax2, cax3) = plt.subplots(nrows=3, ncols=1)
    # average response time
    sns.lineplot(data=c, x="timestamp", y="response_time", ax=cax1)
    cax1.set_ylabel("Response time [ms]")
    # cpu
    sns.lineplot(data=c, x="timestamp", y="cpu", ax=cax2)
    cax2.set_ylabel("CPU usage [%]")
    # memory
    sns.lineplot(data=c, x="timestamp", y="memory", ax=cax3)
    cax3.set_ylabel("Memory usage [%]")
    cfig.savefig(os.path.join(dir_path, "targets.png"))
    cfig.clf()

    description = {"response_time": "Average response time [ms]", "cpu": "CPU usage [%]", "memory": "Memory usage [%]"}
    for t in ["response_time", "cpu", "memory"]:
        ax = sns.histplot(c, x=t)
        ax.set_xlabel(description[t])
        fig = ax.get_figure()
        fig.savefig(os.path.join(dir_path, f"{t}_histogram.png"))
        fig.clf()
    # rps
    ax = sns.lineplot(data=c, x="timestamp", y="rps")
    ax.set_xlabel("Minutes")
    ax.set_ylabel("Requests per second")
    ax.figure.savefig(os.path.join(dir_path, "rps.png"))
    cfig.clf()
    calc_eval_metrics(c, n, dir_path)


def calc_eval_metrics(c: pd.DataFrame, n: pd.DataFrame, dir_path: str):
    # init thresholds
    r = int(os.getenv("TARGET_RESPONSE"))
    min_usage = int(os.getenv("MIN_USAGE"))
    max_usage = int(os.getenv("MAX_USAGE"))
    # validate
    response_time_exceeded = c[c.response_time > r].count()
    response_time_valid = c[c.response_time <= r].count()
    cpu_in_threshold = c[(min_usage <= c.cpu) & (c.cpu <= max_usage)].count()
    cpu_out_threshold = c[(c.cpu < min_usage) | (c.cpu > max_usage)].count()
    memory_in_threshold = c[(min_usage <= c.memory) & (c.memory <= max_usage)].count()
    memory_out_threshold = c[(c.memory < min_usage) | (c.memory > max_usage)].count()
    # create dataframe
    result = pd.DataFrame()
    result["response time exceeded"] = response_time_exceeded
    result["response time valid"] = response_time_valid
    result["cpu in threshold"] = cpu_in_threshold
    result["cpu out of threshold"] = cpu_out_threshold
    result["memory in threshold"] = memory_in_threshold
    result["memory out of threshold"] = memory_out_threshold
    result["average cpu limit"] = n["kube_pod_container_resource_limits_cpu_cores"].mean()
    result["average memory limit"] = n["kube_pod_container_resource_limits_memory_bytes"].mean()
    result["average pods"] = n["kube_deployment_spec_replicas"].mean()
    result["metric 50/50"] = 0.5 * response_time_exceeded + 0.5 * (cpu_out_threshold + memory_out_threshold) * n["kube_deployment_spec_replicas"].mean()
    result["median response time"] = c["response_time"].median()
    result["average cpu usage"] = c["cpu"].mean()
    result["average memory usage"] = c["memory"].mean()
    result.to_csv(os.path.join(dir_path, "results.csv"))


def plot_all_evaluation():
    raw_path = os.path.join(os.getcwd(), "data", "raw")
    for (dir_path, dir_names, filenames) in os.walk(raw_path):
        for d in dir_names:
            plot_evaluation(d)


if __name__ == '__main__':
    plot_all_evaluation()
