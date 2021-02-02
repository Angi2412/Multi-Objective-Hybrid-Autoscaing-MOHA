from gevent import monkey

monkey.patch_all()
import os

import pandas as pd
from dotenv import load_dotenv
import seaborn as sns
import logging
import re

# init
load_dotenv()
# init logger
logging.getLogger().setLevel(logging.INFO)


def get_prometheus_data() -> pd.DataFrame:
    """
    Gets data from prometheus.
    :return: prometheus data
    """
    # config
    load_dotenv(override=True)
    prometheus_data = None
    i = 0
    # check if folder exists
    data_path = os.path.join(os.getcwd(), "data", "raw", os.getenv("LAST_DATA"))
    if os.path.exists(data_path):
        # search for prometheus metric files
        for (dir_path, dir_names, filenames) in os.walk(data_path):
            for file in filenames:
                if "metrics" in file:
                    i = i + 1
                    # concat metrics
                    tmp_data = pd.read_csv(filepath_or_buffer=os.path.join(data_path, file), delimiter=',')
                    tmp_data.insert(0, 'Iteration', i)
                    if prometheus_data is None:
                        prometheus_data = tmp_data
                    else:
                        prometheus_data = pd.concat([prometheus_data, tmp_data])
    return prometheus_data


def filter_data() -> pd.DataFrame:
    """
    Filters data from prometheus.
    :return: filtered data
    """
    prometheus = get_prometheus_data()
    # filter by namespace
    filtered_data = pd.concat(objs=[prometheus[prometheus.namespace.eq(os.getenv("NAMESPACE"))],
                                    prometheus[prometheus.dst_namespace.eq(os.getenv("NAMESPACE"))]])
    # convert timestamp
    filtered_data["timestamp"] = filtered_data["timestamp"].apply(lambda x: x * 1000)
    filtered_data["timestamp"] = pd.to_datetime(filtered_data["timestamp"], unit="ms")
    # create pivot table
    filtered_data = pd.pivot_table(filtered_data, index=["timestamp", "Iteration"], columns=["__name__"],
                                   values="value").reset_index()
    # interpolate missing data
    interpolated = filtered_data.interpolate()
    interpolated["datapoints"] = interpolated.groupby(['Iteration']).cumcount() + 1

    interpolated["CPU usage [%]"] = interpolated["container_cpu_usage_seconds_total"] / interpolated[
        "kube_pod_container_resource_limits_cpu_cores"] * 100
    interpolated["Memory usage [%]"] = interpolated["container_memory_usage_bytes"] / interpolated[
        "kube_pod_container_resource_limits_memory_bytes"] * 100
    interpolated["Average response time [ms]"] = interpolated["response_latency_ms_sum"] / interpolated[
        "response_latency_ms_count"]
    # save to csv
    save_data(interpolated)
    return interpolated


def save_data(data: pd.DataFrame) -> None:
    save_path = os.path.join(os.getcwd(), "data", "filtered", f"{os.getenv('LAST_DATA')}_filtered.csv")
    if not os.path.exists(save_path):
        data.to_csv(path_or_buf=save_path)
    else:
        logging.warning("Filtered data already exists.")


def plot_filtered_data() -> None:
    """
    Plots a given metric from filtered data from prometheus.
    :return: None
    """
    # init
    data = filter_data()
    metrics = ["CPU usage [%]", "Memory usage [%]", "Average response time [ms]"]
    # create directory
    dir_path = os.path.join(os.getcwd(), "data", "plots", f"{os.getenv('LAST_DATA')}")
    os.mkdir(dir_path)
    # create and save plots
    for metric in metrics:
        plt = sns.lineplot(data=data, x="datapoints", y=metric, hue="Iteration")
        plt.figure.savefig(os.path.join(dir_path, f"{metric}.png"))
        plt.figure.clf()


def format_for_extra_p(data: pd.DataFrame) -> None:
    # init
    save_path = os.path.join(os.getcwd(), "data", "formatted", f"{os.getenv('LAST_DATA')}_extrap.txt")
    metrics = ["CPU usage [%]", "Memory usage [%]",
               "kube_pod_container_resource_limits_cpu_cores", "kube_pod_container_resource_limits_memory_bytes"]
    # find min and max of iterations and data points
    min_i = data['Iteration'].min()
    max_i = data['Iteration'].max()
    min_d = data['datapoints'].min()
    max_d = data['datapoints'].max()
    # erase Nan entries
    data = data.dropna()
    # write in txt file
    with open(save_path, "x") as file:
        # write parameters
        for metric in metrics:
            file.write(f"PARAMETER {(re.sub('[^a-zA-Z0-9 _]', '', metric)).rstrip().replace(' ', '_')}\n")
        file.write("\n")
        # write coordinates
        # for every row
        for d in range(min_d, max_d):
            file.write("POINTS ")
            # for every iteration
            for i in range(min_i, max_i):
                for index, row in data.iterrows():
                    if row["Iteration"] is i and row["datapoints"] is d:
                        file.write('(')
                        for m in metrics:
                            file.write(f"{row[m]} ")
                        file.write(')')
            file.write("\n")
        file.write("\n")
        file.write("REGION Test\n")
        file.write("METRIC average_response_time\n")
        file.write("\n")
        # write data
        # for every datapoint
        for d in range(min_d, max_d):
            file.write("DATA ")
            # for every iteration
            for i in range(min_i, max_i):
                for index, row in data.iterrows():
                    if row["Iteration"] is i and row["datapoints"] is d:
                        file.write(str(row["Average response time [ms]"]))
                        file.write(" ")
            file.write("\n")


if __name__ == '__main__':
    format_for_extra_p(filter_data())
