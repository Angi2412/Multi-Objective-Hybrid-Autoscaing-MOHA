from gevent import monkey

monkey.patch_all()
import os

import pandas as pd
from dotenv import load_dotenv
import seaborn as sns
import logging
import re
from sandbox import parameter_variation

# init
load_dotenv()
# init logger
logging.getLogger().setLevel(logging.INFO)


def get_data() -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    """
    Gets data from prometheus.
    :return: prometheus data
    """
    # config
    load_dotenv(override=True)
    prometheus_data = None
    prometheus_custom_data = None
    locust_data = None
    i, j, l = 0, 0, 0
    # check if folder exists
    data_path = os.path.join(os.getcwd(), "data", "raw", os.getenv("LAST_DATA"))
    if os.path.exists(data_path):
        # search for prometheus metric files
        for (dir_path, dir_names, filenames) in os.walk(data_path):
            for file in filenames:
                if "metrics" in file and "custom_metrics" not in file:
                    i = i + 1
                    prometheus_data = get_data_helper(prometheus_data, file, i)
                elif "custom_metrics" in file:
                    j = j + 1
                    prometheus_custom_data = get_data_helper(prometheus_custom_data, file, j)
                elif "locust" in file and "stats" in file and not "history" in file:
                    l = l + 1
                    locust_data = get_data_helper(locust_data, file, l)
    return prometheus_data, prometheus_custom_data, locust_data


def get_data_helper(data: pd.DataFrame, file: str, iteration: int) -> pd.DataFrame:
    data_path = os.path.join(os.getcwd(), "data", "raw", os.getenv("LAST_DATA"))
    # concat metrics
    tmp_data = pd.read_csv(filepath_or_buffer=os.path.join(data_path, file), delimiter=',')
    tmp_data.insert(0, 'Iteration', iteration)
    if data is None:
        data = tmp_data
    else:
        data = pd.concat([data, tmp_data])
    return data


def filter_data() -> pd.DataFrame:
    """
    Filters data from prometheus.
    :return: filtered data
    """
    result = pd.DataFrame(
        columns=["cpu usage [%]", "memory usage [%]", "Average response time [ms]", "Failures [%]"])
    normal, custom, locust = get_data()
    variation_path = os.path.join(os.getcwd(), "data", "raw", os.getenv("LAST_DATA"), "variation_matrix.csv")
    # filter by namespace
    filtered_data = pd.concat(objs=[normal[normal.container.eq(os.getenv("APP_NAME"))]])
    filtered_custom_data = pd.concat(objs=[custom[custom.container.eq(os.getenv("APP_NAME"))]])
    variation = pd.read_csv(filepath_or_buffer=variation_path, delimiter=',')
    variation.index = variation["Unnamed: 0"]
    # create pivot table
    filtered_data = pd.pivot_table(filtered_data, index=["Iteration"], columns=["__name__"],
                                   values="value").reset_index()

    filtered_data = filtered_data.groupby("Iteration").mean()
    filtered_custom_data['cpu'] = filtered_custom_data['cpu'].fillna("memory")
    filtered_custom_data['cpu'] = filtered_custom_data['cpu'].replace("total", "cpu")
    filtered_custom_data = pd.pivot_table(filtered_custom_data, index=["Iteration"], columns=["cpu"],
                                          values="value").reset_index()
    filtered_custom_data = filtered_custom_data.groupby("Iteration").mean()
    locust = locust.loc[locust['Name'] == "Aggregated"]
    locust.index = locust["Iteration"]
    # fill result
    result["CPU usage [%]"] = filtered_custom_data['cpu'] / filtered_data[
        "kube_pod_container_resource_limits_cpu_cores"] * 100
    result["Memory usage [%]"] = filtered_custom_data['memory'] / filtered_data[
        "kube_pod_container_resource_limits_memory_bytes"] * 100
    result["Average response time [ms]"] = locust["Average Response Time"]
    result["Failures [%]"] = locust["Failure Count"] / locust["Request Count"] * 100
    result["Number of pods"] = variation["Pods"]
    result["CPU limit"] = filtered_data["kube_pod_container_resource_limits_cpu_cores"] * 100
    result["Memory limit"] = filtered_data["kube_pod_container_resource_limits_memory_bytes"] / 1048576
    result["Given CPU limit"] = variation["CPU"]
    result["Given memory limit"] = variation["Memory"]
    save_data(result)
    return result


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
    # create directory
    dir_path = os.path.join(os.getcwd(), "data", "plots", f"{os.getenv('LAST_DATA')}")
    os.mkdir(dir_path)
    # create and save plots
    for metric in data:
        plt = sns.lineplot(data=data, x=data.index, y=metric)
        plt.figure.savefig(os.path.join(dir_path, f"{metric}.png"))
        plt.figure.clf()
    # make scatter plot
    g = sns.PairGrid(data)
    g.map(sns.scatterplot)
    g.savefig(os.path.join(dir_path, f"scatterplot.png"))
    g.fig.clf()


def format_for_extra_p(data: pd.DataFrame) -> None:
    # init
    save_path = os.path.join(os.getcwd(), "data", "formatted", os.getenv('LAST_DATA'))
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    # get variation matrix
    variation_path = os.path.join(os.getcwd(), "data", "raw", os.getenv("LAST_DATA"), "variation_matrix.csv")
    variation = parameter_variation(cpu_limit=int(os.getenv("CPU_LIMIT")), memory_limit=int(os.getenv("MEMORY_LIMIT")),
                                    pods_limit=int(os.getenv("PODS_LIMIT")))
    variation_df = pd.read_csv(filepath_or_buffer=variation_path, delimiter=',')
    c_max, m_max, p_max = variation.shape
    # parameter and metrics
    parameter = ["CPU limit", "Memory limit", "Number of pods"]
    metrics = ["Average response time [ms]", "Failures [%]", "Memory usage [%]", "CPU usage [%]"]
    # write in txt file
    for metric in metrics:
        m_name = (re.sub('[^a-zA-Z0-9 _]', '', metric)).rstrip().replace(' ', '_').lower()
        with open(os.path.join(save_path, f"{os.getenv('LAST_DATA')}_{m_name}_extrap.txt"), "x") as file:
            # write parameters
            for par in parameter:
                file.write(f"PARAMETER {(re.sub('[^a-zA-Z0-9 _]', '', par)).rstrip().replace(' ', '_').lower()}\n")
            file.write("\n")
            # write coordinates

            # for every iteration
            for c in range(0, c_max):
                for m in range(0, m_max):
                    file.write("POINTS ")
                    for p in range(0, p_max):
                        file.write('( ')
                        for v in variation[c, m, p]:
                            file.write(f"{v} ")
                        file.write(') ')
                    file.write("\n")
            file.write("\n")
            file.write("REGION Test\n")
            file.write(f"METRIC {m_name}\n")
            # write data
            # for every datapoint
            for i in range(1, (data.index.max() + 1)):
                file.write("DATA ")
                # for test purposes
                #for j in range(0, 5):
                x = data.loc[data.index == i, metric].iloc[0]
                file.write(f"{x} ")
                file.write("\n")


if __name__ == '__main__':
    format_for_extra_p(filter_data())
