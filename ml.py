from gevent import monkey

monkey.patch_all()
import os

import pandas as pd
from dotenv import load_dotenv
import seaborn as sns
import matplotlib.pyplot as plt

# init
load_dotenv()


def get_locust_data():
    locust_file = os.path.join(os.getcwd(), "data", "raw", f"{os.getenv('LAST_DATA')}_locust_stats_history.csv")
    locust_data = pd.read_csv(filepath_or_buffer=locust_file)
    locust_data = locust_data.rename(columns={"Timestamp": "timestamp"})
    locust_data["timestamp"] = locust_data["timestamp"].apply(lambda x: x * 1000)
    locust_data["timestamp"] = pd.to_datetime(locust_data["timestamp"], unit='ms')
    return locust_data


def get_prometheus_data() -> pd.DataFrame:
    # get data
    prometheus_file = os.path.join(os.getcwd(), "data", "raw", f"{os.getenv('LAST_DATA')}_metrics.csv")
    prometheus_data = pd.read_csv(filepath_or_buffer=prometheus_file)
    # convert timestamp to date
    prometheus_data["timestamp"] = prometheus_data["timestamp"].apply(lambda x: x * 1000)
    prometheus_data["timestamp"] = pd.to_datetime(prometheus_data["timestamp"], unit='ms')
    return prometheus_data


def filter_data() -> pd.DataFrame:
    prometheus = get_prometheus_data()
    # filter by namespace
    filtered_data = pd.concat(objs=[prometheus[prometheus.namespace.eq(os.getenv("NAMESPACE"))],
                                    prometheus[prometheus.dst_namespace.eq(os.getenv("NAMESPACE"))]])
    # create pivot table
    filtered_data = pd.pivot_table(filtered_data, index=["timestamp"], columns=["__name__"], values="value")
    # resample by 1 second
    # filtered_data = filtered_data.resample("1s").mean()
    # interpolate missing data
    filtered_data_interpolated = filtered_data.interpolate()
    return filtered_data_interpolated


def plot_filtered_data():
    data = filter_data()

    data.dropna(inplace=True)
    sns.lineplot(x="timestamp", y="container_cpu_usage_seconds_total", data=data)
    sns.lineplot(x="timestamp", y="kube_pod_container_resource_requests_cpu_cores", data=data)
    sns.lineplot(x="timestamp", y="kube_pod_container_resource_limits_cpu_cores", data=data)
    plt.show()


if __name__ == '__main__':
    plot_filtered_data()
