from gevent import monkey

monkey.patch_all()
import os

import pandas as pd
from dotenv import load_dotenv
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


def get_prometheus_data():
    prometheus_file = os.path.join(os.getcwd(), "data", "raw", f"{os.getenv('LAST_DATA')}_metrics.csv")
    prometheus_data = pd.read_csv(filepath_or_buffer=prometheus_file)
    prometheus_data["timestamp"] = prometheus_data["timestamp"].apply(lambda x: x * 1000)
    prometheus_data["timestamp"] = pd.to_datetime(prometheus_data["timestamp"], unit='ms')
    return prometheus_data


def filter_data():
    prometheus = get_prometheus_data()
    filtered_data = prometheus[prometheus.namespace.eq(os.getenv("NAMESPACE"))]
    #filtered_data = prometheus[prometheus.app.eq(os.getenv("APPNAME"))]
    filtered_data = pd.pivot_table(filtered_data, index=["timestamp"], columns=["__name__"], values="value")
    filtered_data["container_cpu_usage_seconds_total"] = filtered_data["container_cpu_usage_seconds_total"].apply(
        lambda x: x * 100)
    filtered_data = filtered_data.rename(columns={"container_cpu_usage_seconds_total": "cpu usage [%]"})
    filtered_data = filtered_data.resample("1s").mean()

    filtered_data_interploated = filtered_data.interpolate()
    # plot
    plt.interactive(True)
    filtered_data_interploated.plot()
    plt.show(block=True)


if __name__ == '__main__':
    filter_data()