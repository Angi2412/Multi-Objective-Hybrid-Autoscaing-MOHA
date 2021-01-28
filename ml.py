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
    # config
    load_dotenv(override=True)
    prometheus_data = None
    i = 0
    # check if folder exists
    data_path = os.path.join(os.getcwd(), "data", "raw", os.getenv("LAST_DATA"))
    if os.path.exists(data_path):
        # search for prometheus metric files
        for (dirpath, dirnames, filenames) in os.walk(data_path):
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
    filtered_data_interpolated = filtered_data.interpolate()
    filtered_data_interpolated["datapoints"] = filtered_data_interpolated.groupby(['Iteration']).cumcount()+1
    return filtered_data_interpolated


def plot_filtered_data(metric: str):
    data = filter_data()

    # pass custom palette:
    sns.lineplot(x=data.datapoints,
                 y=metric,
                 hue='Iteration',
                 data=data)
    plt.show()


if __name__ == '__main__':
    plot_filtered_data("container_cpu_usage_seconds_total")
