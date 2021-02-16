# Copyright (c) 2020 Angelina Horn
from gevent import monkey

monkey.patch_all()
# imports
import datetime as dt
import logging
import os
import time

from dotenv import load_dotenv, set_key

from prometheus_api_client import PrometheusConnect, MetricRangeDataFrame

from locust_loadtest import start_locust

import numpy as np
import pandas as pd
import k8s_tools as k8s

# environment
load_dotenv(override=True)

# init logger
logging.getLogger().setLevel(logging.INFO)


def config_env(**kwargs) -> None:
    """
    Configures the environment file.
    :param kwargs: keys and values to be set.
    :return: None
    """
    arguments = locals()
    env_file = os.path.join(os.getcwd(), ".env")
    for i in arguments["kwargs"].keys():
        key = str(i).upper()
        value = str(arguments["kwargs"][i])
        set_key(dotenv_path=env_file, key_to_set=key, value_to_set=value)


def get_prometheus_data(folder: str, iteration: int) -> None:
    """
    Exports metric data from prometheus to a csv file.
    :param folder: save folder
    :param iteration: number of current iteration
    :return: None
    """
    # metrics to export
    resource_metrics = [
        "kube_pod_container_resource_requests_memory_bytes",
        "kube_pod_container_resource_limits_memory_bytes",
        "kube_pod_container_resource_limits_cpu_cores",
        "kube_pod_container_resource_requests_cpu_cores",
        "container_cpu_cfs_throttled_seconds_total",
        "kube_deployment_spec_replicas"
    ]
    custom_metrics = ["container_memory_usage_bytes", "container_cpu_usage_seconds_total"]
    network_metrics = ["response_latency_ms_sum", "response_latency_ms_count"]
    # get resource metric data resources
    resource_metrics_data = get_prometheus_metric(metric_name=resource_metrics[0], mode="RESOURCES", custom=False)
    for x in range(1, len(resource_metrics)):
        resource_metrics_data = resource_metrics_data + get_prometheus_metric(metric_name=resource_metrics[x],
                                                                              mode="RESOURCES", custom=False)
    # get custom resource metric data resources
    custom_metrics_data = get_prometheus_metric(metric_name=custom_metrics[0],
                                                mode="RESOURCES", custom=True) + get_prometheus_metric(
        metric_name=custom_metrics[1],
        mode="RESOURCES", custom=True)
    # get network metric data
    network_metrics_data = get_prometheus_metric(metric_name=network_metrics[0],
                                                 mode="NETWORK", custom=False) + get_prometheus_metric(
        metric_name=network_metrics[1],
        mode="NETWORK", custom=False)
    # convert to dataframe
    metrics_data = resource_metrics_data + network_metrics_data
    metric_df = MetricRangeDataFrame(metrics_data)
    custom_metrics_df = MetricRangeDataFrame(custom_metrics_data)
    # write to csv file
    metric_df.to_csv(rf"{folder}\metrics_{iteration}.csv")
    custom_metrics_df.to_csv(rf"{folder}\custom_metrics_{iteration}.csv")


def get_prometheus_metric(metric_name: str, mode: str, custom: bool) -> list:
    """
    Gets a given metric from prometheus in a given timeframe.
    :param custom: if custom query should be used
    :param mode: which prometheus to use
    :param metric_name: name of the metric
    :return: metric
    """
    # init
    prom = PrometheusConnect(url=os.getenv(f'PROMETHEUS_{mode}_HOST'), disable_ssl=True)
    start_time = (dt.datetime.now() - dt.timedelta(hours=int(os.getenv("HH")), minutes=int(os.getenv("MM"))))
    # get data
    if custom:
        metric_data = prom.custom_query_range(
            query=f"rate({metric_name}" + "{" + f"namespace='{os.getenv('NAMESPACE')}'" + "}[1m])",
            start_time=start_time,
            end_time=dt.datetime.now(),
            step="61")
    else:
        metric_data = prom.get_metric_range_data(
            metric_name=metric_name,
            start_time=start_time,
            end_time=dt.datetime.now(),
        )
    return metric_data


def benchmark(name: str, users: int, spawn_rate: int, cpu_limit: int,
              memory_limit: int, pods_limit: int) -> None:
    """
    Benchmark methods.
    :param cpu_limit: cpu limit
    :param pods_limit: pods limit
    :param memory_limit: memory limit
    :param name: name of ms
    :param users: number of users
    :param spawn_rate: spawn rate
    :return: None
    """
    # init date
    date = dt.datetime.now()
    date = date.strftime("%Y%m%d-%H%M%S")
    # create folder
    folder_path = os.path.join(os.getcwd(), "data", "raw", date)
    os.mkdir(folder_path)
    # config
    set_key(dotenv_path=os.path.join(os.getcwd(), ".env"), key_to_set="LAST_DATA", value_to_set=date)
    k8s.set_prometheus_info()
    config_env(app_name=name,
               host=os.getenv("HOST"),
               node_port=k8s.k8s_get_app_port(),
               date=date,
               users=users,
               spawn_rate=spawn_rate,
               cpu_limit=cpu_limit,
               memory_limit=memory_limit,
               pods_limit=pods_limit
               )
    # read new environment data
    load_dotenv(override=True)
    # get variation
    variation = parameter_variation(cpu_limit=cpu_limit,
                                    memory_limit=memory_limit,
                                    pods_limit=pods_limit)
    c_max, m_max, p_max = variation.shape
    iteration = 1
    # benchmark
    logging.info("Starting Benchmark.")
    for c in range(0, c_max):
        for m in range(0, m_max):
            for p in range(0, p_max):
                v = variation[c, m, p]
                logging.info(
                    f"Iteration {iteration}/{c_max * m_max * p_max} run{j}/{5} - cpu: {v[0]}m memory: {v[1]}Mi #pods:{v[2]}")
                k8s.k8s_update_all_deployments_in_namespace(cpu_limit=int(v[0]), memory_limit=int(v[1]),
                                                            number_of_replicas=int(v[2]))
                time.sleep(int(os.getenv("SLEEP_TIME")))
                start_locust(iteration=iteration, folder=folder_path)
                get_prometheus_data(folder=folder_path, iteration=iteration)
                iteration = iteration + 1
    logging.info("Finished Benchmark.")


def parameter_variation(cpu_limit: int, memory_limit: int, pods_limit: int) -> np.array:
    """
    Calculates a matrix mit all combination of the parameters.
    :return: parameter variation matrix
    """
    # init parameters: (start, end, step)
    cpu = np.arange(200, cpu_limit, 200)
    memory = np.arange(200, memory_limit, 200)
    pods = np.arange(1, pods_limit, 1)
    iterations = np.arange(1, (cpu.size * memory.size * pods.size) + 1, 1).tolist()
    # init dataframe
    df = pd.DataFrame(index=iterations, columns=["CPU", "Memory", "Pods"])
    csv_path = os.path.join(os.getcwd(), "data", "raw", os.getenv("LAST_DATA"), "variation_matrix.csv")
    # init matrix
    variation_matrix = np.zeros((cpu.size, memory.size, pods.size),
                                dtype=[('cpu', np.int32), ('memory', np.int32), ('pods', np.int32)])
    # fill matrix
    i = 1
    for c in range(0, cpu.size):
        for m in range(0, memory.size):
            for p in range(0, pods.size):
                variation_matrix[c, m, p] = (cpu[c], memory[m], pods[p])
                # fill dataframe
                df.at[i, 'CPU'] = cpu[c]
                df.at[i, 'Memory'] = memory[m]
                df.at[i, 'Pods'] = pods[p]
                i = i + 1
    # save dataframe to csv
    df.to_csv(csv_path)
    return variation_matrix


if __name__ == '__main__':
    for j in range(0, 5):
        benchmark(name="robot-shop", users=100, spawn_rate=50, cpu_limit=1200, memory_limit=1200, pods_limit=6)
