# Copyright (c) 2020 Angelina Horn
import subprocess

from gevent import monkey

from data.loadtest.locust.loadshapes import DoubleWave

monkey.patch_all()

import gevent
from locust.env import Environment
from locust.stats import stats_history, StatsCSVFileWriter
from data.loadtest.locust.teastore_fast import UserBehavior
# imports
import datetime as dt
import logging
import os
import time
import json

from dotenv import load_dotenv, set_key

from prometheus_api_client import PrometheusConnect, MetricRangeDataFrame, MetricSnapshotDataFrame

import numpy as np
import pandas as pd
import k8s_tools as k8s
import requests

# environment
load_dotenv(override=True)

# init logger
p = logging.getLogger(__name__)
p.setLevel(logging.INFO)


def config_env(**kwargs) -> None:
    """Configures the environment file.

    Args:
      kwargs: keys and values to be set.
      **kwargs: 

    Returns:
      None

    """
    arguments = locals()
    env_file = os.path.join(os.getcwd(), ".env")
    for i in arguments["kwargs"].keys():
        key = str(i).upper()
        value = str(arguments["kwargs"][i])
        set_key(dotenv_path=env_file, key_to_set=key, value_to_set=value)


def get_prometheus_data(folder: str, iteration: int, hh: int, mm: int) -> None:
    """Exports metric data from prometheus to a csv file.

    Args:
      mm: minutes
      hh: hours
      folder: save folder
      iteration: number of current iteration
      folder: str: 
      iteration: int: 
      hh: int: 
      mm: int: 

    Returns:
      None

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
    # get resource metric data resources
    resource_metrics_data = get_prometheus_metric(metric_name=resource_metrics[0], mode="RESOURCES", custom=False,
                                                  hh=hh, mm=mm)
    for x in range(1, len(resource_metrics)):
        resource_metrics_data = resource_metrics_data + get_prometheus_metric(metric_name=resource_metrics[x],
                                                                              mode="RESOURCES", custom=False, hh=hh,
                                                                              mm=mm)
    # get custom resource metric data resources
    # memory usage
    custom_memory = get_prometheus_metric(metric_name="memory", mode="RESOURCES", custom=True, hh=hh, mm=mm)
    custom_memory = MetricRangeDataFrame(custom_memory)
    custom_memory.insert(0, 'metric', "memory")
    # cpu usage
    custom_cpu = get_prometheus_metric(metric_name="cpu", mode="RESOURCES", custom=True, hh=hh, mm=mm)
    custom_cpu = MetricRangeDataFrame(custom_cpu)
    custom_cpu.insert(0, 'metric', "cpu")
    # rps
    custom_rps = get_prometheus_metric(metric_name="rps", mode="NETWORK", custom=True, hh=hh, mm=mm)
    custom_rps = MetricRangeDataFrame(custom_rps)
    custom_rps.insert(0, 'metric', "rps")
    # average response time
    custom_latency = get_prometheus_metric(metric_name="response_time", mode="NETWORK", custom=True, hh=hh, mm=mm)
    custom_latency = MetricRangeDataFrame(custom_latency)
    custom_latency.insert(0, 'metric', "response_time")
    # median response time
    custom_med_latency = get_prometheus_metric(metric_name="median_latency", mode="NETWORK", custom=True, hh=hh, mm=mm)
    custom_med_latency = MetricRangeDataFrame(custom_med_latency)
    custom_med_latency.insert(0, 'metric', "median_latency")
    # 95th percentile latency
    custom_95_latency = get_prometheus_metric(metric_name="latency95", mode="NETWORK", custom=True, hh=hh, mm=mm)
    custom_95_latency = MetricRangeDataFrame(custom_95_latency)
    custom_95_latency.insert(0, 'metric', "latency95")
    # convert to dataframe
    metrics_data = resource_metrics_data
    metric_df = MetricRangeDataFrame(metrics_data)
    custom_metrics_df = pd.concat(
        [custom_cpu, custom_memory, custom_rps, custom_latency, custom_med_latency, custom_95_latency])
    # write to csv file
    metric_df.to_csv(rf"{folder}\metrics_{iteration}.csv")
    custom_metrics_df.to_csv(rf"{folder}\custom_metrics_{iteration}.csv")


def get_status(pod: str) -> (list, list):
    """Returns the current parameter and target status.

    Args:
      pod: name of the pod
      pod: str) -> (list: 
      list: 

    Returns:
      current parameter and target status

    """
    # init
    prom_res = PrometheusConnect(url=os.getenv(f'PROMETHEUS_RESOURCES_HOST'), disable_ssl=True)
    prom_net = PrometheusConnect(url=os.getenv(f'PROMETHEUS_NETWORK_HOST'), disable_ssl=True)
    # custom queries
    cpu_usage_query = '(sum(rate(container_cpu_usage_seconds_total{namespace="teastore", container!=""}[1m])) by (pod, ' \
                      'container) /sum(container_spec_cpu_quota{namespace="teastore", ' \
                      'container!=""}/container_spec_cpu_period{namespace="teastore", container!=""}) by (pod, ' \
                      'container) )*100'
    memory_usage_query = 'round(max by (pod)(max_over_time(container_memory_usage_bytes{namespace="teastore",pod=~".*" }[' \
                         '1m]))/ on (pod) (max by (pod) (kube_pod_container_resource_limits)) * 100,0.01)'
    rps_query = 'sum(irate(request_total{deployment="teastore-webui", direction="inbound"}[1m]))'
    response_time = 'sum(response_latency_ms_sum{deployment="teastore-webui", direction="inbound"})/sum(' \
                    'response_latency_ms_count{deployment="teastore-webui", direction="inbound"})'
    # target metrics
    cpu_usage = 0.0
    memory_usage = 0.0
    latency = 0.0
    # get cpu
    cpu_usage_data = MetricSnapshotDataFrame(prom_res.custom_query(cpu_usage_query))
    try:
        if 'pod' in cpu_usage_data.columns:
            cpu_usage_data["pod"] = cpu_usage_data["pod"].str.split("-", n=2).str[1]
            cpu_usage = cpu_usage_data.loc[(cpu_usage_data['pod'] == pod)].at[0, 'value']
        elif not cpu_usage_data.empty:
            cpu_usage = cpu_usage_data.at[0, 'value']
    except Exception as err:
        logging.error(f"Error while gathering cpu usage: {err}")
        print(cpu_usage_data)
    # get memory
    try:
        memory_usage_data = MetricSnapshotDataFrame(prom_res.custom_query(memory_usage_query))
        if 'pod' in memory_usage_data.columns:
            memory_usage_data["pod"] = memory_usage_data["pod"].str.split("-", n=2).str[1]
            memory_usage = memory_usage_data.loc[(memory_usage_data['pod'] == pod)].at[0, 'value']
        else:
            memory_usage = memory_usage_data.at[0, 'value']
    except Exception as err:
        logging.error(f"Error while gathering memory usage: {err}")
    # get average response time
    try:
        latency_data = MetricSnapshotDataFrame(prom_net.custom_query(response_time))
        if not latency_data.empty:
            latency = latency_data.at[0, 'value']
        else:
            raise Exception
    except Exception as err:
        logging.error(f"Error while gathering latency: {err}")
    targets = [float(cpu_usage), float(memory_usage), float(latency)]
    # parameter metrics
    # cpu
    cpu_limit_data = MetricSnapshotDataFrame(
        prom_res.get_current_metric_value("kube_pod_container_resource_limits_cpu_cores"))
    # memory
    memory_limit_data = MetricSnapshotDataFrame(
        prom_res.get_current_metric_value("kube_pod_container_resource_limits_memory_bytes"))
    # number of pods
    number_of_pods_data = MetricSnapshotDataFrame(prom_res.get_current_metric_value("kube_deployment_spec_replicas"))
    # rps
    rps_data = MetricSnapshotDataFrame(prom_net.custom_query(rps_query))
    # filter
    cpu_limit = 0
    memory_limit = 0
    number_of_pods = 0
    rps = 0.0
    try:
        cpu_limit_data["pod"] = cpu_limit_data["pod"].str.split("-", n=2).str[1]
        memory_limit_data["pod"] = memory_limit_data["pod"].str.split("-", n=2).str[1]
        number_of_pods_data["pod"] = number_of_pods_data["pod"].str.split("-", n=2).str[1]
        cpu_limit = cpu_limit_data.loc[(cpu_limit_data['pod'] == pod)]['value'].iloc[0]
        memory_limit = memory_limit_data.loc[(memory_limit_data['pod'] == pod)]['value'].iloc[0]
        number_of_pods = \
            number_of_pods_data.loc[(number_of_pods_data['deployment'] == f"teastore-{pod}")]['value'].iloc[0]
        rps = rps_data.at[0, 'value']
    except Exception as err:
        logging.error(f"Error while gathering parameter: {err}")
    parameters = [int(float(cpu_limit) * 1000), int(float(memory_limit) / 1048576), int(number_of_pods), float(rps)]
    return parameters, targets


def get_prometheus_metric(metric_name: str, mode: str, custom: bool, hh: int, mm: int) -> list:
    """Gets a given metric from prometheus in a given timeframe.

    Args:
      mm: minutes
      hh: hours
      custom: if custom query should be used
      mode: which prometheus to use
      metric_name: name of the metric
      metric_name: str: 
      mode: str: 
      custom: bool: 
      hh: int: 
      mm: int: 

    Returns:
      metric

    """
    # init
    prom = PrometheusConnect(url=os.getenv(f'PROMETHEUS_{mode}_HOST'), disable_ssl=True)
    start_time = (dt.datetime.now() - dt.timedelta(hours=hh, minutes=mm))
    # custom queries
    cpu_usage = '(sum(rate(container_cpu_usage_seconds_total{namespace="teastore", container!=""}[1m])) by (pod, ' \
                'container) /sum(container_spec_cpu_quota{namespace="teastore", ' \
                'container!=""}/container_spec_cpu_period{namespace="teastore", container!=""}) by (pod, ' \
                'container) )*100'
    memory_usage = 'round(max by (pod)(max_over_time(container_memory_usage_bytes{namespace="teastore",pod=~".*" }[' \
                   '1m]))/ on (pod) (max by (pod) (kube_pod_container_resource_limits)) * 100,0.01)'
    rps = 'sum(irate(request_total{deployment="teastore-webui", direction="inbound"}[1m]))'
    response_time = 'sum(response_latency_ms_sum{deployment="teastore-webui", direction="inbound"})/sum(' \
                    'response_latency_ms_count{deployment="teastore-webui", direction="inbound"})'
    median_latency = 'histogram_quantile(0.5, sum(irate(response_latency_ms_bucket{deployment="teastore-webui", ' \
                     'direction="inbound"}[1m])) by (le, replicaset)) '
    latency95 = 'histogram_quantile(0.95, sum(irate(response_latency_ms_bucket{deployment="teastore-webui", ' \
                'direction="inbound"}[1m])) by (le, replicaset)) '
    query = None
    # get data
    if custom:
        if metric_name == "cpu":
            query = cpu_usage
        elif metric_name == "memory":
            query = memory_usage
        elif metric_name == "rps":
            query = rps
        elif metric_name == "response_time":
            query = response_time
        elif metric_name == "median_latency":
            query = median_latency
        elif metric_name == "latency95":
            query = latency95
        else:
            logging.error("Accepts cpu, memory or rps but received " + metric_name)
        metric_data = prom.custom_query_range(
            query=query,
            start_time=start_time,
            end_time=dt.datetime.now(),
            step="10")
    else:
        metric_data = prom.get_metric_range_data(
            metric_name=metric_name,
            start_time=start_time,
            end_time=dt.datetime.now(),
        )
    return metric_data


def evaluation(load: int, spawn_rate: int, hh: int, mm: int, load_testing: str) -> None:
    """Start a evaluation run and gathers its metrics.

    Args:
      load: maximum number of users/rps
      spawn_rate: only used with locust
      hh: hours
      mm: minutes
      load_testing: Locust or JMeter
      load: int: 
      spawn_rate: int: 
      hh: int: 
      mm: int: 
      load_testing: str: 

    Returns:
      none

    """
    # init date
    date = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    # create folder
    folder_path = os.path.join(os.getcwd(), "data", "raw", f"{date}_eval")
    os.mkdir(folder_path)
    # create deployments
    k8s.k8s_create_teastore()
    k8s.deploy_autoscaler_docker()
    # config
    k8s.set_prometheus_info()
    config_env(
        host=os.getenv("HOST"),
        node_port=k8s.k8s_get_app_port(),
        date=date,
        load=load,
        spawn_rate=spawn_rate,
        HH=hh,
        MM=mm
    )
    # evaluation
    logging.info("Starting Evaluation.")
    logging.info("Start Locust.")
    if load_testing == "Locust":
        start_locust(iteration=0, folder=folder_path, history=True, custom_shape=True, users=load,
                     spawn_rate=spawn_rate, hh=hh, mm=mm)
    elif load_testing == "JMeter":
        start_jmeter(0, date, False, load)
    # get prometheus data
    time.sleep(30)
    get_prometheus_data(folder=folder_path, iteration=0, hh=hh, mm=mm)
    # clean up
    k8s.delete_autoscaler_docker()
    k8s.k8s_delete_namespace()
    logging.info("Finished Benchmark.")


def benchmark(name: str, load: list, spawn_rate: int, expressions: int,
              step: int, run: int, run_max: int, custom_shape: bool, history: bool,
              sample: bool, locust: bool) -> None:
    """Starts the benchmark.

    Args:
      history: enable locust history
      custom_shape: if using custom load shape
      run_max: number of runs
      run: current run
      expressions: number of expressions per parameter
      step: size of step
      name: name of ms
      load: number of users or rps
      spawn_rate: spawn rate
      sample: enable sample run
      locust: use locust or jmeter
      name: str: 
      load: list: 
      spawn_rate: int: 
      expressions: int: 
      step: int: 
      run: int: 
      run_max: int: 
      custom_shape: bool: 
      history: bool: 
      sample: bool: 
      locust: bool: 

    Returns:
      None

    """
    # init date
    # read new environment data
    load_dotenv(override=True)
    date = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    # create folder
    folder_path = os.path.join(os.getcwd(), "data", "raw", date)
    os.mkdir(folder_path)
    k8s.k8s_create_teastore()
    # config
    set_key(dotenv_path=os.path.join(os.getcwd(), ".env"), key_to_set="LAST_DATA", value_to_set=date)
    k8s.set_prometheus_info()
    config_env(app_name=name,
               host=os.getenv("HOST"),
               node_port=k8s.k8s_get_app_port(),
               date=date,
               load=load,
               spawn_rate=spawn_rate
               )
    iteration = 1
    scale_only = "webui"
    # get variation
    variations = parameter_variation_namespace(expressions, step, sample, load)
    c_max, m_max, p_max, l_max = variations[os.getenv("UI")].shape

    # benchmark
    logging.info("Starting Benchmark.")
    for c in range(0, c_max):
        for m in range(0, m_max):
            for p in range(0, p_max):
                for l in range(0, l_max):
                    logging.info(
                        f"Iteration: {iteration}/{c_max * m_max * p_max} run: {run}/ {run_max}")
                    # for every pod in deployment
                    for pod in variations.keys():
                        # check that pod is scalable
                        if scale_only in pod:
                            # get parameter variation
                            v = variations[pod][c, m, p, l]
                            # check if variation is empty
                            if v[0] == 0 or v[1] == 0 or v[2] == 0:
                                break
                            logging.info(f"{pod}: cpu: {int(v[0])}m - memory: {int(v[1])}Mi - # pods: {int(v[2])}")
                            # update resources of pod
                            k8s.k8s_update_deployment(deployment_name=pod, cpu_limit=int(v[0]),
                                                      memory_limit=int(v[1]),
                                                      number_of_replicas=int(v[2]), replace=True)
                            # wait for deployment
                            time.sleep(90)
                            while not k8s.check_teastore_health():
                                time.sleep(10)
                    # start load test
                    logging.info("Start Load.")
                    if locust:
                        start_locust(iteration=iteration, folder=folder_path, history=history,
                                     custom_shape=custom_shape, users=l, spawn_rate=spawn_rate, hh=int(os.getenv("HH")),
                                     mm=int(os.getenv("MM")))
                    else:
                        start_jmeter(iteration, date, True, l)
                    # get prometheus data
                    get_prometheus_data(folder=folder_path, iteration=iteration, hh=int(os.getenv("HH")),
                                        mm=int(os.getenv("MM")))
                    iteration = iteration + 1
    k8s.k8s_delete_namespace()
    logging.info("Finished Benchmark.")


def parameter_variation_namespace(expressions: int, step: int, sample: bool, load: list) -> dict:
    """Generates the parameter variation matrix for every deployment in a namespace with given values.

    Args:
      load: load
      expressions: number of expressions
      step: size of step
      sample: enable sample run
      expressions: int: 
      step: int: 
      sample: bool: 
      load: list: 

    Returns:
      dict of parameter variation matrices

    """
    resource_requests = k8s.get_resource_requests()
    variation = dict()
    for p in resource_requests.keys():
        if p == os.getenv("SCALE_POD"):
            logging.debug("Pod: " + p)
            # cpu
            p_cpu_request = int(resource_requests[p]["cpu"].split("m")[0])
            p_cpu_limit = p_cpu_request + (expressions * step)
            logging.debug(f"cpu request: {p_cpu_request}m - cpu limit: {p_cpu_limit}m")
            # memory
            p_memory_request = int(resource_requests[p]["memory"].split("Mi")[0])
            p_memory_limit = p_memory_request + (expressions * step)
            logging.debug(f"memory request: {p_memory_request}Mi - memory limit: {p_memory_limit}Mi")
            p_pod_limit = expressions
            # parameter variation matrix
            variation[p] = parameter_variation(p, p_cpu_request, p_cpu_limit, p_memory_request,
                                               p_memory_limit, 1, p_pod_limit, step, invert=False, sample=sample,
                                               save=True, load=load)
    return variation


def parameter_variation(pod: str, cpu_request: int, cpu_limit: int, memory_request: int, memory_limit: int,
                        pods_request: int,
                        pods_limit: int, step: int, invert: bool, sample: bool, save: bool, load: list) -> np.array:
    """Calculates a matrix mit all combination of the parameters.
    :return: parameter variation matrix

    Args:
      pod: str: 
      cpu_request: int: 
      cpu_limit: int: 
      memory_request: int: 
      memory_limit: int: 
      pods_request: int: 
      pods_limit: int: 
      step: int: 
      invert: bool: 
      sample: bool: 
      save: bool: 
      load: list: 

    Returns:

    """
    # init parameters: (start, end, step)
    cpu = np.arange(cpu_request, cpu_limit, step, np.int32)
    memory = np.arange(memory_request, memory_limit, step, np.int32)
    pods = np.arange(pods_request, pods_limit + 1, 1, np.int32)
    load = np.array(load, np.int32)
    if invert:
        cpu = np.flip(cpu)
        memory = np.flip(memory)
        pods = np.flip(pods)
    iterations = np.arange(1, (cpu.size * memory.size * pods.size) + 1, 1).tolist()
    if sample:
        cpu = cpu[(cpu == cpu.min()) | (cpu == np.median(cpu)) | (cpu == cpu.max())]
        memory = memory[
            (memory == memory.min()) | (memory == np.median(memory)) | (memory == memory.max())]
    # init dataframe
    df = pd.DataFrame(index=iterations, columns=["CPU", "Memory", "Pods"])
    csv_path = os.path.join(os.getcwd(), "data", "raw", os.getenv("LAST_DATA"), f"{pod}_variation.csv")
    # init matrix
    variation_matrix = np.zeros((cpu.size, memory.size, pods.size, load.size),
                                dtype=[('cpu', np.int32), ('memory', np.int32), ('pods', np.int32), ('load', np.int32)])
    # fill matrix
    i = 1
    for c in range(0, cpu.size):
        for m in range(0, memory.size):
            for p in range(0, pods.size):
                for l in range(0, load.size):
                    if sample:
                        if m != c:
                            print("here")
                            break
                    variation_matrix[c, m, p] = (cpu[c], memory[m], pods[p], load[l])
                    # fill dataframe
                    df.at[i, 'CPU'] = cpu[c]
                    df.at[i, 'Memory'] = memory[m]
                    df.at[i, 'Pods'] = pods[p]
                    df.at[i, 'RPS'] = load[l]
                    i = i + 1
    logging.debug(df.head())
    if save:
        # save dataframe to csv
        if not os.path.exists(csv_path):
            df.to_csv(csv_path)
    return variation_matrix


def parameter_variation_array(cpu_limits: list, memory_limits: list, pod_limits: list, rps: float) -> np.array:
    """Creates a parameter variation matrix given discrete values.

    Args:
      cpu_limits: list of cpu limits
      memory_limits: list of memory limits
      pod_limits: list of pod limits
      rps: current load
      cpu_limits: list: 
      memory_limits: list: 
      pod_limits: list: 
      rps: float: 

    Returns:
      parameter variation matrix

    """
    cpu = np.array(cpu_limits, dtype=np.int32)
    memory = np.array(memory_limits, dtype=np.int32)
    pods = np.array(pod_limits, dtype=np.int32)
    variation_matrix = np.zeros((cpu.size, memory.size, pods.size, 1),
                                dtype=[('cpu', np.int32), ('memory', np.int32), ('pods', np.int32),
                                       ('load', np.float64)])
    for c in range(0, cpu.size):
        for m in range(0, memory.size):
            for p in range(0, pods.size):
                variation_matrix[c, m, p] = (cpu[c], memory[m], pods[p], rps)
    return variation_matrix


def start_locust(iteration: int, folder: str, history: bool, custom_shape: bool, users: int, spawn_rate: int, hh: int,
                 mm: int) -> None:
    """Start a locust load test.

    Args:
      spawn_rate: user spawn rate
      users: number of users
      custom_shape: use custom load shape
      iteration: number of current iteration
      folder: name of folder
      history: enables stats
      hh: duration hours
      mm: duration minutes
      iteration: int: 
      folder: str: 
      history: bool: 
      custom_shape: bool: 
      users: int: 
      spawn_rate: int: 
      hh: int: 
      mm: int: 

    Returns:
      None

    """
    load_dotenv(override=True)
    # setup Environment and Runner

    env = Environment(user_classes=[UserBehavior], shape_class=DoubleWave,
                      host=f"http://{os.getenv('HOST')}:{os.getenv('NODE_PORT')}/{os.getenv('ROUTE')}")
    env.create_local_runner()
    # CSV writer
    stats_path = os.path.join(folder, f"locust_{iteration}")
    if history:
        csv_writer = StatsCSVFileWriter(
            environment=env,
            base_filepath=stats_path,
            full_history=True,
            percentiles_to_report=[90.0, 50.0]
        )
        # start a greenlet that save current stats to history
        gevent.spawn(stats_history, env.runner)
        # spawn csv writer
        gevent.spawn(csv_writer)
    # start the test
    if custom_shape:
        env.runner.start_shape()
    else:
        env.runner.start(user_count=users, spawn_rate=spawn_rate)
    # stop the runner in a given time
    time_in_seconds = ((hh * 60 * 60) + mm * 60)
    gevent.spawn_later(time_in_seconds, lambda: env.runner.quit())
    # wait for the greenlets
    env.runner.greenlet.join()


def get_persistence_data() -> None:
    """Gets persistence data from the TeaStore.
    :return: None

    Args:

    Returns:

    """
    base_path = os.path.join(os.getcwd(), "data", "loadtest")
    persistence_url = "http://localhost:30090/tools.descartes.teastore.persistence/rest"
    # get category ids
    categories_request = requests.get(persistence_url + "/categories").json()
    tmp_categories = list()
    for c in categories_request:
        tmp_categories.append(c["id"])
    with open(os.path.join(base_path, "categories.json"), 'x') as outfile:
        json.dump(tmp_categories, outfile)
    # get product ids
    products_request = requests.get(persistence_url + "/products").json()
    tmp_products = list()
    for p in products_request:
        tmp_products.append(p["id"])
    with open(os.path.join(base_path, "products.json"), 'x') as outfile:
        json.dump(tmp_products, outfile)
    # get users
    users = requests.get(persistence_url + "/users").json()
    with open(os.path.join(base_path, "users.json"), 'x') as outfile:
        json.dump(users, outfile)


def start(name: str, load: list, spawn_rate: int, expressions: int, step: int, runs: int,
          custom_shape: bool, history: bool, sample: bool, locust: bool) -> None:
    """Starts the generation of a dataset.

    Args:
      name: application name
      load: maximum load
      spawn_rate: only used with Locust
      expressions: number of expressions per parameter
      step: step size
      runs: number of stability runs
      custom_shape: if custom shape should be used
      history: only used with locust
      sample: if a sample run should be executed
      locust: if Locust is used
      name: str: 
      load: list: 
      spawn_rate: int: 
      expressions: int: 
      step: int: 
      runs: int: 
      custom_shape: bool: 
      history: bool: 
      sample: bool: 
      locust: bool: 

    Returns:
      None

    """
    date = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    set_key(dotenv_path=os.path.join(os.getcwd(), ".env"), key_to_set="FIRST_DATA", value_to_set=date)
    for i in range(1, runs + 1):
        benchmark(name, load, spawn_rate, expressions, step, i, runs, custom_shape, history, sample,
                  locust)


def start_jmeter(iteration: int, date: str, evaluation: bool, rps: int):
    """Stats a jMeter run.

    Args:
      iteration: current iteration
      date: current date
      evaluation: if evaluation is used
      rps: requests per second
      iteration: int: 
      date: str: 
      evaluation: bool: 
      rps: int: 

    Returns:

    """
    work_directory = os.getcwd()
    jmeter_path = os.path.join(os.getcwd(), "data", "loadtest", "jmeter", "bin")
    os.chdir(jmeter_path)
    if not evaluation:
        cmd = ["java", "-jar", "ApacheJMeter.jar", "-t", "teastore_browse_rps.jmx", "-Jhostname", os.getenv("HOST"),
               "-Jport", os.getenv('NODE_PORT'), "-l", f"{date}_{iteration}.log",
               '-Jload_profile', f'const({rps},{int(os.getenv("MM")) * 60}s)', "-n"]
    else:
        # f'-Jload_profile=step(2,{rps},2,180s) const({rps},240s) step({rps},2,2,180s)'
        cmd = ["java", "-jar", "ApacheJMeter.jar", "-t", "teastore_browse_rps.jmx", "-Jhostname", os.getenv("HOST"),
               "-Jport", os.getenv('NODE_PORT'), "-l", f"{date}_{iteration}.log",
               "-Jjmeterengine.force.system.exit=true", "-n"]
    logging.info(cmd)
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True, bufsize=1, universal_newlines=True) as p:
        for line in p.stdout:
            print(line, end='')  # process line here
    os.chdir(work_directory)


def change_build(alg: str, hpa: bool, weights: str) -> None:
    """Changes the autoscaler build to the given parameters and builds the docker image.

    Args:
      alg: which estimator to use
      hpa: if only horizontal scaling is enabled
      weights: mcdm weight distribution
      alg: str: 
      hpa: bool: 
      weights: str: 

    Returns:
      None

    """
    # change environment variables
    if alg in ["svr", "neural_network", "linear_b"]:
        set_key(os.path.join(os.getcwd(), "prod.env"), "ALGORITHM", alg)
        set_key(os.path.join(os.getcwd(), ".env"), "ALGORITHM", alg)
    if hpa:
        set_key(os.path.join(os.getcwd(), "prod.env"), "HPA", "True")
        set_key(os.path.join(os.getcwd(), ".env"), "HPA", "True")
    else:
        set_key(os.path.join(os.getcwd(), "prod.env"), "HPA", "False")
        set_key(os.path.join(os.getcwd(), ".env"), "HPA", "False")
    # set weights
    set_key(os.path.join(os.getcwd(), "prod.env"), "WEIGHTS", weights)
    set_key(os.path.join(os.getcwd(), ".env"), "WEIGHTS", weights)
    # build docker image
    k8s.buil_autoscaler_docker()
    logging.info(f"Changed build. alg: {alg} - hpa:{hpa} - w: {weights}")
