# Copyright (c) 2020 Angelina Horn
import datetime
import datetime as dt
import logging
import os
import platform
import subprocess
import time
from pathlib import Path

import docker
from dotenv import load_dotenv, set_key
from kubernetes import client, config
from prometheus_api_client import PrometheusConnect, MetricRangeDataFrame, Metric

# deployment
load_dotenv()
NAMESPACE = os.getenv("NAMESPACE")
DEPLOYMENT_NAME = os.getenv("DEPLOYMENT")
DESIRED_PORT = int(os.getenv("PORT"))
ROUTE = os.getenv("ROUTE")
HOST = os.getenv("HOST")

# init logger
logging.getLogger().setLevel(logging.INFO)


def forward_port(port: int) -> None:
    """
    Forwards port of given pod to a given port.
    :param port: port of pod
    :return: None
    """
    # init
    config.load_kube_config()
    v1 = client.CoreV1Api()
    ret = v1.list_namespaced_pod(watch=False, namespace=NAMESPACE)
    pod_name = DEPLOYMENT_NAME
    # search for pod name
    for i in ret.items:
        if DEPLOYMENT_NAME in i.metadata.name:
            pod_name = i.metadata.name
            break
    # forward port
    if pod_name is not DEPLOYMENT_NAME:
        os.system(f"kubectl port-forward -n {NAMESPACE} {pod_name} {port}:{DESIRED_PORT} &")
        # list_files = subprocess.run(["kubectl", "port-forward","-n", NAMESPACE, pod_name, f"{port}:{DESIRED_PORT}"])
        logging.info("Port was forwarded.")
    else:
        logging.error("Could not find pod name.")


def deploy_to_cluster(name: str, port: int, image: str) -> None:
    """
    Creates a kubernetes deployment and pushes it into the cluster.
    :param name: name of the deployment
    :param port: deployment port
    :param image: docker image
    :return: None
    """
    # init API
    config.load_kube_config()
    apps_v1 = client.AppsV1Api()

    # Configure Pod template container
    container = client.V1Container(
        name=name,
        image=image,
        ports=[client.V1ContainerPort(container_port=port)],
        resources=client.V1ResourceRequirements(
            requests={"cpu": "100m", "memory": "200Mi"},
            limits={"cpu": "500m", "memory": "500Mi"}
        ),
        image_pull_policy="Never"
    )
    # Create and configure a spec section
    template = client.V1PodTemplateSpec(
        metadata=client.V1ObjectMeta(labels={"app": name}),
        spec=client.V1PodSpec(containers=[container]))
    # Create the specification of deployment
    spec = client.V1DeploymentSpec(
        replicas=1,
        template=template,
        selector={'matchLabels': {'app': name}})
    # Instantiate the deployment object
    deployment = client.V1Deployment(
        api_version="apps/v1",
        kind="Deployment",
        metadata=client.V1ObjectMeta(name=DEPLOYMENT_NAME),
        spec=spec)
    # Create deployment
    try:
        api_response = apps_v1.create_namespaced_deployment(
            body=deployment,
            namespace=NAMESPACE)
        logging.info("Deployment created. status='%s'" % str(api_response.status))
    except Exception as e:
        logging.info(f"Error while deployment: {e}")


def delete_deployment(api_instance: client.AppsV1Api) -> None:
    """
    Deletes a given deployment.
    :param api_instance: kubernetes api instance
    :return: None
    """
    # Delete deployment
    api_response = api_instance.delete_namespaced_deployment(
        name=DEPLOYMENT_NAME,
        namespace=NAMESPACE,
        body=client.V1DeleteOptions(
            propagation_policy='Foreground',
            grace_period_seconds=5))
    logging.info("Deployment deleted. status='%s'" % str(api_response.status))


def build_image(name: str, docker_path: str) -> str:
    """
    Builds a docker image from a given path with a given name.
    :param name: desired name of docker image
    :param docker_path: path to docker image
    :return: name of docker image
    """
    # init
    directory_path = None
    image_name = None
    try:
        image_name = f"{name}:latest"
        directory_path = Path(docker_path).parent
        if Path(docker_path).exists() and directory_path.exists():
            pass
        else:
            raise FileNotFoundError
    except FileNotFoundError:
        logging.error("Could not find Dockerfile.")
    try:
        if platform.system() == "Windows":
            docker_client = docker.APIClient(base_url='tcp://localhost:2375')
        else:
            docker_client = docker.APIClient(base_url='unix://var/run/docker.sock')
        # build image
        build = docker_client.build(path=str(directory_path), tag=name, dockerfile=docker_path, rm=True)
        logging.info(list(build))
        logging.info(f"Build image with tag: {image_name}")
        return image_name
    except Exception as e:
        logging.error(f"Could not build image: {e}")


def config_locust(host: str, route: str, port: int, testfile: str) -> None:
    """
    Configures locust by setting environment variables.
    :param host: ms host
    :param route: ms route
    :param port: ms port
    :param testfile: ms testfile
    :return: None
    """
    arguments = locals()
    env_file = os.path.join(os.getcwd(), ".env")
    for i in arguments:
        key = str(i).upper()
        value = str(arguments[i])
        set_key(dotenv_path=env_file, key_to_set=key, value_to_set=value)


def start_locust(users: int, hatch: int, time_hh: int, time_mm: int) -> None:
    """
    Starts the locust load testing.
    :param users: number of simulated users
    :param hatch: number of simulated hatches
    :param time_hh: runtime hours
    :param time_mm: runtime minutes
    :return: None
    """
    response = subprocess.call(["locust", "--csv=locust", "--headless", "-u", users, "-r", hatch, "--run-time", f"{time_hh}h{time_mm}m"])
    logging.info(response)


def get_prometheus_data(time_hh: int, time_mm: int) -> None:
    """
    Exports data from Prometheus.
    :return: None
    """
    # get metric data
    metrics_memory_data = get_prometheus_metric(metric_name="container_memory_usage_bytes", time_hh=time_hh,
                                                time_mm=time_mm) + get_prometheus_metric(
        metric_name="kube_pod_container_resource_limits_memory_bytes", time_hh=time_hh, time_mm=time_mm)
    metrics_cpu_data = get_prometheus_metric(metric_name="container_cpu_usage_seconds_total", time_hh=time_hh,
                                             time_mm=time_mm) + get_prometheus_metric(
        metric_name="kube_pod_container_resource_limits_cpu_cores", time_hh=time_hh, time_mm=time_mm)
    # convert to dataframe
    metric_memory_df = MetricRangeDataFrame(metrics_memory_data)
    metric_cpu_df = MetricRangeDataFrame(metrics_cpu_data)
    # init timestamp
    x = datetime.datetime.now()
    x = x.strftime("%Y%m%d-%H%M%S")
    # write to csv file
    metric_memory_df.to_csv(rf"data\{x}_memory.csv")
    metric_cpu_df.to_csv(rf"data\{x}_csv.csv")


def get_prometheus_metric(metric_name: str, time_hh: int, time_mm: int) -> list:
    """
    Gets a given metric from prometheus in a given timeframe.
    :param metric_name: name of the metric
    :param time_hh: hours
    :param time_mm: minutes
    :return: metric
    """
    prom = PrometheusConnect(url=os.getenv("PROMETHEUS_HOST"), disable_ssl=True)
    metric_data = prom.get_metric_range_data(
        metric_name=metric_name,
        start_time=(dt.datetime.now() - dt.timedelta(hours=time_hh, minutes=time_mm)),
        end_time=dt.datetime.now(),
    )
    return metric_data


def deployment(name: str, port: int, docker_path: str):
    image = build_image(name=name, docker_path=docker_path)
    deploy_to_cluster(name=name, port=port, image=image)
    time.sleep(60)
    forward_port(port=port)


def benchmark():
    # init time
    hh = int(os.getenv("HH"))
    mm = int(os.getenv("MM"))
    sleep_time = (hh * 60 * 60) + (mm * 60)
    # benchmark
    #config_locust(host=HOST, route=ROUTE, port=DESIRED_PORT, testfile="test")
    #start_locust(users=10, hatch=1, time_hh=hh, time_mm=mm)
    #time.sleep(sleep_time)
    get_prometheus_data(time_hh=hh, time_mm=mm)


if __name__ == '__main__':
    """
    Main method for test purposes.
    """
    docker_path = os.path.join(os.getcwd(), "webservice", "Dockerfile")
    testfile_path = os.path.join(os.getcwd(), "data", "test.txt")
    #deployment(name="testmonday", port=5000, docker_path=docker_path)
    #forward_port(port=5000)
    benchmark()
