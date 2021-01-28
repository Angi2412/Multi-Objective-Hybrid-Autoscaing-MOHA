# Copyright (c) 2020 Angelina Horn
from gevent import monkey

monkey.patch_all()
import datetime as dt
import logging
import os
import platform
import time
from pathlib import Path

import docker
from dotenv import load_dotenv, set_key
from gevent import monkey
from kubernetes import client, config
from prometheus_api_client import PrometheusConnect, MetricRangeDataFrame

from benchmark import start_locust

# deployment
monkey.patch_all()
load_dotenv(override=True)
NAMESPACE = os.getenv("NAMESPACE")
DEPLOYMENT_NAME = os.getenv("DEPLOYMENT")
TARGET_PORT = os.getenv("PORT")
ROUTE = os.getenv("ROUTE")
HOST = os.getenv("HOST")
HH = int(os.getenv("HH"))
MM = int(os.getenv("MM"))

# init logger
logging.getLogger().setLevel(logging.INFO)


def k8s_deployment(name: str, port: int, image: str) -> None:
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

    # create namespace
    os.system("kubectl create namespace sandbox")
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
        metadata=client.V1ObjectMeta(labels={"app": name}, annotations={"linkerd.io/inject": "enabled"}),
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
        logging.info(f"Successfully deployed deployment: {api_response.status}.")
    except Exception as e:
        logging.info(f"Error while deployment: {e}")


def k8s_service(app_port: int, app_name: str) -> None:
    """
    Creates and deploys a service to kubernetes.
    :param app_port: port of the app
    :param app_name: name of the app
    :return: None
    """
    # Create Service
    config.load_kube_config()
    api_instance = client.CoreV1Api()
    service = client.V1Service()  # V1Service

    # Creating Meta Data
    metadata = client.V1ObjectMeta()
    metadata.name = "sandbox-service"

    service.metadata = metadata

    # Creating spec
    spec = client.V1ServiceSpec(
        type="NodePort",
        ports=int(TARGET_PORT)
    )

    # Creating Port object
    port = client.V1ServicePort(
        protocol='TCP',
        target_port=int(TARGET_PORT),
        port=app_port
    )

    spec.ports = [port]
    spec.selector = {"app": app_name}

    service.spec = spec
    try:
        api_response = api_instance.create_namespaced_service(
            body=service,
            namespace=NAMESPACE
        )
        logging.info(f"Successfully deployed service: {api_response.status}")
    except Exception as e:
        logging.error("Error while deploying service: %s\n" % e)


def get_k8s_app_port() -> int:
    """
    Returns the node port of the created service.
    :return: node port
    """
    # init
    config.load_kube_config()
    v1 = client.CoreV1Api()

    # iterate through namespaces services
    ret = v1.list_namespaced_service(namespace=NAMESPACE)
    for i in ret.items:
        logging.info(i.metadata.name)
        if os.getenv("SERVICE") in i.metadata.name:
            return int(i.spec.ports[0].node_port)


def build_docker_image(name: str, docker_path: str) -> str:
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
        logging.info(f"Build image with tag: {image_name}")
        return image_name
    except Exception as e:
        logging.error(f"Could not build image: {e}")


def config_locust(appname: str, host: str, route: str, port: int, testfile: str, date: str, users: int,
                  spawn_rate: int) -> None:
    """
    Configures locust by setting environment variables.
    :param appname: name of the microservice
    :param spawn_rate: benchmark spawn rate
    :param users: benchmark users
    :param date: timestamp
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


def get_prometheus_data(mode: str, folder: str, iteration: int) -> None:
    """
    Exports metric data from prometheus to a csv file.
    :param folder: save folder
    :param iteration: number of current iteration
    :param mode: where to save
    :return:
    """
    # metrics to export
    resource_metrics = [
        "container_memory_usage_bytes",
        "kube_pod_container_resource_requests_memory_bytes",
        "kube_pod_container_resource_limits_memory_bytes",
        "container_cpu_usage_seconds_total",
        "kube_pod_container_resource_limits_cpu_cores",
        "kube_pod_container_resource_requests_cpu_cores",
        "container_cpu_cfs_throttled_seconds_total",
        "kube_deployment_spec_replicas"
    ]
    network_metrics = ["request_total", "response_latency_ms_bucket"]
    # get resource metric data resources
    resource_metrics_data = get_prometheus_metric(metric_name=resource_metrics[0], mode="RESOURCES")
    for x in range(1, len(resource_metrics)):
        resource_metrics_data = resource_metrics_data + get_prometheus_metric(metric_name=resource_metrics[x],
                                                                              mode="RESOURCES")
    # get network metric data
    network_metrics_data = get_prometheus_metric(metric_name=network_metrics[0],
                                                 mode="NETWORK") + get_prometheus_metric(metric_name=network_metrics[1],
                                                                                         mode="NETWORK")
    # convert to dataframe
    metrics_data = resource_metrics_data + network_metrics_data
    metric_df = MetricRangeDataFrame(metrics_data)
    # write to csv file
    metric_df.to_csv(rf"{folder}\metrics_{iteration}.csv")


def get_prometheus_metric(metric_name: str, mode: str) -> list:
    """
    Gets a given metric from prometheus in a given timeframe.
    :param mode: which prometheus to use
    :param metric_name: name of the metric
    :return: metric
    """
    # init
    prom = PrometheusConnect(url=os.getenv(f'PROMETHEUS_{mode}_HOST'), disable_ssl=True)
    # get data
    metric_data = prom.get_metric_range_data(
        metric_name=metric_name,
        start_time=(dt.datetime.now() - dt.timedelta(hours=HH, minutes=MM)),
        end_time=dt.datetime.now(),
    )
    return metric_data


def deployment(name: str, port: int, docker_path: str) -> None:
    """
    Deployment methods.
    :param name: ms name
    :param port: ms port
    :param docker_path: ms docker file
    :return: None
    """
    image = build_docker_image(name=name, docker_path=docker_path)
    k8s_deployment(name=name, port=port, image=image)
    k8s_service(app_port=port, app_name=name)


def benchmark(name: str, route: str, testfile: str, users: int, spawn_rate: int, iterations: int) -> None:
    """
    Benchmark methods.
    :param iterations: number of iterations
    :param name: name of ms
    :param route: API route
    :param testfile: test file
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
    config_locust(appname=name,
                  host=HOST,
                  route=route,
                  port=get_k8s_app_port(),
                  testfile=testfile,
                  date=date,
                  users=users,
                  spawn_rate=spawn_rate
                  )
    # benchmark
    for i in range(0, iterations):
        time.sleep(30)
        start_locust(iteration=i, folder=folder_path)
        get_prometheus_data(mode="raw", folder=folder_path, iteration=i)


def set_prometheus_info() -> None:
    """
    Sets corresponding environment variable to NodePort of each Prometheus service instance.
    :return: None
    """
    # init
    config.load_kube_config()
    v1 = client.CoreV1Api()

    # iterate through default namespaces services
    ret_default = v1.list_namespaced_service(namespace="default")
    for service in ret_default.items:
        if "prometheus-kube-prometheus-prometheus" in service.metadata.name:
            # set env variable
            set_key(dotenv_path=os.path.join(os.getcwd(), ".env"), key_to_set="PROMETHEUS_RESSOURCES_HOST",
                    value_to_set=f"http://localhost:{service.spec.ports[0].node_port}")
            logging.info(f"PROMETHEUS_RESSOURCES_HOST: {os.getenv('PROMETHEUS_RESSOURCES_HOST')}")
            break
    # iterate through linkerd namespaces services
    ret_linkerd = v1.list_namespaced_service(namespace="linkerd")
    for service in ret_linkerd.items:
        if "linkerd-prometheus" in service.metadata.name:
            # set env variable
            set_key(dotenv_path=os.path.join(os.getcwd(), ".env"), key_to_set="PROMETHEUS_NETWORK_HOST",
                    value_to_set=f"http://localhost:{service.spec.ports[0].node_port}")
            logging.info(f"PROMETHEUS_NETWORK_HOST: {os.getenv('PROMETHEUS_NETWORK_HOST')}")
            break


if __name__ == '__main__':
    test_docker_path = os.path.join(os.getcwd(), "webservice", "Dockerfile")
    # deployment(name="webserver", port=5000, docker_path=test_docker_path)
    benchmark(name="webserver", route="square", testfile="test", users=1, spawn_rate=10, iterations=5)
