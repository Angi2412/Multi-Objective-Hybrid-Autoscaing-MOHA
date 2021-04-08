# Copyright (c) 2020 Angelina Horn
import subprocess

from gevent import monkey

monkey.patch_all()
# imports
import logging
import os
import platform
import time
from pathlib import Path

import docker
import yaml
from dotenv import load_dotenv, set_key
from kubernetes import client, config, utils
import requests

# environment
load_dotenv(override=True)
kube_config = os.path.join(os.getcwd(), "config")
# init logger
k = logging.getLogger()
k.setLevel(logging.INFO)


def k8s_create_teastore():
    # create deployment
    work_directory = os.getcwd()
    cmd_ns = f"kubectl create namespace teastore"
    logging.debug(subprocess.check_output(cmd_ns, shell=True))
    logging.info("Created namespace.")
    time.sleep(10)
    try:
        teastore_path = os.path.join(os.getcwd(), "k8s", "TeaStore", "examples", "kubernetes")
        os.chdir(teastore_path)
        cmd = f"kubectl create -f teastore-clusterip-linkerd.yaml -n teastore"
        logging.debug(subprocess.check_output(cmd, shell=True))
        logging.info("Deployed teastore.")
        os.chdir(work_directory)
        # wait until deployment is ready
        time.sleep(int(os.getenv("SLEEP_TIME")))
        while not check_teastore_health():
            time.sleep(10)
        logging.info("Teastore is alive.")
    except Exception as err:
        logging.error(f"Error while deploying teastore: {err}")
        k8s_delete_namespace()


def k8s_create_deployment_with_helm() -> None:
    """
    Creates deployment with helm chart.
    :return: None
    """
    # creates namespace
    os.system(f"kubectl create namespace {os.getenv('NAMESPACE')}")
    # deployment with helm
    os.system(f"helm install {os.getenv('APP_NAME')} --namespace {os.getenv('NAMESPACE')} .")


def k8s_create_deployment_from_file(yaml_file: str) -> None:
    """
    Creates a deployment from a yaml file.
    :param yaml_file: name of yaml file
    :return: None
    """
    # init
    config.load_kube_config(config_file=kube_config)
    k8s_client = client.ApiClient()
    # create namespace
    os.system(f"kubectl create namespace {os.getenv('NAMESPACE')}")
    time.sleep(10)
    # create deployment from file
    utils.create_from_yaml(k8s_client, yaml_file, True, os.getenv("NAMESPACE"))


def k8s_update_deployment_from_file(yaml_file: str, cpu_limit: int, memory_limit: int, number_of_replicas: int) -> None:
    """
    Updates at least one deployment from a yaml file.
    :param yaml_file: name of yaml file
    :param cpu_limit: limit of CPU
    :param memory_limit: limit of memory
    :param number_of_replicas: number of replicas
    :return: None
    """
    yaml_file_path = os.path.join(os.getcwd(), "k8s", f"{yaml_file}.yaml")
    deployment_names = list()
    # read yaml file
    with open(os.path.abspath(yaml_file_path)) as f:
        yml_document_all = yaml.safe_load_all(f)
        # for every deployment in yaml_file
        for yml_document in yml_document_all:
            # add deployment to list if not already done
            if yml_document["metadata"]["name"] not in deployment_names:
                deployment_names.append(yml_document["metadata"]["name"])
        # update deployment for each deployment from list
        try:
            for deployment in deployment_names:
                k8s_update_deployment(deployment, cpu_limit, memory_limit, number_of_replicas, replace=True)
        except Exception as err:
            logging.error(f"Error while updating deployment from file: {err}")


def k8s_create_deployment_from_image(name: str, port: int, docker_path: str) -> None:
    """
    Deployment methods.
    :param name: ms name
    :param port: ms port
    :param docker_path: ms docker file
    :return: None
    """
    # create namespace
    os.system(f"kubectl create namespace {os.getenv('NAMESPACE')}")
    image = build_docker_image(name=name, docker_path=docker_path)
    k8s_create_deployment(k8s_deployment(name=name, port=port, image=image))
    time.sleep(60)


def k8s_deployment(name: str, port: int, image: str) -> client.V1Deployment:
    """
    Creates a Kubernetes deployment with given specification.
    :param name: name of the deployment
    :param port: deployment port
    :param image: docker image
    :return: deployment body
    """
    # Configure Pod template container
    container = client.V1Container(
        name=name,
        image=image,
        ports=[client.V1ContainerPort(container_port=port)],
        resources=client.V1ResourceRequirements(
            requests={"cpu": "100m", "memory": "100"},
            limits={"cpu": "200m", "memory": "300Mi"}
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
        selector={'matchLabels': {'app': "teastore"}})
    # Instantiate the deployment object
    deployment = client.V1Deployment(
        api_version="apps/v1",
        kind="Deployment",
        metadata=client.V1ObjectMeta(name=os.getenv("APP_NAME")),
        spec=spec)
    return deployment


def k8s_create_deployment(deployment: client.V1Deployment) -> None:
    """
    Deploys a namespaced deployment to the Kubernetes cluster.
    :param deployment: Deployment body
    :return: None
    """
    # init API
    config.load_kube_config(config_file=kube_config)
    apps_v1 = client.AppsV1Api()
    try:
        api_response = apps_v1.create_namespaced_deployment(
            body=deployment,
            namespace=os.getenv("NAMESPACE"))
        logging.info(f"Successfully deployed deployment: {api_response.status}.")
    except Exception as e:
        logging.info(f"Error while deployment: {e}")


def k8s_update_all_deployments_in_namespace(cpu_limit: int, memory_limit: int,
                                            number_of_replicas: int) -> None:
    """
    Updates all deployments in a given namespace with given values.
    :param cpu_limit: limit of cpu
    :param memory_limit: limit of memory
    :param number_of_replicas: limit of number of replicas
    :return: None
    """
    not_scalable = ["mysql", "mongodb", "redis", "rabbitmq"]
    # init API
    config.load_kube_config(config_file=kube_config)
    apps_v1 = client.AppsV1Api()
    # read deployment
    counter = 0
    ret = apps_v1.list_namespaced_deployment(namespace=os.getenv("NAMESPACE"))
    # for every pod in namespace
    for i in ret.items:
        counter = counter + 1
        logging.info(f"Updating deployments in namespace: {counter}/{len(ret.items) - len(not_scalable)}")
        # update deployment
        if not any(ext in i.metadata.name for ext in not_scalable):
            k8s_update_deployment(deployment_name=i.metadata.name, cpu_limit=cpu_limit, memory_limit=memory_limit,
                                  number_of_replicas=number_of_replicas, replace=True)
        else:
            # keep number of replicas for not scalable pods
            k8s_update_deployment(deployment_name=i.metadata.name, cpu_limit=cpu_limit, memory_limit=memory_limit,
                                  number_of_replicas=1, replace=True)


def k8s_update_deployment(deployment_name: str, cpu_limit: int, memory_limit: int,
                          number_of_replicas: int, replace: bool) -> client.V1Deployment:
    """
    Updates a given deployment with given values for replicas, cpu limit and replicas limit.
    :param replace: if the deployment should be replaced or patched
    :param deployment_name: name of deployment
    :param cpu_limit: new cpu limit
    :param memory_limit: new memory limit
    :param number_of_replicas: new number of replicas
    :return: None
    """
    # init API
    config.load_kube_config(config_file=kube_config)
    apps_v1 = client.AppsV1Api()
    # read deployment
    deployment = apps_v1.read_namespaced_deployment(name=deployment_name, namespace=os.getenv("NAMESPACE"))
    # get resource requests
    resource_requests = get_resource_requests()
    resource_requests = resource_requests[deployment_name]
    # updates cpu and memory limits
    new_resources = client.V1ResourceRequirements(
        requests={"cpu": resource_requests["cpu"], "memory": resource_requests["memory"]},
        limits={"cpu": f"{cpu_limit}m", "memory": f"{memory_limit}Mi"}
    )
    deployment.spec.template.spec.containers[0].resources = new_resources
    # updates number of replicas
    deployment.spec.replicas = number_of_replicas
    if not replace:
        # updates the deployment
        try:
            api_response = apps_v1.patch_namespaced_deployment(
                name=deployment_name,
                namespace=os.getenv("NAMESPACE"),
                body=deployment)
            logging.info(f"Deployment updated of {deployment_name}.")
            logging.debug(f"f Deployment update: {api_response.status}")
        except Exception as err:
            logging.info(f"Error while deployment: {err}")
    else:
        try:
            api_response = apps_v1.replace_namespaced_deployment(
                name=deployment_name,
                namespace=os.getenv("NAMESPACE"),
                body=deployment)
            logging.info(f"Deployment updated of {deployment_name}.")
            logging.debug(f"f Deployment update: {api_response.status}")
        except Exception as err:
            logging.info(f"Error while deployment: {err}")
    return deployment


def check_teastore_health() -> bool:
    try:
        health = requests.get(
            f"http://{os.getenv('HOST')}:{k8s_get_app_port()}/{os.getenv('ROUTE')}/rest/ready/isready")
        if health.ok:
            return bool(health.json())
        else:
            return False
    except requests.exceptions.ConnectionError:
        return False


def k8s_delete_namespace() -> None:
    """
    Deletes namespace.
    :return: None
    """
    # init
    config.load_kube_config(config_file=kube_config)
    v1 = client.CoreV1Api()
    try:
        resp = v1.list_namespace()
        for n in resp.items:
            if os.getenv("NAMESPACE") in n.metadata.name:
                api_response = v1.delete_namespace(os.getenv("NAMESPACE"))
                time.sleep(180)
                logging.info("Deleted namespace " + os.getenv("NAMESPACE"))
                logging.debug(api_response)
                return
        logging.warning(f"Could not delete namespace: {os.getenv('NAMESPACE')} because is does not exist.")
    except client.exceptions.ApiException as e:
        print("Exception when calling CoreV1Api->delete_namespaced_pod: %s\n" % e)


def k8s_get_app_port() -> int:
    """
    Returns the node port of the created service.
    :return: node port
    """
    # init
    config.load_kube_config(config_file=kube_config)
    v1 = client.CoreV1Api()
    # iterate through namespaces services
    ret = v1.list_namespaced_service(namespace=os.getenv("NAMESPACE"))
    for i in ret.items:
        if os.getenv("UI") in i.metadata.name:
            # checks if node port is not NoneType
            if i.spec.ports[0].node_port:
                logging.debug(f"NodePort for service {i.metadata.name} is {i.spec.ports[0].node_port}")
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
        logging.info(list(build))
        logging.info(f"Build image with tag: {image_name}")
        return image_name
    except Exception as e:
        logging.error(f"Could not build image: {e}")


def set_prometheus_info() -> None:
    """
    Sets corresponding environment variable to NodePort of each Prometheus service instance.
    :return: None
    """
    # init
    print(kube_config)
    config.load_kube_config(config_file=kube_config)
    v1 = client.CoreV1Api()
    # iterate through default namespaces services
    ret_default = v1.list_namespaced_service(namespace="default")
    for service in ret_default.items:
        if "prometheus-kube-prometheus-prometheus" in service.metadata.name:
            # set env variable
            if os.getenv("PRODUCTION") == "True":
                port = service.spec.ports[0].port
                host = service.spec.cluster_ip
            else:
                port = service.spec.ports[0].node_port
                host = "localhost"
            set_key(dotenv_path=os.path.join(os.getcwd(), ".env"), key_to_set="PROMETHEUS_RESOURCES_HOST",
                    value_to_set=f"http://{host}:{port}")
            break
    # iterate through linkerd namespaces services
    ret_linkerd = v1.list_namespaced_service(namespace="linkerd")
    for service in ret_linkerd.items:
        if "linkerd-prometheus" in service.metadata.name:
            # set env variable
            if os.getenv("PRODUCTION") == "True":
                port = service.spec.ports[0].port
                host = service.spec.cluster_ip
            else:
                port = service.spec.ports[0].node_port
                host = "localhost"
            set_key(dotenv_path=os.path.join(os.getcwd(), ".env"), key_to_set="PROMETHEUS_NETWORK_HOST",
                    value_to_set=f"http://{host}:{port}")
            break
    ret_teastore = v1.list_namespaced_service(namespace="teastore")
    for service in ret_teastore.items:
        if "webui" in service.metadata.name:
            # set env variable
            if os.getenv("PRODUCTION") == "True":
                port = str(service.spec.ports[0].port)
                host = str(service.spec.cluster_ip)
            else:
                port = str(service.spec.ports[0].node_port)
                host = "localhost"
            set_key(dotenv_path=os.path.join(os.getcwd(), ".env"), key_to_set="HOST",
                    value_to_set=host)
            set_key(dotenv_path=os.path.join(os.getcwd(), ".env"), key_to_set="NODE_PORT",
                    value_to_set=port)
            break
    load_dotenv(override=True)
    logging.info(f"PROMETHEUS_RESOURCES_HOST: {os.getenv('PROMETHEUS_RESOURCES_HOST')}")
    logging.info(f"PROMETHEUS_NETWORK_HOST: {os.getenv('PROMETHEUS_NETWORK_HOST')}")
    logging.info(f"Teastore: http://{os.getenv('HOST')}:{os.getenv('NODE_PORT')}")


def get_resource_requests() -> dict:
    """
    Gets the resource request for every deployment in a given namespace.
    :return: resource requests
    """
    # init API
    config.load_kube_config(config_file=kube_config)
    apps_v1 = client.AppsV1Api()
    resource_requests = dict()
    # read deployment
    ret = apps_v1.list_namespaced_deployment(namespace=os.getenv("NAMESPACE"))
    for i in ret.items:
        deployment = apps_v1.read_namespaced_deployment(name=i.metadata.name, namespace=os.getenv("NAMESPACE"))
        resources = deployment.spec.template.spec.containers[0].resources.requests
        resource_requests[i.metadata.name] = resources
    return resource_requests


def create_autoscaler() -> None:
    """
    Deploys the autoscaler app with cron jobs for scaling and online learning.
    :return: None
    """
    work_directory = os.getcwd()
    try:
        yaml_path = os.path.join(os.getcwd(), "k8s")
        os.chdir(yaml_path)
        os.system(f"kubectl create -f autoscaler.yaml -n teastore")
        time.sleep(10)
        autoscaler_status = requests.get(f"http://{os.getenv('HOST')}:30050/heartbeat").json()
        while not bool(autoscaler_status["success"]):
            time.sleep(10)
        logging.info("Autoscaler is up and alive.")
        # os.system(f"kubectl create -f cronjob.yaml -n teastore")
        time.sleep(10)
        os.chdir(work_directory)
    except Exception as err:
        logging.error(f"Error while creating autoscaler: {err}")


def deploy_autoscaler_docker():
    client = docker.from_env()
    client.containers.run(image="angi2412/autoscaler", name="autoscaler", detach=True)


def delete_autoscaler_docker():
    client = docker.from_env()
    c = client.containers.get("autoscaler")
    c.remove(force=True)


if __name__ == '__main__':
    k8s_create_teastore()