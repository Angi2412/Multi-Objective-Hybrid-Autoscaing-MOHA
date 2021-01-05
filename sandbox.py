# Copyright (c) 2020 Angelina Horn
import json
import logging
import os
import platform
import subprocess
from pathlib import Path

import docker
import requests
from dotenv import load_dotenv, set_key
from kubernetes import client, config

# deployment
load_dotenv()
NAMESPACE = os.getenv("NAMESPACE")
DEPLOYMENT_NAME = os.getenv("DEPLOYMENT")
DESIRED_PORT = int(os.getenv("PORT"))
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
        list_files = subprocess.run(["kubectl", "port-forward", pod_name, f"{port}:{DESIRED_PORT}"])
        logging.info("The exit code was: %d" % list_files.returncode)
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
    locust_file = os.path.join(os.getcwd(), "data", "locustfile.py")
    response = subprocess.run(
        ["locust", "-f", locust_file, "--headless", "-u", users, "-r", hatch, "--run-time", f"{time_hh}h{time_mm}m"])
    logging.info(response)


def execute(name: str, port: int, docker_path: str, route: str, input_type: str, testfile_path: str) -> bool:
    """
    Executes series of methods in order to use the sandbox.
    :param name: desired name
    :param port: desired port
    :param docker_path: path to docker image
    :param route: api route
    :param input_type: type of input argument
    :param testfile_path: path to test file
    :return: if execution was successful
    """
    try:
        # image = build_image(name=name, docker_path=docker_path)
        # deploy_to_cluster(name=name, port=port, image=image)
        # forward_port(port=port)
        config_locust(host=HOST, route=route, port=DESIRED_PORT, testfile="test")
        start_locust(users=10, hatch=1, time_hh=0, time_mm=3)
        return True
    except Exception as e:
        logging.error(f"Error while executing: {e}")
        return False


def get_grafana_data() -> None:
    """
    Exports data from grafana to a json file.
    :return: None
    """
    export_dir = os.getenv("GRAFANA_DIR")
    headers = {'Authorization': f"Bearer {os.getenv('GRAFANA_API_KEY')}"}
    response = requests.get(f"{os.getenv('GRAFANA_HOST')}/api/search?query=&", headers=headers)
    response.raise_for_status()
    dashboards = response.json()

    if not os.path.exists(export_dir):
        os.makedirs(export_dir)

    for d in dashboards:
        print("Saving: " + d['title'])
        response = requests.get('%s/api/dashboards/%s' % (os.getenv("GRAFANA_HOST"), d['uri']), headers=headers)
        data = response.json()['dashboard']
        dash = json.dumps(data, sort_keys=True, indent=4, separators=(',', ': '))
        name = data['title'].replace(' ', '_').replace('/', '_').replace(':', '').replace('[', '').replace(']', '')
        tmp = open(export_dir + name + '.json', 'w')
        tmp.write(dash)
        tmp.write('\n')
        tmp.close()


if __name__ == '__main__':
    """
    Main method for test purposes.
    """
    #docker_path = os.path.join(os.getcwd(), "webservice", "Dockerfile")
    #testfile_path = os.path.join(os.getcwd(), "data", "test.txt")
    #execute(name="testmonday", port=5000, docker_path=docker_path, route="square", input_type="int",
            #testfile_path=testfile_path)
    get_grafana_data()
