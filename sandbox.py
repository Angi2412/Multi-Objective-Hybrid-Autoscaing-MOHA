# Copyright (c) 2020 Angelina Horn
import logging
import os
import platform
from pathlib import Path

import docker
from kubernetes import client, config

# deployment
DEPLOYMENT_NAME = "sandbox-deployment"
# init
logging.getLogger().setLevel(logging.INFO)


def deploy_to_cluster(name: str, port: int, image: str):
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
            namespace="default")
        print("Deployment created. status='%s'" % str(api_response.status))
    except Exception as e:
        logging.info(f"Error while deployment: {e}")


def delete_deployment(api_instance):
    # Delete deployment
    api_response = api_instance.delete_namespaced_deployment(
        name=DEPLOYMENT_NAME,
        namespace="default",
        body=client.V1DeleteOptions(
            propagation_policy='Foreground',
            grace_period_seconds=5))
    print("Deployment deleted. status='%s'" % str(api_response.status))


def build_image(name: str, docker_path: str) -> str:
    # init
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
    except docker.errors.BuildError as e:
        logging.error(f"Could not build image: {e}")
    except docker.errors.APIError as e:
        logging.error(f"Error while using Docker API: {e}")


def execute(name: str, port: int, docker_path: str, route: str, input_type: str, testfile_path: str) -> bool:
    try:
        #image = build_image(name, docker_path)
        #image = "testmonday:latest"
        #deploy_to_cluster(name=name, port=port, image=image)
        forward_port(name="sandbox-deployment-6ccf889d4c-qq4fc", port=5000)
        # print(host)
        # benchmark.configBenchmark(host=host, route=route, input_type=input_type, testfile_path=testfile_path)
        # benchmark.startBenchmark()
        return True
    except Exception as e:
        logging.error(f"Error while executing: {e}")
        return False


def forward_port(name: str, port: int):
    api_instance = client.CoreV1Api(client.ApiClient())
    api_port_response = api_instance.connect_get_namespaced_pod_portforward(name, "default", ports=port)
    print(api_port_response)


if __name__ == '__main__':
    docker_path = os.path.join(os.getcwd(), "webservice", "Dockerfile")
    testfile_path = os.path.join(os.getcwd(), "data", "test.txt")
    execute(name="testmonday", port=5000, docker_path=docker_path, route="/healthcheck", input_type="",
            testfile_path=testfile_path)
