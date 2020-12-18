# Copyright (c) 2020 Angelina Horn
import logging
import os
from pathlib import Path

import docker
from kubernetes_py import K8sConfig, K8sContainer, K8sDeployment, K8sService

# deployment path
deployment_path = os.path.join(os.getcwd(), "deployment.yaml")


def deploy_to_cluster(name: str, port: int, image: str) -> str:
    # load config
    # cfg_default = K8sConfig()
    cfg_cert = K8sConfig(
        kubeconfig=None,
        api_host="127.0.0.1:49153",
        cert=('C:/Users/Angi/.minikube/profiles/minikube/client.crt',
              'C:/Users/Angi/.minikube/profiles/minikube/client.key')
    )
    print(cfg_cert.api_host)
    try:
        # create container
        container = K8sContainer(name=name, image=image)
        container.add_port(
            container_port=port,
            host_port=port,
            name=name
        )
        # create deployment
        # deployment = K8sDeployment(
        #     config=cfg_cert,
        #     name=name,
        #     replicas=1
        # )
        # deployment.add_container(container)
        # deployment.create()
        # create service
        svc = K8sService(config=cfg_cert, name=name)
        svc.add_port(name=f"{name}-port", port=port, target_port=port, protocol="NodePort")
        svc.add_selector(selector=dict(name=name))
        svc.set_cluster_ip('192.168.1.100')
        svc.create()
        return "True"
    except Exception as e:
        logging.error(f"Error while deploying: {e}")


def build_image(name: str, docker_path: str) -> str:
    # init
    try:
        image = f"{name}:latest"
        directory_path = Path(docker_path).parent
        if Path(docker_path).exists() and directory_path.exists():
            pass
        else:
            raise FileNotFoundError
    except FileNotFoundError:
        logging.error("Could not find Dockerfile.")
    try:
        # docker_client = docker.APIClient(base_url='unix://var/run/docker.sock')
        docker_client = docker.APIClient(base_url='tcp://localhost:2375')
        # build image
        build = docker_client.build(path=str(directory_path), tag=name, dockerfile=docker_path)
        print(list(build))
        return image
    except docker.errors.BuildError as e:
        logging.error(f"Could not build image: {e}")
    except docker.errors.APIError as e:
        logging.error(f"Error while using Docker API: {e}")


def execute(name: str, port: int, docker_path: str, route: str, input_type: str, testfile_path: str) -> bool:
    try:
        image = build_image(name, docker_path)
        print(f"Image name: {image}")
        host = deploy_to_cluster(name=name, port=port, image=image)
        print(host)
        # benchmark.configBenchmark(host=host, route=route, input_type=input_type, testfile_path=testfile_path)
        # benchmark.startBenchmark()
        return True
    except Exception as e:
        logging.error(f"Error while executing: {e}")
        return False


if __name__ == '__main__':
    docker_path = os.path.join(os.getcwd(), "webservice", "Dockerfile")
    testfile_path = os.path.join(os.getcwd(), "data", "test.txt")
    execute(name="test", port=80, docker_path=docker_path, route="/healthcheck", input_type="",
            testfile_path=testfile_path)
