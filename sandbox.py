# This will be the sandbox in which the first performance data will be collected.
import os
import logging
import yaml
from kubernetes import client, config

# deployment path
deployment_path = os.path.join(os.getcwd(), "deployment.yaml")


def deploy():
    # load config
    config.load_kube_config()
    # load deployment.yaml
    try:
        with open(deployment_path) as f:
            dep = yaml.safe_load(f)
    except FileNotFoundError:
        logging.error("Could not find config.")
    k8s_apps_v1 = client.AppsV1Api()
    # create namespaced deployment
    try:
        resp = k8s_apps_v1.create_namespaced_deployment(
            body=dep, namespace="default")
    except Exception as e:
        logging.error(f"Error while deploying: {e}")
    print("Deployment created. status='%s'" % resp.metadata.name)


def write_config(**kwargs):
    template = """
    apiVersion: v1
    kind: pod
    metadata:
      name: {name}
    spec:
      replicas: {replicas}
      template:
        metadata:
          labels:
            run: {name}
        spec:
          containers:
          - name: {name}
            image: {image}
            ports:
            - containerPort: {port}"""
    try:
        with open(deployment_path, 'w') as file:
            file.write(template.format(**kwargs))
    except IOError:
        logging.error("IOError while writing config.")


# test
if __name__ == '__main__':
    write_config(name="someName", image="myImg", replicas="many", port="1111")
