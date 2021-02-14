import os
import time

from kubernetes import client, config

import k8s_tools

# init
test_yaml_file = "teastore"
teastore_pods = 7


def test_k8s_delete_namespace():
    # init
    config.load_kube_config()
    v1 = client.CoreV1Api()
    # delete namespace
    k8s_tools.k8s_delete_namespace()
    time.sleep(60)
    # check if namespace is deleted
    ret = v1.list_namespace()
    assert os.getenv("NAMESPACE") not in ret.items()


def test_k8s_create_deployment_from_file():
    # init
    config.load_kube_config()
    v1 = client.CoreV1Api()
    counter = 0
    # create deployment
    k8s_tools.k8s_create_deployment_from_file(test_yaml_file)
    time.sleep(60)
    ret = v1.list_namespaced_pod(namespace=os.getenv("NAMESPACE"))
    for i in ret.items:
        if os.getenv("APP_NAME") in i.metadata.name:
            counter = counter + 1
    assert counter == teastore_pods
    k8s_tools.k8s_delete_namespace()
