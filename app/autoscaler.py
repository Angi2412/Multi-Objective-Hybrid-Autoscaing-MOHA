import datetime as dt
import logging
import math
import os
import sched
import time

from dotenv import load_dotenv

import benchmark
import formatting
import ml
from k8s_tools import k8s_update_deployment, set_prometheus_info

s = sched.scheduler(time.time, time.sleep)
# init logger
logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')


def heartbeat():
    """
    Sends heartbeat if application is running.
    :return: heartbeat
    """
    return True


def scale():
    """
    Autoscaling loop.
    """
    load_dotenv()
    parameter_status, target_status = benchmark.get_status("webui")
    logging.info("Started autosclaing:")
    logging.info(f"Parameter status: {parameter_status} ")
    logging.info(f"Target Status: {target_status}")
    if check_target_status(target_status):
        logging.info("Target status is fine.")
        return
    else:
        # get scaling decision
        best_parameters = ml.get_best_parameters_hpa(cpu_limit=parameter_status[0], memory_limit=parameter_status[1],
                                                     number_of_pods=parameter_status[2], rps=parameter_status[3],
                                                     alg=os.getenv("ALGORITHM"),
                                                     response_time=target_status[2], cpu_usage=target_status[0],
                                                     memory_usage=target_status[1])
        # scale if not the same
        if best_parameters is not None:
            k8s_update_deployment(os.getenv("SCALE_POD"), int(best_parameters[3]), int(best_parameters[4]),
                                  int(best_parameters[5]), replace=False)
        else:
            logging.info("Did not scale.")
    logging.info("Finished Autoscaling.")


def improve() -> None:
    """
    Improves machine learning model in set period.
    :return: None
    """
    load_dotenv()
    date = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    # get Prometheus data from set period
    benchmark.get_prometheus_data(date, 0, int(os.getenv("PERIOD_HH")), int(os.getenv("PERIOD_MM")))
    # filter data
    formatting.filter_data(date)
    curr = formatting.get_filtered_data(date)
    prev = formatting.get_filtered_data(os.getenv("LAST_TRAINED_DATA"))
    # combine old and new data
    formatting.combine_data([prev, curr], date)
    # train all models
    ml.train_for_all_targets(os.getenv("ALGORITHM"))


def check_target_status(targets: list) -> bool:
    """
    Checks if the current microservice status is healthy.
    :param targets: current status
    :return: if healthy
    """
    load_dotenv()
    # check if average response time is healthy
    if targets[0] == 0.0 or targets[1] == 0.0 or targets[2] == 0.0:
        logging.error("Error while gathering status.")
        return True
    if targets[2] > float(os.getenv("TARGET_RESPONSE")):
        logging.info(f"Average response time not in bound: {targets[2]}")
        return False
    # check if cpu usage is healthy
    elif not int(os.getenv("MIN_USAGE")) < targets[0] < int(os.getenv("MAX_USAGE")):
        logging.info(f"CPU usage not in bound: {targets[0]}%")
        return False
    # check if memory usage is healthy
    elif not int(os.getenv("MIN_USAGE")) < targets[1] < int(os.getenv("MAX_USAGE")):
        logging.info(f"Memory usage not in bound: {targets[1]}%")
        return False
    else:
        return True


def scale_hpa():
    # current status
    parameter_status, target_status = benchmark.get_status("webui")
    # average usage
    avg_usage = ((int(os.getenv("MIN_USAGE")) + int(os.getenv("MAX_USAGE"))) / 2)
    logging.info("Started autosclaing:")
    logging.info(f"Parameter status: {parameter_status} ")
    logging.info(f"Target Status: {target_status}")

    # cpu usage
    cpu = math.ceil(parameter_status[2] * (target_status[0] / avg_usage))
    # memory usage
    memory = math.ceil(parameter_status[2] * (target_status[1] / avg_usage))
    # average response time
    pods = math.ceil(parameter_status[2] * (target_status[2] / int(os.getenv("TARGET_RESPONSE"))))
    # scale decision: max pods
    scale_pods = max(cpu, memory, pods)
    # cap pods
    if scale_pods > int(os.getenv("MAX_PODS")):
        scale_pods = int(os.getenv("MAX_PODS"))
    # scale if not the same
    logging.info(f"New pods: {scale_pods}")
    if scale_pods != parameter_status[2]:
        k8s_update_deployment(os.getenv("SCALE_POD"), int(parameter_status[0]), int(parameter_status[1]),
                              scale_pods, replace=False)
    logging.info("Finished Autoscaling.")


def autoscale(sc, hpa):
    if hpa:
        scale_hpa()
        s.enter(int(os.getenv("SCALING_TIME")), 1, autoscale, (sc, True,))
    else:
        scale()
        s.enter(int(os.getenv("SCALING_TIME")), 1, autoscale, (sc, False,))


if __name__ == '__main__':
    load_dotenv()
    set_prometheus_info()
    a = "False"
    if a == "True":
        s.enter(int(os.getenv("SCALING_TIME")), 1, autoscale, (s, True,))
    else:
        s.enter(int(os.getenv("SCALING_TIME")), 1, autoscale, (s, False,))
    s.run()
