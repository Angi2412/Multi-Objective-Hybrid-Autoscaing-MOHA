import datetime as dt
import os

from dotenv import load_dotenv
from flask import Flask, jsonify

import benchmark
import formatting
import ml
from k8s_tools import k8s_update_deployment

app = Flask(__name__)


@app.route('/heartbeat')
def heartbeat():
    """
    Sends heartbeat if application is running.
    :return: heartbeat
    """
    return jsonify(success=True)


@app.route('/scale')
def scale():
    """
    Autoscaling loop.
    """
    load_dotenv()
    parameter_status, target_status = benchmark.get_status("webui")
    print(f"Parameter status: {parameter_status}")
    print(f"Target Status: {target_status}")
    if check_target_status(target_status):
        return jsonify(success=True)
    else:
        best_parameters = ml.get_best_parameters(cpu_limit=parameter_status[0], memory_limit=parameter_status[1],
                                                 number_of_pods=parameter_status[2], window=int(os.getenv("WINDOW")))
        k8s_update_deployment(os.getenv("SCALE_POD"), best_parameters[0], best_parameters[1],
                              best_parameters[2], replace=False)
        print(f"Best parameters:")
        print(f"cpu: {best_parameters[0]}m - memory: {best_parameters[1]}Mi - pods: {best_parameters[2]}")
        return jsonify(success=True)


@app.route('/improve')
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
    ml.train_for_all_targets(date, "svr")


def check_target_status(targets: list) -> bool:
    """
    Checks if the current microservice status is healthy.
    :param targets: current status
    :return: if healthy
    """
    load_dotenv()
    # check if average response time is healthy
    if targets[0] > float(os.getenv("MAX_RESPONSE")):
        print("Average response time not in bound.")
        return False
    # check if cpu usage is healthy
    elif not int(os.getenv("MIN_USAGE")) < targets[1] < int(os.getenv("MAX_USAGE")):
        print(f"CPU usage not in bound: {targets[1]}")
        return False
    # check if memory usage is healthy
    elif not int(os.getenv("MIN_USAGE")) < targets[2] < int(os.getenv("MAX_USAGE")):
        print(f"Memory usage not in bound: {targets[2]}")
        return False
    else:
        return True


if __name__ == '__main__':
    scale()
    # app.run(host="localhost", port=5000, debug=False)
