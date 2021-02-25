import os

from dotenv import load_dotenv
from flask import Flask, jsonify

import benchmark
import formatting
import k8s_tools
import ml
import datetime as dt

app = Flask(__name__)


@app.route("/heartbeat")
def heartbeat():
    return jsonify({"status": "healthy"})


@app.route('/live')
def live():
    load_dotenv()
    parameter_status, target_status = benchmark.get_status("webui")
    if check_target_status(target_status):
        pass
    else:
        best_parameters = ml.get_best_parameters(cpu_limit=parameter_status[0], memory_limit=parameter_status[1],
                                                 number_of_pods=parameter_status[2], window=5)
        k8s_tools.k8s_update_deployment(os.getenv("SCALE_POD"), best_parameters[0], best_parameters[1],
                                        best_parameters[2], replace=False)


@app.route('/improve')
def improve():
    date = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    benchmark.get_prometheus_data(date, 0)
    formatting.filter_data(date)
    ml.train_for_all_targets(date, "neural")


def check_target_status(targets: list) -> bool:
    if not int(os.getenv("MIN_RESPONSE")) < targets[0] < int(os.getenv("MAX_RESPONSE")):
        return False
    elif not int(os.getenv("MIN_USAGE")) < targets[1] < int(os.getenv("MAX_USAGE")):
        return False
    elif not int(os.getenv("MIN_USAGE")) < targets[2] < int(os.getenv("MAX_USAGE")):
        return False
    else:
        return True
