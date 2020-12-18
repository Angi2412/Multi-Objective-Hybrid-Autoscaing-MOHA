# Copyright (c) 2020 Angelina Horn
import os

import gevent
from locust import HttpUser, task
from locust.env import Environment
from locust.log import setup_logging
from locust.stats import stats_printer, stats_history, StatsCSVFileWriter

setup_logging("INFO", None)


class User(HttpUser):
    # init
    host = "https://docs.locust.io"
    route = "/"
    testfile_path = ""
    input_type = "id"

    @task
    def my_task(self):
        # read testfile as list
        testfile = open(self.testfile_path, "r")
        test_input = testfile.readlines()
        # call route for every test point
        for i in test_input:
            self.client.get(f"{self.route}?{self.input_type}={i}")


def configBenchmark(host: str, route: str, input_type: str, testfile_path: str):
    User.host = host
    User.route = route
    User.testfile_path = testfile_path
    User.input_type = input_type


def startBenchmark():
    # setup Environment and Runner
    env = Environment(user_classes=[User])
    env.create_local_runner()

    # start a WebUI instance
    env.create_web_ui("127.0.0.1", 8089)

    # start a greenlet that periodically outputs the current stats
    gevent.spawn(stats_printer(env.stats))

    # start a greenlet that writes output to csv
    gevent.spawn(StatsCSVFileWriter(environment=env.stats, base_filepath=os.path.join(os.getcwd(), "data")))

    # start a greenlet that save current stats to history
    gevent.spawn(stats_history, env.runner)

    # start the test
    env.runner.start(1, spawn_rate=10)

    # in 60 seconds stop the runner
    gevent.spawn_later(60, lambda: env.runner.quit())

    # wait for the greenlets
    env.runner.greenlet.join()

    # stop the web server for good measures
    env.web_ui.stop()
