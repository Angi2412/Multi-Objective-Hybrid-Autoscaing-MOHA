from gevent import monkey

monkey.patch_all()
import os

import gevent
from dotenv import load_dotenv
from locust import HttpUser, task
from locust.env import Environment
from locust.log import setup_logging
from locust.stats import stats_history, StatsCSVFileWriter

setup_logging("INFO", None)


class User(HttpUser):
    # init
    load_dotenv(override=True)
    host = f"http://{os.getenv('HOST')}:{os.getenv('NODE_PORT')}/"
    route = os.getenv("ROUTE")
    testfile_path = os.path.join(os.getcwd(), "data", "loadtest", f"{os.getenv('TESTFILE')}.txt")

    @task
    def my_task(self):
        # read testfile as list
        testfile = open(self.testfile_path, "r")
        test_input = testfile.readlines()
        # call route for every test point
        for i in test_input:
            self.client.get(f"{self.route}/{int(i)}")

    @task
    def healthcheck(self):
        self.client.get("healthcheck")


def start_locust(iteration: int, folder: str):
    gevent.monkey.patch_all()
    # setup Environment and Runner
    env = Environment(user_classes=[User])
    env.create_local_runner()

    # CSV writer
    load_dotenv(override=True)
    stats_path = os.path.join(folder, f"locust_{iteration}")
    csv_writer = StatsCSVFileWriter(
        environment=env,
        base_filepath=stats_path,
        full_history=True,
        percentiles_to_report=[90.0, 95.0]
    )

    # start a WebUI instance
    env.create_web_ui(host="127.0.0.1", port=8089, stats_csv_writer=csv_writer)

    # start a greenlet that save current stats to history
    gevent.spawn(stats_history, env.runner)

    # spawn csv writer
    gevent.spawn(csv_writer)

    # start the test
    env.runner.start(user_count=int(os.getenv("USERS")), spawn_rate=int(os.getenv("SPAWN_RATE")))

    # stop the runner in a given time
    time_in_seconds = (int(os.getenv("HH")) * 60 * 60) + (int(os.getenv("MM")) * 60)
    gevent.spawn_later(time_in_seconds, lambda: env.runner.quit())

    # wait for the greenlets
    env.runner.greenlet.join()

    # stop the web server for good measures
    env.web_ui.stop()
