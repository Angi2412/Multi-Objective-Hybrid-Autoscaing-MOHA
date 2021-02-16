from gevent import monkey

monkey.patch_all()
import os

import gevent
from dotenv import load_dotenv
from locust.env import Environment
from locust.log import setup_logging
from locust.stats import stats_history, StatsCSVFileWriter
from data.loadtest.robotshop import UserBehavior

# init logger
setup_logging("INFO", None)


def start_locust(iteration: int, folder: str):
    load_dotenv(override=True)
    # setup Environment and Runner
    env = Environment(user_classes=[UserBehavior], host=f"http://{os.getenv('HOST')}:{os.getenv('NODE_PORT')}/")
    env.create_local_runner()

    # CSV writer
    stats_path = os.path.join(folder, f"locust_{iteration}")
    csv_writer = StatsCSVFileWriter(
        environment=env,
        base_filepath=stats_path,
        full_history=True,
        percentiles_to_report=[90.0, 50.0]
    )

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
