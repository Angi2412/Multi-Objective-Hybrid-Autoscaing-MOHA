import math
import os

from dotenv import load_dotenv
from locust import LoadTestShape


class DoubleWave(LoadTestShape):
    """
    A shape to imitate some specific user behaviour. In this example, midday
    and evening meal times. First peak of users appear at time_limit/3 and
    second peak appears at 2*time_limit/3
    Settings:
        min_users -- minimum users
        peak_one_users -- users in first peak
        peak_two_users -- users in second peak
        time_limit -- total length of test
    """
    load_dotenv()
    min_users = int(os.getenv("SPAWN_RATE"))
    peak_one_users = int(os.getenv("LOAD")) / 2
    peak_two_users = int(os.getenv("LOAD"))
    time_limit = (int(os.getenv("HH")) * 60 * 60) + (int(os.getenv("MM")) * 60)

    def tick(self):
        run_time = round(self.get_run_time())

        if run_time < self.time_limit:
            user_count = (
                    (self.peak_one_users - self.min_users)
                    * math.e ** -(((run_time / (self.time_limit / 10 * 2 / 3)) - 5) ** 2)
                    + (self.peak_two_users - self.min_users)
                    * math.e ** -(((run_time / (self.time_limit / 10 * 2 / 3)) - 10) ** 2)
                    + self.min_users
            )
            return round(user_count), round(user_count)
        else:
            return None


class StagesShape(LoadTestShape):
    """
    A simply load test shape class that has different user and spawn_rate at
    different stages.
    Keyword arguments:
        stages -- A list of dicts, each representing a stage with the following keys:
            duration -- When this many seconds pass the test is advanced to the next stage
            users -- Total user count
            spawn_rate -- Number of users to start/stop per second
            stop -- A boolean that can stop that test at a specific stage
        stop_at_end -- Can be set to stop once all stages have run.
    """

    stages = [
        {"duration": 60, "users": 1, "spawn_rate": 1},
        {"duration": 100, "users": 3, "spawn_rate": 1},
        {"duration": 180, "users": 5, "spawn_rate": 1},
        {"duration": 220, "users": 3, "spawn_rate": 1},
        {"duration": 230, "users": 2, "spawn_rate": 1},
        {"duration": 240, "users": 1, "spawn_rate": 1},
    ]

    def tick(self):
        run_time = self.get_run_time()

        for stage in self.stages:
            if run_time < stage["duration"]:
                tick_data = (stage["users"], stage["spawn_rate"])
                return tick_data

        return None
