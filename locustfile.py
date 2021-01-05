from locust import HttpUser, task
from dotenv import load_dotenv
import os


class User(HttpUser):
    # init
    load_dotenv()
    host = "{os.getenv('HOST')}:{os.getenv('PORT')}"
    route = os.getenv("ROUTE")
    testfile_path = os.path.join(os.getcwd(), f"{os.getenv('TESTFILE')}.txt")

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
