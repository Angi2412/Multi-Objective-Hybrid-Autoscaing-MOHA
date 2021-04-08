# Copyright 2020 Locust Pod Autoscaler Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from locust import HttpLocust, TaskSet, task, between

class UserBehavior(TaskSet):
    @task(1)
    def profile(self):
        self.client.get("/fibonacci?n=10")

class WebsiteUser(HttpLocust):
    task_set = UserBehavior
    wait_time = between(5, 9)