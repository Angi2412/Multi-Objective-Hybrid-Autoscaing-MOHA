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

from flask import Flask, abort, request
import json

app = Flask(__name__)


@app.route("/fibonacci")
def fibonacci_endpoint():
    nVal = request.args.get("n")
    try:
        n = int(nVal)
        return str(fibonacci(n))
    except ValueError:
        abort(400, f"Value '{nVal}' is not an integer")


def fibonacci(n):
    # First Fibonacci number is 0 
    if n == 1:
        return 0
    # Second Fibonacci number is 1 
    elif n == 2:
        return 1
    else:
        return fibonacci(n - 1) + fibonacci(n - 2)


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)
