import flask

app = flask.Flask(__name__)


@app.route("/")
def hello():
    return flask.jsonify({
        "hello": "world"
    })


@app.route("/healthcheck")
def health():
    return flask.jsonify({
        "status": "ok"
    })


@app.route("/healthcheck/fail")
def health_fail():
    return flask.jsonify({
        "status": "error"
    }), 500


if __name__ == '__main__':
    app.run(host="0.0.0.0")