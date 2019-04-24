import json
import requests

from flask import Flask, render_template, request

app = Flask(__name__)


@app.route("/")
def index():
    return render_template('index.html')


@app.route("/query", methods=["POST"])
def query():
    data = request.json

    requests.post


    answers = process(question=data['question'])
    return json.dumps(answers)
