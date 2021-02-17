from flask import Flask

app = Flask(__name__)

@app.route("/")
def index():
    return "here's your app ready to deploy"
