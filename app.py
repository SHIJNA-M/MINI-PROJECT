

from flask import Flask, render_template, url_for, request
from spamDetection import detection

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():

    if request.method == 'POST':
        message = request.form['message']
        if message:
            prediction = detection.check(message)

    return render_template('result.html', prediction=prediction.lower())


if __name__ == '__main__':
    app.run()
