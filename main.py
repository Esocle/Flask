from tkinter import N
import flask
import numpy as np
import cv2
import io

import tflite_color

app = flask.Flask(__name__)

@app.route('/test')
def test():
    return flask.render_template('test.html')

@app.route('/outputs/<path:filename>')
def output(filename):
    return flask.send_from_directory('outputs', filename)
    # return flask.send_file('Quakka.jpg')


@app.route('/color_service', methods=['POST'])
def color_service():
    # file = flask.request.files['file']
    # nparr = np.frombuffer(file.read(), np.uint8)
    nparr = np.frombuffer(flask.request.data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    image = tflite_color.run(image)
    # image = image.astype(np.float32)
    # image += 20
    # image = np.clip(image, 0, 255).astype(np.uint8)
    _, buf = cv2.imencode('.jpg', image)
    return flask.send_file(io.BytesIO(buf), download_name='result.jpg', mimetype='image/jpeg')



app.run(debug=True)