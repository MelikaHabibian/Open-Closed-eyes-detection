from sys import stdout
from process import webopencv
import logging
from flask import Flask, render_template, Response, request, jsonify
from flask_socketio import SocketIO
from camera import Camera
from utils import base64_to_pil_image, pil_image_to_base64
import cv2

faceCascade = cv2.CascadeClassifier('Haarcascades/haarcascade_frontalface_alt.xml')
eyeCascade = cv2.CascadeClassifier('Haarcascades/haarcascade_eye_tree_eyeglasses.xml')
#----------------- Video Transmission ------------------------------#
app = Flask(__name__)
app.logger.addHandler(logging.StreamHandler(stdout))
app.config['DEBUG'] = True
socketio = SocketIO(app)
camera = Camera(webopencv())

#---------------- Video Transmission --------------------------------#


#---------------- Video Socket Connections --------------------------#
@socketio.on('input image', namespace='/test')
def test_message(input):
    input = input.split(",")[1]
    camera.enqueue_input(input)
    #camera.enqueue_input(base64_to_pil_image(input))


@socketio.on('connect', namespace='/test')
def test_connect():
    app.logger.info("client connected")


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')


def gen():
    """Video streaming generator function."""

    app.logger.info("starting to generate frames!")
    while True:
        img = camera.get_frame() #pil_image_to_base64(camera.get_frame())
        if ret:
            frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Detect faces in the image
            faces = faceCascade.detectMultiScale(
                frame,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                # flags = cv2.CV_HAAR_SCALE_IMAGE
            )
            # print("Found {0} faces!".format(len(faces)))
            if len(faces) > 0:
                # Draw a rectangle around the faces
                for (x, y, w, h) in faces:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                frame_tmp = img[faces[0][1]:faces[0][1] + faces[0][3], faces[0][0]:faces[0][0] + faces[0][2]:1, :]
                frame = frame[faces[0][1]:faces[0][1] + faces[0][3], faces[0][0]:faces[0][0] + faces[0][2]:1]
                eyes = eyeCascade.detectMultiScale(
                    frame,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30),
                    # flags = cv2.CV_HAAR_SCALE_IMAGE
                )
                if len(eyes) == 0:
                    status= 'closed eyes!!!'
                else:
                    status= 'open eyes!!!'
                frame_tmp = cv2.resize(frame_tmp, (1000, 1000), interpolation=cv2.INTER_LINEAR)
                cv2.putText(frame_tmp, status, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 2, cv2.LINE_4)
                
            try:
                ret, buffer = cv2.imencode('.jpg', cv2.flip(frame_tmp,1))
                frame_tmp = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_tmp + b'\r\n')
            except Exception as e:
                pass


@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    socketio.run(app)
