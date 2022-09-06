

from flask import Flask, render_template, Response
import cv2

app=Flask(__name__)
eye_cascPath = 'C:/Users/PQ-CO/Desktop/Closed-Eye-Detection-with-opencv-master/haarcascade_eye_tree_eyeglasses.xml'  #eye detect model
face_cascPath = 'C:/Users/PQ-CO/Desktop/Closed-Eye-Detection-with-opencv-master/haarcascade_frontalface_alt.xml'  #face detect model
faceCascade = cv2.CascadeClassifier(face_cascPath)
eyeCascade = cv2.CascadeClassifier(eye_cascPath)

cap = cv2.VideoCapture(0)


def gen_frames():  
    while True:
        ret, img = cap.read()
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

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
if __name__=='__main__':
    app.run(host = "0.0.0.0", port = 5000)

cap.release()
cv2.destroyAllWindows()     
