import tensorflow as tf
import numpy as np
import cv2
import threading
import urllib.request

from RoboCam.robocam import RoboCam # pip install RoboCam

# Initialize RoboCam
rcam = RoboCam()
rcam.CameraStreamInit()

model_filename = 'C://Users//admin//Documents//model//keras_model.h5'

model = tf.keras.models.load_model(model_filename, compile=False)

size = (224, 224)

def preprocessing(frame):
    frame_fliped = cv2.flip(frame, 1)
    frame_resized = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
    frame_normalized = (frame_resized.astype(np.float32) / 127.0) - 1
    frame_reshaped = frame_normalized.reshape((1, 224, 224, 3))
    return frame_reshaped

def predict(frame):
    prediction = model.predict(frame)
    return prediction

def fetch_stream(url):
    global frame, bytes, running

    bytes = b''
    frame = None

    try:
        stream = urllib.request.urlopen(url)
        while running:
            try:
                bytes += stream.read(1024)
                a = bytes.find(b'\xff\xd8')
                b = bytes.find(b'\xff\xd9')
                if a != -1 and b != -1:
                    jpg = bytes[a:b+2]
                    bytes = bytes[b+2:]
                    frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
            except Exception as e:
                print(f"Error in stream: {e}")
                break
        stream.close()
    except Exception as e:
        print(f"Could not open stream: {e}")

def process_frames():
    global frame, running

    while running:
        if frame is not None:
            preprocessed = preprocessing(frame)
            prediction = predict(preprocessed)
            print(prediction)
            # idx = np.argmax(prediction)
            if prediction[0, 0] > prediction[0, 1]:
                print('face in')
                cv2.putText(frame, 'in', (0, 25), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 100), thickness=2)
            else:
                cv2.putText(frame, 'out', (0, 25), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 100), thickness=2)
                print('face out')

            cv2.imshow("VideoFrame", frame)
            if cv2.waitKey(1) > 0:
                running = False

url = 'http://192.168.4.1:81/stream'
running = True

# Start the streaming thread
stream_thread = threading.Thread(target=fetch_stream, args=(url,))
stream_thread.start()

# Start the processing thread
process_thread = threading.Thread(target=process_frames)
process_thread.start()

# Wait for both threads to finish
stream_thread.join()
process_thread.join()

cv2.destroyAllWindows()
