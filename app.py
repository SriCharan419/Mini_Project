from flask import Flask, render_template, request
import numpy as np
import cv2
from keras.models import load_model
import webbrowser

app = Flask(__name__)

# Disable file caching for development
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1

# Load pre-trained models and configuration files
haarcascade = "haarcascade_frontalface_default.xml"
label_map = ['Anger', 'Neutral', 'Fear', 'Happy', 'Sad', 'Surprise']
model = load_model('model.h5')
cascade = cv2.CascadeClassifier(haarcascade)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/capture_photo', methods=["POST"])
def capture_photo():
    # Fetch user input from the form
    user_name = request.form['name']
    language = request.form['language']
    
    found = False
    cap = cv2.VideoCapture(0)

    while not(found):
        # Capture frame from webcam
        _, frm = cap.read()
        gray = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = cascade.detectMultiScale(gray, 1.4, 1)

        for x, y, w, h in faces:
            found = True
            roi = gray[y:y + h, x:x + w]
            # Save the captured face to a static folder
            cv2.imwrite("static/face.jpg", roi)

    # Preprocess the detected face for prediction
    roi = cv2.resize(roi, (48, 48))
    roi = roi / 255.0
    roi = np.reshape(roi, (1, 48, 48, 1))

    # Predict the emotion
    prediction = model.predict(roi)
    prediction = np.argmax(prediction)
    emotion = label_map[prediction]

    cap.release()

    # Search YouTube for songs based on emotion and preferred language
    song_link = f"https://www.youtube.com/results?search_query={emotion}+{language}+songs"
    webbrowser.open(song_link)

    # Render the result page
    return render_template("emotion_detect.html", name=user_name, emotion=emotion, link=song_link)

if __name__ == "__main__":
    app.run(debug=True)
