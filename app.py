from flask import Flask, render_template, request, redirect
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
import os

app = Flask(__name__)

face_cascade = cv2.CascadeClassifier("haarcascade.xml")


# --------- AUTO DETECT CAMERA ---------
def get_camera_index():
    for index in range(3):
        cam = cv2.VideoCapture(index)
        if cam.isOpened():
            cam.release()
            return index
    return None


# --------- HOME ---------
@app.route("/")
def home():
    return render_template("index.html")


# --------- REGISTER ---------
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form["name"]
        path = f"dataset/{name}"
        os.makedirs(path, exist_ok=True)

        cam_index = get_camera_index()

        if cam_index is None:
            return "Camera not detected"

        cam = cv2.VideoCapture(cam_index)
        count = 0

        while True:
            ret, frame = cam.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                count += 1
                cv2.imwrite(f"{path}/{count}.jpg", gray[y:y+h, x:x+w])
                cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

            cv2.imshow("Register - Press Enter to Stop", frame)

            if cv2.waitKey(1) == 13 or count >= 30:
                break

        cam.release()
        cv2.destroyAllWindows()

        return redirect("/")

    return render_template("register.html")


# --------- TRAIN MODEL ---------
@app.route("/train")
def train():
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    faces = []
    labels = []
    label_dict = {}
    current_label = 0

    for person in os.listdir("dataset"):
        person_path = os.path.join("dataset", person)

        if not os.path.isdir(person_path):
            continue

        label_dict[current_label] = person

        for image in os.listdir(person_path):
            img_path = os.path.join(person_path, image)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if img is not None:
                faces.append(img)
                labels.append(current_label)

        current_label += 1

    if len(faces) == 0:
        return "No images found in dataset"

    recognizer.train(faces, np.array(labels))
    recognizer.save("trainer.yml")
    np.save("labels.npy", label_dict)

    return redirect("/")


# --------- MARK ATTENDANCE ---------
def mark_attendance(name):
    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    time = now.strftime("%H:%M:%S")

    columns = ["Name", "Date", "Time"]

    if os.path.exists("attendance.xlsx"):
        df = pd.read_excel("attendance.xlsx")
        if list(df.columns) != columns:
            df = pd.DataFrame(columns=columns)
    else:
        df = pd.DataFrame(columns=columns)

    if not ((df["Name"] == name) & (df["Date"] == date)).any():
        new_row = pd.DataFrame([[name, date, time]], columns=columns)
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_excel("attendance.xlsx", index=False)


# --------- START ATTENDANCE ---------
@app.route("/start")
def start():

    if not os.path.exists("trainer.yml"):
        return "Please train model first"

    cam_index = get_camera_index()

    if cam_index is None:
        return "Camera not detected"

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("trainer.yml")
    labels = np.load("labels.npy", allow_pickle=True).item()

    cam = cv2.VideoCapture(cam_index)

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            id, confidence = recognizer.predict(gray[y:y+h, x:x+w])

            if confidence < 70:
                name = labels[id]
                mark_attendance(name)
                cv2.putText(frame, name, (x,y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        cv2.imshow("Attendance - Press Enter to Stop", frame)

        if cv2.waitKey(1) == 13:
            break

    cam.release()
    cv2.destroyAllWindows()

    return redirect("/")


# --------- VIEW ATTENDANCE ---------
@app.route("/attendance")
def attendance():
    if os.path.exists("attendance.xlsx"):
        df = pd.read_excel("attendance.xlsx")
        data = df.values.tolist()
    else:
        data = []

    return render_template("attendance.html", data=data)


if __name__ == "__main__":
    app.run(debug=True)