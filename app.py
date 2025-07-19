from flask import Flask, render_template, request, jsonify, redirect, session, url_for, flash
import cv2
import os
from datetime import datetime, timedelta
from PIL import Image
import numpy as np
import mysql.connector


app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Database connection
def get_database_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="Swarna@0902",
        database="Authorized_user"
    )

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['POST', 'GET'])
def login():
    user = request.form.get('user')
    if user == 'student':
        session['user'] = 'student'  
        return redirect('/student_dashboard')
    return redirect('/')

@app.route('/admin_dashboard')
def admin_dashboard():
    if session.get('user') != 'admin':
        return redirect(url_for('index'))
    return render_template('admin_dashboard.html')

@app.route('/admin_login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if username == 'admin' and password == 'admin123':
            session['user'] = 'admin'
            return redirect('/admin_dashboard')  
        else:
            return "Invalid credentials", 401
    return render_template('admin_login.html')
            
@app.route('/student_dashboard')
def student_dashboard():
    if session.get('user') != 'student':
        return redirect(url_for('index'))
    return render_template('student_dashboard.html')


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

@app.route('/generate_dataset', methods=['POST'])
def generate_dataset():
    if session.get('user') != 'admin':
        flash("Unauthorized access.")
        return redirect(url_for('index'))
    
    name = request.form.get('name')
    roll = request.form.get('roll')

    if not name or not roll:
        flash("Please enter all fields.")
        return redirect(url_for('admin_dashboard'))

    mydb = get_database_connection()
    mycursor = mydb.cursor()
    mycursor.execute("SELECT * FROM my_table")
    myresult = mycursor.fetchall()
    user_id = len(myresult) + 1

    sql = "INSERT INTO my_table(Id, Name, Roll) VALUES(%s, %s, %s)"
    val = (user_id, name, roll)
    mycursor.execute(sql, val)
    mydb.commit()

    face_classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    def face_cropped(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)
        if len(faces) == 0:
            return None
        for (x, y, w, h) in faces:
            return img[y:y+h, x:x+w]

    cap = cv2.VideoCapture(0)
    img_id = 0

    while True:
        ret, frame = cap.read()
        if face_cropped(frame) is not None:
            img_id += 1
            face = cv2.resize(face_cropped(frame), (200, 200))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            file_name_path = f"data/user.{user_id}.{img_id}.jpg"
            cv2.imwrite(file_name_path, face)
            if img_id == 50:
                break

    cap.release()
    cv2.destroyAllWindows()
    flash("Images Captured Successfully!")
    return redirect(url_for('admin_dashboard'))

@app.route('/train_classifier', methods=['POST'])
def train_classifier():
    if session.get('user') != 'admin':
        flash("Unauthorized access")
        return redirect(url_for('index'))

    data_dir = "data"
    path = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]
    faces = []
    ids = []

    for image in path:
        img = Image.open(image).convert('L')
        image_np = np.array(img, 'uint8')
        user_id = int(os.path.split(image)[1].split(".")[1])
        faces.append(image_np)
        ids.append(user_id)

    ids = np.array(ids)
    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.train(faces, ids)
    clf.write("classifier.xml")
    
    flash("Model trained successfully!")
    return redirect(url_for('admin_dashboard'))

@app.route('/recognize_faces', methods=['POST'])
def recognize_faces():
    if session.get('user') != 'student':
        return jsonify({"message": "Unauthorized"}), 403

    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.read("classifier.xml")

    mydb = get_database_connection()
    mycursor = mydb.cursor()

    cap = cv2.VideoCapture(0)
    start_time = datetime.now()
    end_time = start_time + timedelta(seconds=5)

    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            user_id, confidence = clf.predict(roi_gray)
            confidence = int(100 * (1 - confidence / 300))

            mycursor.execute("SELECT Name, Roll FROM my_table WHERE Id=%s", (user_id,))
            result = mycursor.fetchone()

            if result and confidence > 75:
                name, roll = result
                now = datetime.now()
                date = now.strftime("%Y-%m-%d")
                time = now.strftime("%H:%M:%S")

                mycursor.execute("SELECT * FROM attendance_table WHERE UserId = %s AND Date = %s", (user_id, date))
                attendance_exists = mycursor.fetchone()

                if attendance_exists:
                    flash(f"{name} has already marked attendance today.")
                    
                else:
                    sql = "INSERT INTO attendance_table(UserId, Name, Date, Time, Roll) VALUES (%s, %s, %s, %s, %s)"
                    val = (user_id, name, date, time, roll)
                    mycursor.execute(sql, val)
                    mydb.commit()
                    flash(f"Attendance logged for {name}.")
                return redirect(url_for('student_dashboard'))
        if datetime.now() > end_time:
            break

    cap.release()
    cv2.destroyAllWindows()
    return jsonify({"message": "No face recognized or confidence too low."})

if __name__ == '__main__':
    app.run(debug=True)
