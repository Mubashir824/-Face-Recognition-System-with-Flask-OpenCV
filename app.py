from flask import Flask, render_template, request
import sqlite3
import cv2
import numpy as np

app = Flask(__name__)
DB_FILE = 'database.db'

# Create users table if not exists
def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            age INTEGER NOT NULL,
            photo BLOB NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

# Capture image from webcam
def capture_image():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None

# Basic image comparison (histogram-based)
def compare_faces(img1, img2):
    hist1 = cv2.calcHist([cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)], [0], None, [256], [0, 256])
    cv2.normalize(hist1, hist1)
    cv2.normalize(hist2, hist2)
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['POST'])
def register():
    name = request.form['name']
    age = int(request.form['age'])

    frame = capture_image()
    if frame is None:
        return "Failed to capture image from webcam."

    _, buffer = cv2.imencode('.jpg', frame)
    img_binary = buffer.tobytes()

    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('INSERT INTO users (name, age, photo) VALUES (?, ?, ?)', (name, age, img_binary))
    conn.commit()
    conn.close()

    return f"User '{name}' registered successfully!"

@app.route('/detect', methods=['POST'])
def detect():
    input_img = capture_image()
    if input_img is None:
        return "Failed to capture image from webcam."

    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('SELECT name, age, photo FROM users')
    users = c.fetchall()
    conn.close()

    for name, age, photo_blob in users:
        db_img_arr = np.frombuffer(photo_blob, np.uint8)
        db_img = cv2.imdecode(db_img_arr, cv2.IMREAD_COLOR)

        score = compare_faces(input_img, db_img)
        
        if score > 0.9:
            # Show match image with name/age overlay
            display_img = input_img.copy()
            text = f"Name: {name}, Age: {age}"

            # Styling text
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            thickness = 2
            color = (0, 255, 0)
            size = cv2.getTextSize(text, font, font_scale, thickness)[0]

            # Position text at bottom center
            text_x = int((display_img.shape[1] - size[0]) / 2)
            text_y = display_img.shape[0] - 30

            cv2.putText(display_img, text, (text_x, text_y), font, font_scale, color, thickness)

            cv2.imshow("User Detected", display_img)
            cv2.waitKey(3000)  # Display for 3 seconds
            cv2.destroyAllWindows()

            return f"Match Foung   Name : {name}   Age : {age}."

    return "No matching face found."

if __name__ == '__main__':
    init_db()
    app.run(debug=True)
