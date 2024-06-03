from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
import base64
import os

app = Flask(__name__)

# Define the directory to save the processed images
output_dir = os.path.join(app.root_path, 'output_images')

# Ensure the directory exists, create it if it doesn't
os.makedirs(output_dir, exist_ok=True)

def non_max_suppression(boxes, overlapThresh):
    if len(boxes) == 0:
        return []

    pick = []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        suppress = [last]

        for pos in range(last):
            j = idxs[pos]
            xx1 = max(x1[i], x1[j])
            yy1 = max(y1[i], y1[j])
            xx2 = min(x2[i], x2[j])
            yy2 = min(y2[i], y2[j])

            w = max(0, xx2 - xx1 + 1)
            h = max(0, yy2 - yy1 + 1)

            overlap = float(w * h) / area[j]

            if overlap > overlapThresh:
                suppress.append(pos)

        idxs = np.delete(idxs, suppress)

    return boxes[pick]

@app.route('/')
def welcome():
    return render_template('welcome.html')

@app.route('/index')
def show_index():
    return render_template('index.html')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Receive image from the HTML form
        image_file = request.files['image']
        nparr = np.frombuffer(image_file.read(), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Convert the image to grayscale
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Load the face detection classifier
        face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # Perform face detection
        faces = face_classifier.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Convert faces to (x, y, w, h) format
        faces_formatted = np.array([[x, y, w, h] for (x, y, w, h) in faces])

        # Apply NMS
        faces_nms = non_max_suppression(faces_formatted, 0.3)

        # Drawing a Bounding Box for each face after NMS
        for (x, y, w, h) in faces_nms:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Save the processed image
        filename = os.path.join(output_dir, 'processed_image.jpg')
        cv2.imwrite(filename, img)

        # Encode the processed image
        with open(filename, "rb") as img_file:
            encoded_image = base64.b64encode(img_file.read()).decode('utf-8')

        # Return the base64 encoded image
        return jsonify({'image_data': encoded_image})
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
