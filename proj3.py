import base64
import cv2
import numpy as np
from flask import Flask, render_template, jsonify, request, Blueprint
from PIL import Image
import io

CONFIG_PATH = "yolov3.cfg"
WEIGHTS_PATH = "yolov3.weights"
CLASSES_PATH = "coco.names"

proj3 = Blueprint('proj3',__name__,template_folder='templates')

SAFE_OBJECTS = ["pen", "pencil", "toy", "book", "cup", "chair", "bed"]
HARMFUL_OBJECTS = ["knife", "scissors"]

def load_yolo_model():
    net = cv2.dnn.readNet(WEIGHTS_PATH, CONFIG_PATH)
    with open(CLASSES_PATH, "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    return net, classes, output_layers

net, classes, output_layers = load_yolo_model()

def detect_objects(frame):
    height, width, _ = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    highest_confidence = 0.4
    detected_object = "None"
    status = "Safe"
    best_box = None

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > highest_confidence:
                highest_confidence = confidence
                label = classes[class_id]
                if label not in SAFE_OBJECTS and label not in HARMFUL_OBJECTS:
                    break
                detected_object = label
                status = "Not Safe" if label in HARMFUL_OBJECTS else "Safe"

                center_x, center_y, w, h = (
                    int(detection[0] * width),
                    int(detection[1] * height),
                    int(detection[2] * width),
                    int(detection[3] * height),
                )
                best_box = (center_x, center_y, w, h)

    if best_box:
        x, y = int(best_box[0] - best_box[2] / 2), int(best_box[1] - best_box[3] / 2)
        color = (0, 0, 255) if status == "Not Safe" else (0, 255, 0)
        cv2.rectangle(frame, (x, y), (x + best_box[2], y + best_box[3]), color, 2)
        cv2.putText(frame, detected_object, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    _, buffer = cv2.imencode(".jpg", frame)
    encoded_image = base64.b64encode(buffer).decode("utf-8")

    return jsonify({"image": encoded_image, "detections": detected_object, "status": status})

@proj3.route("/")
def index():
    return render_template("index2.html")

@proj3.route("/upload_image", methods=["POST"])
def upload_image():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    image = Image.open(file.stream)
    frame = np.array(image)

    return detect_objects(frame)

if __name__ == "__main__":
    proj3.run(debug=True)
