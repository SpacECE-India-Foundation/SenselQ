from flask import send_from_directory
from flask_cors import CORS
from io import BytesIO
from PIL import Image
import time
import ana
from flask import Flask, render_template, request, jsonify, redirect, url_for, session, flash, Response
import os
import cv2
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
from bson.binary import Binary
from pymongo import MongoClient
import pickle
import numpy as np
import base64
from proj3 import proj3

app = Flask(__name__, static_folder='dist', static_url_path='/')
app.register_blueprint(proj3,url_prefix='/proj3')
app.secret_key = 'secret_key'
CORS(app)

print('downloading weights for yoloy3')
import requests

url = "https://github.com/patrick013/Object-Detection---Yolov3/raw/master/model/yolov3.weights"
filename = "yolov3.weights"

# Check if the file already exists
if os.path.exists(filename):
    print(f"{filename} already exists. Skipping download.")
else:
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(filename, "wb") as file:
            for chunk in response.iter_content(1024):
                file.write(chunk)
        print("Download completed successfully.")
    else:
        print("Failed to download the file. Status Code:", response.status_code)

load_dotenv()

# Set up the file upload folder
app.config['UPLOAD_FOLDER'] = 'uploads/'  
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif', 'webp', 'avif'} 

# MongoDB setup
try:
    MONGO_URL = os.getenv('MONGO_URL')
    if not MONGO_URL:
        raise ValueError("MongoDB URL is missing!")
    client = MongoClient(MONGO_URL)
    db = client['activity_planner']
    videos_collection = db['videos']
except Exception as e:
    print("MongoDB Connection Error:", str(e))
    exit(1)  # Exit if connection fails


# Load the pre-trained model
with open('rf_model.pkl', 'rb') as file:
    model = pickle.load(file)



#routes
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    
@app.route('/', methods=['GET'])
def home():
    return render_template('landingpage.html')
@app.route('/proj1', methods=['GET'])
def home1():
    return send_from_directory(app.static_folder, "proj1.html")

@app.route('/upload-image', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'GET':
        
        return jsonify({"message": "Send a POST request to upload an image."}), 200
    
    if request.method == 'POST':
        # Get image data from request
        image_data = request.json['image']
        image_data_bytes = base64.b64decode(image_data)
        image_bytes = BytesIO(image_data_bytes)
        result = ana.analyze_emotion_from_image(image_bytes)
        # Return the result
        print(result)
        return jsonify({
            "message": "Image uploaded and processed successfully!",
            "filename": filename,
            "result": result['dominant_emotion']
        }), 200

@app.route('/proj2')
def home2():
    return render_template('index.html')

@app.route('/proj2/predict', methods=['POST'])
def predict():
    try:
        # Parse input data
        input_data = [
            int(request.form['Olympiad_Participation']),
            int(request.form['Scholarship']),
            int(request.form['School']),
            int(request.form['Grasp_pow']),
            int(request.form['Career_sprt']),
            int(request.form['Act_sprt']),
            int(request.form['Fant_arts']),
            int(request.form['Won_arts']),
            int(request.form['Time_art']),
            int(request.form['age'])
        ]
        
        age = int(request.form['age'])
        session['age'] = age  # Save the age in the session
        print("Received Age:", age)

        # Determine age group
        if 4 <= age <= 5:
            age_group = "4-5"
        elif 6 <= age <= 8:
            age_group = "6-8"
        else:
            return jsonify({'error': 'Invalid age provided.'}), 400
        print("Determined Age Group:", age_group)

        # Predict hobby using the model
        input_data = np.array(input_data).reshape(1, -1)
        prediction = model.predict(input_data)
        #print("Model Prediction:", prediction)

        # Map prediction to hobby
        hobby_mapping = {0: 'Academics', 1: 'Arts', 2: 'Sports'}
        predicted_hobby = hobby_mapping[prediction[0]]
        #print("Predicted Hobby:", predicted_hobby)

        # Fetch videos for the predicted hobby
        videos = videos_collection.find_one({"hobby": predicted_hobby})
        if not videos:
            return jsonify({'error': f'No videos found for hobby: {predicted_hobby}.'}), 404
        #print("Videos Document Retrieved:", videos)

        # Filter videos by age group
        filtered_videos = [
            video for video in videos.get('videos', [])
            if video.get('age_group') == age_group
        ]

        # Convert binary image to Base64
        for video in filtered_videos:
            if 'image' in video:
                video['image'] = base64.b64encode(video['image']).decode('utf-8')

        return jsonify({
            'Predicted Hobby': predicted_hobby,
            'Videos': filtered_videos
        })

    except Exception as e:
        print("Error in /predict route:", str(e))  
        return jsonify({'error': str(e)}), 500

# Upload video and image page
@app.route('/proj2/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        try:
            age = int(request.form.get('age'))
            hobby = request.form.get('hobby')
            video_url = request.form.get('url')
            video_title = request.form.get('title')
            video_description = request.form.get('description')
            video_materials = request.form.get('materials')
            video_duration = request.form.get('duration')

            if 4 <= age <= 5:
                age_group = "4-5"
            elif 6 <= age <= 8:
                age_group = "6-8"
            else:
                flash("Invalid age group.", "danger")
                return redirect(url_for('upload'))
            
            # Handle the uploaded file
            image_file = request.files.get('file')
            if image_file and allowed_file(image_file.filename):
                # Secure the filename and save it to the upload folder
                filename = secure_filename(image_file.filename)
                image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                image_file.save(image_path)
                
                # Read the image as binary data
                with open(image_path, 'rb') as img_file:
                    image_binary = Binary(img_file.read())

                # Create video entry
                video_entry = {
                    "url": video_url,
                    "title": video_title,
                    "description": video_description,
                    "image": image_binary,  # Store image as binary
                    "materials": video_materials,
                    "duration": video_duration,
                    "age_group": age_group  
                }

                # Check if a document with the hobby already exists
                existing_document = videos_collection.find_one({"hobby": hobby})

                if existing_document:
                    # Append new video to the existing document's `videos` array
                    videos_collection.update_one(
                        {"hobby": hobby},
                        {"$push": {"videos": video_entry}}
                    )
                    flash("Video added to the existing hobby.", "success")
                else:
                    # Create a new document for the hobby
                    new_document = {
                        "hobby": hobby,
                        "videos": [video_entry]
                    }
                    videos_collection.insert_one(new_document)
                    flash("New hobby document created with the video.", "success")
                
                return redirect(url_for('upload'))

            else:
                flash("Invalid file type. Please upload an image.", "danger")
                return redirect(url_for('upload'))
        
        except Exception as e:
            flash(f"Error occurred: {str(e)}", "danger")
            return redirect(url_for('upload'))

    return render_template('upload.html')

# Videos page
@app.route('/proj2/videos')
def show_videos():
    hobby = request.args.get('hobby')  # Get the hobby from query parameters
    
    if not hobby:
        return "Hobby parameter is missing.", 400

    # Get age from session
    age = session.get('age')
    if not age:
        return "Age is not stored in session.", 400

    # Determine age group
    if 4 <= age <= 5:
        age_group = "4-5"
    elif 6 <= age <= 8:
        age_group = "6-8"
    else:
        return "Invalid age group.", 400
    print("Hobby:", hobby, "Age Group:", age_group)

    # Fetch videos from MongoDB
    document = videos_collection.find_one({"hobby": hobby})
    if not document:
        return f"No videos found for hobby: {hobby}.", 404

    # print("Fetched Document:", document)

    # Filter videos by age group
    videos = [
        video for video in document.get('videos', [])
        if video.get('age_group') == age_group
    ]
    if not videos:
        return f"No videos found for hobby: {hobby} and age group: {age_group}.", 404

    # Convert binary image to base64
    for video in videos:
        if 'image' in video:
            video['image'] = base64.b64encode(video['image']).decode('utf-8')

    return render_template("videos.html", hobby=hobby, age_group=age_group, videos=videos)

if __name__ == "__main__":
    app.run(debug=True)
