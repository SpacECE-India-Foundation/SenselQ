from deepface import DeepFace
from PIL import Image
import os
import json
import numpy as np
def analyze_emotion_from_image(image_bytes):
    """
    Analyze emotions from a single image using DeepFace.
    
    Args:
        image_path (str): Path to the image file.

    Returns:
        dict: A JSON-compatible dictionary containing the dominant emotion or error message.
    """
    image = Image.open(image_bytes)
    image=image.convert('RGB')
    image_np = np.array(image)
    try:
        # First, check for faces in the image
        analysis = DeepFace.analyze(
            img_path=image_np,
            actions=["emotion"],  # Only analyze emotions
            enforce_detection=True  # Ensure face detection is performed
        )

        # If no face is detected, the analysis will raise an exception
        if isinstance(analysis, list):  # Multiple faces detected
            emotions = [{"dominant_emotion": face.get('dominant_emotion', 'N/A')} for face in analysis]
        else:  # Single face detected
            emotions = {"dominant_emotion": analysis.get('dominant_emotion', 'N/A')}
        
        return emotions[0] if isinstance(emotions, list) else emotions

    except Exception as e:
        return {"status": "error", "message": f"No face detected or error: {str(e)}"}

# Example Usage
if __name__ == "__main__":
    # Define the image path
    image_path = "/home/chirag/Desktop/landing_page/uploads/image_1737725300.png"  # Replace with your image path

    # Check if the image file exists
    if not os.path.exists(image_path):
        print(json.dumps({"status": "error", "message": "Image file not found"}))
    else:
        # Call the function and print the result
        result = analyze_emotion_from_image(image_path)
        print(json.dumps(result, indent=4))
