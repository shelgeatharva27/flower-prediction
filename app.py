import os
import numpy as np
import pickle
from flask import Flask, request, render_template, send_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Initialize Flask app
app = Flask(__name__)

# Allowed image formats
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif','webp'}

# Load model and class labels
model = load_model("flower_model.h5")  # Load trained model
with open("flower_labels.pkl", "rb") as f:
    class_names = pickle.load(f)  # Load class labels

def allowed_file(filename):
    """Check if the uploaded file has a valid image extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_flower(img_path):
    """Process and predict flower class from the given image."""
    try:
        # Load and preprocess the image
        img = Image.open(img_path)
        img = img.convert("RGB")  # Convert to RGB
        img = img.resize((224, 224))  # Resize for model input
        img_array = np.array(img) / 255.0  # Normalize pixel values
        img_array = np.expand_dims(img_array, axis=0)  # Convert to batch format

        # Predict with the model
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions)
        confidence = np.max(predictions)

        # Only return flower name if confidence is high
        if confidence > 0.60:
            return class_names[predicted_class]
        else:
            return "No Flower Detected"

    except Exception as e:
        print(f"🔴 Error processing image: {e}")  # Debugging
        return "Error processing image"

@app.route("/", methods=["GET", "POST"])
def upload_file():
    """Handle image upload and prediction."""
    if request.method == "POST":
        if "file" not in request.files:
            return "No file uploaded", 400

        file = request.files["file"]

        if file.filename == "":
            return "No selected file", 400

        if not allowed_file(file.filename):
            return "Invalid file format. Only PNG, JPG, JPEG, and GIF are allowed.", 400

        # Save the uploaded file
        filepath = os.path.join("static", file.filename)
        file.save(filepath)

        # Predict flower
        result = predict_flower(filepath)

        return render_template("index.html", filename=file.filename, result=result)

    return render_template("index.html", filename=None, result=None)

# Route to serve static files
@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('static', filename)

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
