from flask import Flask, render_template, request
import cv2
import numpy as np
import joblib
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

#Tạo thư mục nếu chưa có
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load models
rf_model = joblib.load("models/random_forest_model.pkl")
svm_model = joblib.load("models/svm_model.pkl")
scaler = joblib.load("models/scaler.pkl")
pca = joblib.load("models/pca.pkl")
id_to_label_dict = joblib.load("models/id_to_label_dict.pkl")

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (45, 45))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_flat = image.flatten().reshape(1, -1)
    image_scaled = scaler.transform(image_flat)
    image_pca = pca.transform(image_scaled)
    return image_pca

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["image"]
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            image_pca = preprocess_image(filepath)

            rf_pred = rf_model.predict(image_pca)[0]
            svm_pred = svm_model.predict(image_pca)[0]

            rf_label = id_to_label_dict[rf_pred]
            svm_label = id_to_label_dict[svm_pred]

            return render_template("index.html", 
                                   rf_label=rf_label, 
                                   svm_label=svm_label,
                                   image_path=filepath)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
