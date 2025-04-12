# 🍎 Fruit Classifier

This is a simple fruit image classification web app built with **Flask**, **OpenCV**, and machine learning models such as **Random Forest** and **Support Vector Machine (SVM)**.  
Users can upload fruit images, and the system predicts the fruit type using both models.

---

## ✨ Features

- Upload a fruit image through the web interface  
- Predict fruit type using two ML models: 🌳 Random Forest and 🤖 SVM  
- PCA and StandardScaler used for preprocessing  
- Clean and responsive interface for visualization  

---

## 📦 Model Files (store in `models/` directory)

- `random_forest_model.pkl` → Trained Random Forest classifier  
- `svm_model.pkl` → Trained SVM classifier  
- `scaler.pkl` → StandardScaler for feature normalization  
- `pca.pkl` → PCA for dimensionality reduction  
- `id_to_label_dict.pkl` → Mapping class IDs to readable labels  

---

## ⚙️ Installation

```bash
# 1. Clone the repository
git clone https://github.com/DucNhat03/fruit-classifier.git
cd fruit-classifier

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the application
python app.py
```
##📦 Requirements:
- Python 3.x
- Flask
- OpenCV (opencv-python)
- NumPy
- scikit-learn
- joblib

## 📸 Image Preprocessing Workflow:
- Resize the uploaded image to 45x45 pixels
- Convert image from BGR to RGB color format
- Flatten the image to a 1D vector
- Normalize pixel values using StandardScaler (scaler.pkl)
- Reduce dimensionality using PCA (pca.pkl)

## ✨ Example Output:
> Input:
![image](https://github.com/user-attachments/assets/4b34a54d-e632-4fe1-90b1-f16f1dd3afd4)
> Output:
![image](https://github.com/user-attachments/assets/f050f077-c631-42af-8553-c77cc4294728)


## 🙋‍♂️ Author: Duc Nhat
GitHub: https://github.com/DucNhat03
