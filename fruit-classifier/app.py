from flask import Flask, render_template, request  # Nhập các module Flask cần thiết
import cv2  # Thư viện xử lý ảnh OpenCV
import numpy as np  # Thư viện tính toán số học
import joblib  # Dùng để load mô hình đã train
import os  # Thư viện thao tác với hệ thống file

app = Flask(__name__)  # Khởi tạo ứng dụng Flask
UPLOAD_FOLDER = 'static/uploads'  # Thư mục lưu ảnh người dùng upload
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER  # Cấu hình thư mục upload cho Flask

# Tạo thư mục upload nếu chưa tồn tại
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load các mô hình và công cụ tiền xử lý đã lưu
rf_model = joblib.load("models/random_forest_model.pkl")  # Load mô hình Random Forest
svm_model = joblib.load("models/svm_model.pkl")  # Load mô hình SVM
scaler = joblib.load("models/scaler.pkl")  # Load đối tượng chuẩn hóa dữ liệu
pca = joblib.load("models/pca.pkl")  # Load mô hình giảm chiều PCA
id_to_label_dict = joblib.load("models/id_to_label_dict.pkl")  # Load từ điển ánh xạ ID -> nhãn

# Hàm tiền xử lý ảnh đầu vào
def preprocess_image(image_path):
    image = cv2.imread(image_path)  # Đọc ảnh từ đường dẫn
    image = cv2.resize(image, (45, 45))  # Resize ảnh về kích thước cố định (45x45)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Chuyển ảnh từ BGR sang RGB
    image_flat = image.flatten().reshape(1, -1)  # Chuyển ảnh thành vector hàng 1 chiều
    image_scaled = scaler.transform(image_flat)  # Chuẩn hóa vector ảnh
    image_pca = pca.transform(image_scaled)  # Giảm chiều ảnh bằng PCA
    return image_pca  # Trả về ảnh đã được xử lý

# Route chính của trang web
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":  # Nếu người dùng gửi ảnh lên
        file = request.files["image"]  # Lấy file ảnh từ form
        if file:  # Kiểm tra có ảnh hay không
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)  # Tạo đường dẫn lưu ảnh
            file.save(filepath)  # Lưu ảnh vào thư mục upload

            image_pca = preprocess_image(filepath)  # Tiền xử lý ảnh

            rf_pred = rf_model.predict(image_pca)[0]  # Dự đoán với mô hình Random Forest
            svm_pred = svm_model.predict(image_pca)[0]  # Dự đoán với mô hình SVM

            # rf_label = id_to_label_dict[rf_pred]  # Lấy nhãn từ ID dự đoán Random Forest
            # svm_label = id_to_label_dict[svm_pred]  # Lấy nhãn từ ID dự đoán SVM

            rf_label = id_to_label_dict[rf_pred].split("\\")[-1]
            svm_label = id_to_label_dict[svm_pred].split("\\")[-1]


            # Trả kết quả về giao diện index.html cùng nhãn và ảnh đã upload
            return render_template("index.html", 
                                   rf_label=rf_label, 
                                   svm_label=svm_label,
                                   image_path=filepath)

    # Nếu truy cập lần đầu hoặc không upload ảnh
    return render_template("index.html")

# Chạy ứng dụng Flask ở chế độ debug
if __name__ == "__main__":
    app.run(debug=True)
