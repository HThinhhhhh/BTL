import joblib  # Để tải mô hình đã lưu
from fastapi import FastAPI  # Thư viện API
from pydantic import BaseModel  # Để định nghĩa input đầu vào

# Khởi tạo ứng dụng FastAPI
app = FastAPI()

# 1. TẢI CÁC FILE ĐÃ LƯU
# Tải "bộ từ điển" TF-IDF
try:
    tfidf_vec = joblib.load('tfidf_vectorizer.pkl')
    print("Tải tfidf_vectorizer.pkl thành công!")
except FileNotFoundError:
    print("LỖI: Không tìm thấy file tfidf_vectorizer.pkl")
    tfidf_vec = None

# Tải "bộ não" Logistic Regression
try:
    model = joblib.load('model_logreg.pkl')
    print("Tải logistic_regression_model.pkl thành công!")
except FileNotFoundError:
    print("LỖI: Không tìm thấy file logistic_regression_model.pkl")
    model = None


# 2. ĐỊNH NGHĨA INPUT
# Định nghĩa kiểu dữ liệu mà API sẽ nhận vào
# Nó phải là một object JSON có 1 trường tên là "text"
class ReviewInput(BaseModel):
    text: str


# 3. TẠO ENDPOINT (ĐƯỜNG DẪN)
# Tạo một endpoint tên là /predict
# Nó sẽ nhận dữ liệu kiểu POST
@app.post("/predict")
def predict_sentiment(review_input: ReviewInput):
    # Lấy text thô từ input (ví dụ: "This product is amazing")
    raw_text = review_input.text
    print(f"Nhận được văn bản: {raw_text}")

    # LƯU Ý QUAN TRỌNG:
    # Bạn phải thực hiện LẠI bước Tiền xử lý (Preprocessing)
    # (chuyển chữ thường, xóa icon, lemmatize...)
    # cho `raw_text` giống hệt như cách bạn đã làm ở Bước 1.
    # (Đoạn code này chưa bao gồm hàm preprocessing, bạn cần tự thêm vào)

    # TẠM THỜI, ta giả sử `raw_text` đã sạch
    # (Trong thực tế bạn phải thêm hàm clean_text() ở đây)
    clean_text = raw_text.lower()  # VÍ DỤ TẠM

    # 1. Biến text sạch thành một list (vì TfidfVectorizer cần đầu vào là list)
    text_to_vectorize = [clean_text]

    # 2. Dùng TF-IDF đã tải để biến chữ thành SỐ
    # (Dùng .transform() chứ KHÔNG dùng .fit_transform())
    vectorized_text = tfidf_vec.transform(text_to_vectorize)

    # 3. Dùng MÔ HÌNH đã tải để DỰ ĐOÁN
    prediction = model.predict(vectorized_text)

    # prediction sẽ là một array (ví dụ: ['Pos']),
    # chúng ta lấy phần tử đầu tiên
    result = prediction[0]

    print(f"Dự đoán là: {result}")

    # 4. Trả kết quả về dạng JSON
    return {"sentiment": result}