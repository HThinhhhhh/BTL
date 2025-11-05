import joblib
import pandas as pd
from fastapi import FastAPI
from starlette.responses import Response

# === BƯỚC 1: COPY CÁC THƯ VIỆN TIỀN XỬ LÝ VÀO ĐÂY ===
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# === TẢI DỮ LIỆU NLTK (chỉ cần chạy 1 lần) ===
# (Chúng ta đặt nó ở đây để đảm bảo server luôn có)
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

app = FastAPI()

# === TẢI MODEL VÀ VECTORIZER ===
try:
    tfidf_vec = joblib.load('tfidf_vectorizer.pkl')
    print("Tải tfidf_vectorizer.pkl thành công!")
except FileNotFoundError:
    tfidf_vec = None

try:
    model = joblib.load('model_logreg.pkl')
    print("Tải logistic_regression_model.pkl thành công!")
except FileNotFoundError:
    model = None


# === BƯỚC 2: COPY HÀM TIỀN XỬ LÝ VÀO ĐÂY ===
def preprocess_text(text):
    if pd.isna(text):  # (Giả sử bạn cũng import pandas as pd)
        return ""

    # 1. Chuyển thành chữ thường
    text = str(text).lower()

    # 2. Xóa URL và ký tự đặc biệt (chỉ giữ chữ cái và khoảng trắng)
    text = re.sub(r'http[s]?://\S+|www\.\S+', '', text)  # Xóa URL
    text = re.sub(r'[^\w\s]', '', text)  # Xóa icon, dấu câu, số

    # 3. Tokenize (tách từ)
    tokens = word_tokenize(text)

    # 4. Loại bỏ stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words and len(word) > 1]

    # 5. Lemmatization (đưa về dạng gốc)
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    # Ghép lại thành chuỗi
    return ' '.join(tokens)


# === BƯỚC 3: SỬA LẠI ENDPOINT ===
@app.get("/predict", response_class=Response)
def predict_sentiment(text: str):
    raw_text = text
    print(f"Nhận được văn bản thô: {raw_text}")

    # === THAY THẾ DÒNG TẠM THỜI BẰNG HÀM THẬT ===
    clean_text = preprocess_text(raw_text)
    print(f"Văn bản đã làm sạch: {clean_text}")

    # (Các bước còn lại giữ nguyên)
    vectorized_text = tfidf_vec.transform([clean_text])
    prediction = model.predict(vectorized_text)
    result = prediction[0]

    print(f"Dự đoán là: {result}")

    return Response(content=result, media_type="text/plain")