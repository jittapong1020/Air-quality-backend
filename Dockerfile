# 1. ใช้ Python 3.9 เป็นฐาน
FROM python:3.9-slim

# 2. ตั้งค่าโฟลเดอร์ทำงานใน Docker
WORKDIR /app

# 3. ก๊อปปี้ไฟล์ requirements.txt เข้าไปก่อนเพื่อติดตั้ง Library
COPY requirements.txt .

# 4. ติดตั้ง Library ต่างๆ (รวม scikit-learn สำหรับไฟล์ .pkl ของคุณด้วย)
RUN pip install --no-cache-dir -r requirements.txt

# 5. ก๊อปปี้ไฟล์ทั้งหมด (main.py, rf_model.pkl) เข้าไปใน Docker
COPY . .

# 6. เปิด Port 10000 (Port มาตรฐานของ Render)
EXPOSE 10000

# 7. สั่งรัน Server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]